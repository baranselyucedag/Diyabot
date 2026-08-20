"""LLM İstemcisi — NVIDIA OpenAI uyumlu API ile JSON Schema çıktısı (async).

Memory maintenance pipeline'ı asenkron çalışır (conversation_lock altında);
bu yüzden LLM çağrıları da async'tir (AsyncOpenAI).
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from src.api.env import load_project_env
from src.api.memory.config import MEMORY_CONFIG

# .env yükle
load_project_env()

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")


def get_api_key() -> str:
    """NVIDIA_API_KEY'i frontend/.env (veya ortam) üzerinden okur."""
    key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "NVIDIA_API_KEY tanımlı değil. "
            "Key'i frontend/.env dosyasına yazın (https://build.nvidia.com — VITE_ öneki KULLANMA)."
        )
    return key


@lru_cache(maxsize=4)
def _get_client_cached(base_url: str, api_key: str) -> AsyncOpenAI:
    """AsyncOpenAI client örneği — aynı (base_url, api_key) için tekrar kullanılır."""
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def get_client(base_url: str = DEFAULT_BASE_URL) -> AsyncOpenAI:
    """OpenAI uyumlu async client döndürür (connection pooling için önbellekte tutar)."""
    return _get_client_cached(base_url, get_api_key())


async def _create_with_retry(client: AsyncOpenAI, **kwargs):
    """LLM isteğini üstel geri çekilmeli retry ile gönderir.

    Geçici ağ/sunucu hatalarında belirtilen sayıda tekrar dener; denemeler
    arasında `llm_retry_backoff_seconds * 2^attempt` kadar bekler. Tüm denemeler
    başarısız olursa RuntimeError fırlatır.
    """
    max_retries = MEMORY_CONFIG["llm_max_retries"]
    backoff = MEMORY_CONFIG["llm_retry_backoff_seconds"]

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(**kwargs)
        except Exception as exc:  # ağ/API geçici hatası — yeniden dene
            last_exc = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff * (2 ** attempt))

    raise RuntimeError(
        f"LLM çağrısı {max_retries} denemeden sonra başarısız: {last_exc}"
    ) from last_exc


async def llm_call_json_schema(
    prompt: str,
    schema_model: Type[T],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    system_prompt: str = "",
) -> T:
    """
    LLM'i JSON Schema modunda çağırır, sonucu Pydantic modeline parse eder.

    Args:
        prompt: User prompt (veya tam prompt)
        schema_model: Çıktı için Pydantic model sınıfı
        model: Model adı
        base_url: API base URL
        temperature: Sıcaklık (0.0 = deterministik)
        max_tokens: Maksimum token
        system_prompt: İsteğe bağlı system prompt

    Returns:
        schema_model örneği (validate edilmiş)

    Raises:
        RuntimeError: API hatası veya parse hatası
    """
    client = get_client(base_url)

    schema = schema_model.model_json_schema()
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_model.__name__,
            "strict": True,
            "schema": schema,
        },
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    timeout = MEMORY_CONFIG["llm_request_timeout_seconds"]
    base_kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
                "force_nonempty_content": True,
            }
        },
    )

    try:
        resp = await _create_with_retry(
            client, response_format=response_format, **base_kwargs
        )
    except RuntimeError:
        # Fallback: response_format=json_schema desteklenmiyorsa json_object dene
        resp = await _create_with_retry(
            client, response_format={"type": "json_object"}, **base_kwargs
        )

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("LLM boş cevap döndü.")

    try:
        return schema_model.model_validate_json(content)
    except Exception as exc:
        # Son çare: plain json parse etmeye çalış
        import json
        try:
            parsed = json.loads(content)
            return schema_model.model_validate(parsed)
        except Exception as exc2:
            raise RuntimeError(f"JSON Schema parse hatası: {exc} | fallback de başarısız: {exc2}") from exc2
