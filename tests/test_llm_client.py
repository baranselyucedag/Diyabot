from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import APIStatusError
from pydantic import BaseModel

from src.api.memory import llm_client


def _status_error(status: int) -> APIStatusError:
    return APIStatusError(
        message=f"status {status}",
        response=httpx.Response(
            status, request=httpx.Request("POST", "https://test.invalid")
        ),
        body=None,
    )


def test_retry_does_not_retry_401():
    calls = 0

    class Client:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    nonlocal calls
                    calls += 1
                    raise _status_error(401)

    with pytest.raises(APIStatusError):
        asyncio.run(llm_client._create_with_retry(Client()))
    assert calls == 1


def test_retry_retries_500_then_succeeds(monkeypatch):
    calls = 0
    monkeypatch.setitem(llm_client.MEMORY_CONFIG, "llm_max_retries", 3)
    monkeypatch.setitem(llm_client.MEMORY_CONFIG, "llm_retry_backoff_seconds", 0)

    class Client:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    nonlocal calls
                    calls += 1
                    if calls < 3:
                        raise _status_error(500)
                    return "ok"

    assert asyncio.run(llm_client._create_with_retry(Client())) == "ok"
    assert calls == 3


def _fake_response(payload: str):
    class Message:
        content = payload

    class Choice:
        message = Message()

    class Resp:
        choices = [Choice()]

    return Resp()


def test_json_schema_rejection_falls_back_to_json_object(monkeypatch):
    """400 + 'response_format' → retry'siz, bir kereye mahsus json_object denemesi."""
    monkeypatch.setitem(llm_client.MEMORY_CONFIG, "llm_max_retries", 3)
    formats_seen = []

    class Client:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    formats_seen.append(kwargs.get("response_format"))
                    if kwargs.get("response_format", {}).get("type") == "json_schema":
                        raise APIStatusError(
                            message="Invalid response_format: json_schema unsupported",
                            response=httpx.Response(
                                400,
                                request=httpx.Request("POST", "https://test.invalid"),
                            ),
                            body=None,
                        )
                    return _fake_response('{"a": 1}')

    class Schema(BaseModel):
        a: int

    monkeypatch.setattr(llm_client, "get_client", lambda base_url=None: Client())
    result = asyncio.run(llm_client.llm_call_json_schema("p", Schema))
    assert result.a == 1
    assert [f["type"] for f in formats_seen] == ["json_schema", "json_object"]


def test_plain_400_does_not_fallback(monkeypatch):
    """'response_format' geçmeyen 400 → fallback YOK, hata doğrudan fırlar."""
    calls = 0

    class Client:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    nonlocal calls
                    calls += 1
                    raise APIStatusError(
                        message="context length exceeded",
                        response=httpx.Response(
                            400, request=httpx.Request("POST", "https://test.invalid")
                        ),
                        body=None,
                    )

    class Schema(BaseModel):
        a: int

    monkeypatch.setattr(llm_client, "get_client", lambda base_url=None: Client())
    with pytest.raises(APIStatusError):
        asyncio.run(llm_client.llm_call_json_schema("p", Schema))
    assert calls == 1
