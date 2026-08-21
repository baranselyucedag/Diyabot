"""FastAPI — POST /chat (frontend sözleşmesi)."""

from __future__ import annotations

import os
import uuid
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from src.api.env import load_project_env

# frontend/.env → NVIDIA_API_KEY vb.
load_project_env()

from src.api.pipeline import run_chat
from src.api.memory import storage as memory_storage
from src.api.memory.metrics import render_metrics

app = FastAPI(
    title="Tip-2 Diyabet Chatbot API",
    version="0.1.0",
    description="RAG: bge-m3 retrieve → mmarco rerank → NVIDIA Nemotron LLM",
)

# Vite sık port değiştirir (5173, 5174, …) — lokal için regex
_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_extra = os.getenv("CORS_ORIGINS", "")
if _extra.strip():
    _origins.extend([o.strip() for o in _extra.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    history: list[dict] = Field(default_factory=list)


class SourceOut(BaseModel):
    document: str
    section: str
    section_label: Optional[str] = None
    snippet: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    triage_level: Literal["GREEN", "YELLOW", "RED", "REFUSE", "EMERGENCY"]
    sources: list[SourceOut]
    disclaimer: str
    follow_ups: list[str] = []


@app.get("/health")
def health() -> Any:
    """Canlılık + memory-ready kontrolü.

    - ``data/`` dizinine yazma izni var mı (probe dosyası yaz/sil).
    - memory modülü sağlıklı yüklenmiş mi (``get_conversation_lock`` çağrılabilir).

    Sorun varsa 503 + degraded; yoksa 200 + ok.
    """
    problems: list[str] = []

    try:
        data_dir = memory_storage.DATA_DIR
        probe = data_dir / ".health_probe"
        data_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("probe", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        problems.append(f"data yazılamıyor: {exc}")

    try:
        if memory_storage.get_conversation_lock("health_probe") is None:
            problems.append("get_conversation_lock None döndü")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"memory modülü sağlıksız: {exc}")

    if problems:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "memory_ready": False,
                "reason": "; ".join(problems),
            },
        )
    return {"status": "ok", "memory_ready": True}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """Prometheus text formatında in-memory sayaçları döndürür."""
    return PlainTextResponse(render_metrics(), media_type="text/plain")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> dict[str, Any]:
    """Frontend'in beklediği chat endpoint'i."""
    conversation_id = req.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    try:
        return await run_chat(req.message, conversation_id, req.history)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"İndeks eksik: {exc}. Önce: python -m src.retrieval.embed build",
        ) from exc
    except RuntimeError as exc:
        # örn. NVIDIA_API_KEY yok
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Chat hatası: {exc}") from exc


def main() -> None:
    """uvicorn src.api.app:app --reload --port 8000"""
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
