"""AI / LLM portfolio Q&A endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.data.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ai"])


class AskRequest(BaseModel):
    """User question payload."""

    question: str = Field(..., min_length=1, max_length=500)
    include_docs: bool = Field(
        default=False,
        description="When true, prepend top RAG chunks from indexed docs (requires rag.enabled).",
    )


class AskResponse(BaseModel):
    """LLM answer payload."""

    answer: str
    llm_available: bool


def _get_sqlite_store(request: Request) -> SQLiteStore:
    """SQLite from app lifespan."""
    return request.app.state.sqlite_store


@router.post(
    "/ai/ask",
    summary="Ask about your portfolio",
    description="Submit a plain-English question and receive an LLM-generated answer based on portfolio data.",
    response_model=AskResponse,
)
def ask_question(
    body: AskRequest,
    request: Request,
    sqlite: SQLiteStore = Depends(_get_sqlite_store),
) -> AskResponse:
    """Build ``PortfolioQA`` from app state and answer the question."""
    settings = request.app.state.settings
    if not settings.llm.enabled:
        raise HTTPException(status_code=503, detail="LLM is disabled in configuration")

    from src.ai.llm_client import OllamaClient
    from src.ai.portfolio_qa import PortfolioQA

    llm_client = OllamaClient(settings.llm)
    parquet = request.app.state.parquet_store
    ledger = getattr(request.app.state, "lot_ledger", None)

    qa = PortfolioQA(
        client=llm_client,
        sqlite_store=sqlite,
        parquet_store=parquet,
        lot_ledger=ledger,
        settings=settings,
    )
    available = llm_client.is_available()
    answer = qa.ask(body.question, include_docs=body.include_docs)
    return AskResponse(answer=answer, llm_available=available)
