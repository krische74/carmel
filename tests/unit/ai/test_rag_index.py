"""Document RAG chunking and retrieval (embeddings mocked)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

from src.ai.rag_index import chunk_text, index_docs, retrieve
from src.config import LLMConfig, RAGConfig, Settings
from src.data.storage.sqlite_store import SQLiteStore


def test_chunk_text_splits_and_overlaps() -> None:
    body = "A" * 500 + "\n\n" + "B" * 500 + "\n\n" + "C" * 500
    chunks = chunk_text(body, target=400, overlap=50)
    assert len(chunks) >= 2


def test_retrieve_returns_expected_chunk_order(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "h.db")
    store.upsert_rag_chunk(
        source_path="docs/x.md",
        chunk_index=0,
        text="alpha unique token xyz",
        embedding=[1.0, 0.0, 0.0],
        indexed_at="2026-01-01T00:00:00+00:00",
    )
    store.upsert_rag_chunk(
        source_path="docs/y.md",
        chunk_index=0,
        text="beta other",
        embedding=[0.0, 1.0, 0.0],
        indexed_at="2026-01-01T00:00:00+00:00",
    )
    client = MagicMock()
    client.embed.return_value = [0.9, 0.1, 0.0]
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        llm=LLMConfig(enabled=True),
        rag=RAGConfig(enabled=True, top_k=1),
    )
    out = retrieve("query about alpha", s, client, store, top_k=1)
    assert out and "alpha" in out[0]


def test_retrieve_empty_index_returns_empty(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "e.db")
    client = MagicMock()
    client.embed.return_value = [1.0, 0.0]
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        llm=LLMConfig(enabled=True),
        rag=RAGConfig(enabled=True),
    )
    assert retrieve("q", s, client, store) == []


def test_index_docs_skips_when_llm_disabled(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "d.db")
    client = MagicMock()
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        llm=LLMConfig(enabled=False),
        rag=RAGConfig(enabled=True, index_paths=["AGENTS.md"]),
    )
    n = index_docs(s, store, client)
    assert n == 0
    client.embed.assert_not_called()
