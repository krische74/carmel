"""Index and retrieve documentation chunks (Ollama embeddings + SQLite)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from src.config import PROJECT_ROOT

if TYPE_CHECKING:
    from pathlib import Path

    from src.ai.llm_client import OllamaClient
    from src.config import Settings
    from src.data.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

CHUNK_TARGET = 1200
CHUNK_OVERLAP = 100
TEXT_GLOBS = ("*.md", "*.txt")


def _allowed_roots(settings: Settings) -> list[Path]:
    """Resolve configured paths; only files under ``PROJECT_ROOT`` are allowed."""
    root = PROJECT_ROOT.resolve()
    out: list[Path] = []
    for raw in settings.rag.index_paths:
        p = (root / str(raw).strip()).resolve()
        try:
            p.relative_to(root)
        except ValueError:
            logger.warning("RAG path outside project root ignored: %s", raw)
            continue
        out.append(p)
    return out


def _collect_files(settings: Settings) -> list[Path]:
    files: list[Path] = []
    for base in _allowed_roots(settings):
        if base.is_file():
            files.append(base)
        elif base.is_dir():
            for pat in TEXT_GLOBS:
                files.extend(base.rglob(pat))
    # de-dupe, stable order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for f in sorted(files, key=lambda x: str(x)):
        if f.is_file() and f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def chunk_text(
    text: str, *, target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping character windows (paragraph-aware when possible)."""
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return [text[:target]] if len(text) > target else [text]

    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= target:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= target:
                buf = p
            else:
                # hard-split long paragraph
                start = 0
                while start < len(p):
                    end = min(start + target, len(p))
                    chunks.append(p[start:end])
                    start = max(end - overlap, start + 1)
                buf = ""
        while len(buf) > target:
            chunks.append(buf[:target])
            buf = buf[max(target - overlap, 1) :]
    if buf:
        chunks.append(buf)
    return chunks


def _rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def index_docs(settings: Settings, sqlite_store: SQLiteStore, client: OllamaClient) -> int:
    """Embed and upsert all whitelisted files. Returns number of chunks written."""
    if not settings.llm.enabled:
        logger.info("RAG index skipped: llm.enabled is false")
        return 0
    model = (settings.rag.embedding_model or "").strip() or settings.llm.model
    indexed_at = datetime.now(UTC).isoformat()
    n = 0
    for fpath in _collect_files(settings):
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("RAG could not read %s: %s", fpath, exc)
            continue
        rel = _rel_path(fpath)
        parts = chunk_text(text)
        for i, chunk in enumerate(parts):
            emb = client.embed(chunk, model=model)
            if emb is None:
                logger.warning("RAG embed failed for %s chunk %s", rel, i)
                continue
            sqlite_store.upsert_rag_chunk(
                source_path=rel,
                chunk_index=i,
                text=chunk,
                embedding=emb,
                indexed_at=indexed_at,
            )
            n += 1
    logger.info("RAG index complete: %s chunks", n)
    return n


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def retrieve(
    query: str,
    settings: Settings,
    client: OllamaClient,
    sqlite_store: SQLiteStore,
    *,
    top_k: int | None = None,
) -> list[str]:
    """Return top ``top_k`` chunk texts by embedding cosine similarity."""
    if not query.strip():
        return []
    k = top_k if top_k is not None else int(settings.rag.top_k)
    k = max(1, min(k, 50))
    model = (settings.rag.embedding_model or "").strip() or settings.llm.model
    qemb = client.embed(query.strip(), model=model)
    if qemb is None:
        return []
    rows = sqlite_store.load_rag_chunks()
    if not rows:
        return []
    scored: list[tuple[float, str]] = []
    for row in rows:
        emb = row.get("embedding") or []
        if not isinstance(emb, list) or not emb:
            continue
        sim = _cosine(qemb, emb)
        scored.append((sim, str(row.get("text", ""))))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:k] if t.strip()]


def retrieve_context(
    query: str,
    settings: Settings,
    client: OllamaClient,
    sqlite_store: SQLiteStore,
) -> list[str]:
    """Convenience wrapper respecting ``settings.rag.enabled``."""
    if not settings.rag.enabled or not settings.llm.enabled:
        return []
    return retrieve(query, settings, client, sqlite_store)
