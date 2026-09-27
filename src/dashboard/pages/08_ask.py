"""Ask Hub: natural-language portfolio Q&A powered by local Ollama."""

from __future__ import annotations

import logging

import streamlit as st

from src.config import get_settings
from src.dashboard.auth import require_auth
from src.dashboard.paths import hub_sqlite_path
from src.data.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

if not require_auth(get_settings()):
    st.stop()

st.title("Ask Hub")
st.caption(
    "Ask questions about your portfolio in plain English. Powered by a local LLM (Ollama).",
)

API_URL = "http://localhost:8000/api/ai/ask"


def _ask_via_api(question: str) -> str | None:
    """Try the REST endpoint first (works when the API server is running)."""
    import httpx

    try:
        r = httpx.post(API_URL, json={"question": question}, timeout=60.0)
        if r.status_code == 200:
            return r.json().get("answer")
        if r.status_code == 503:
            return None
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("Ask Hub API request failed: %s", exc)
    return None


def _ask_direct(question: str) -> str:
    """Fall back to building PortfolioQA directly from stores."""
    from src.ai.llm_client import OllamaClient
    from src.ai.portfolio_qa import PortfolioQA
    from src.config import get_settings
    from src.dashboard.paths import hub_sqlite_path, parquet_root
    from src.data.storage.parquet_store import ParquetStore
    from src.data.storage.sqlite_store import SQLiteStore
    from src.portfolio.tax_lots import LotLedger

    settings = get_settings()
    if not settings.llm.enabled:
        return "LLM is not enabled in configuration. Set `llm.enabled: true` in settings.yaml."

    client = OllamaClient(settings.llm)
    sqlite = SQLiteStore(hub_sqlite_path())
    parquet = ParquetStore(parquet_root())
    ledger = LotLedger(hub_sqlite_path())
    qa = PortfolioQA(client=client, sqlite_store=sqlite, parquet_store=parquet, lot_ledger=ledger)
    return qa.ask(question)


_sqlite_ask = SQLiteStore(hub_sqlite_path())
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
    _rows = _sqlite_ask.get_chat_history(limit=50, account_id=None)
    for row in _rows:
        st.session_state.qa_history.append((str(row["question"]), str(row["answer"])))

question = st.text_input("Your question:", placeholder="e.g. How is SPY performing this month?")
ask_btn = st.button("Ask")

if ask_btn and question.strip():
    with st.spinner("Thinking..."):
        try:
            answer = _ask_via_api(question)
            if answer is None:
                answer = _ask_direct(question)
        except ValueError as exc:
            msg = f"Configuration error: {exc}"
            st.error(msg)
            logger.warning("Ask Hub configuration error: %s", exc)
            answer = msg
        except (OSError, TimeoutError, ConnectionError) as exc:
            msg = f"LLM service unavailable: {exc}"
            st.error(msg)
            logger.warning("Ask Hub LLM/service error: %s", exc)
            answer = msg
        except Exception as exc:
            msg = f"Unexpected error: {exc}"
            st.error(msg)
            logger.exception("Ask Hub unexpected error")
            answer = msg
        st.session_state.qa_history.append((question, answer))
        try:
            _sqlite_ask.log_chat(question, answer, account_id="default")
        except OSError as exc:
            logger.warning("Ask Hub: could not persist chat row: %s", exc)

if st.session_state.qa_history:
    st.divider()
    for q, a in reversed(st.session_state.qa_history[-5:]):
        st.markdown(f"**Q:** {q}")
        st.markdown(a)
        st.divider()
