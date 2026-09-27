"""AI and explanation utilities (template + optional LLM via Ollama)."""

from __future__ import annotations

from src.ai.explainer import (
    enrich_signal_rationale,
    generate_cycle_summary,
    generate_signal_explanation,
    generate_weekly_digest,
)
from src.ai.llm_client import OllamaClient
from src.ai.llm_explainer import (
    generate_llm_cycle_summary,
    generate_llm_explanation,
    generate_llm_weekly_digest,
)
from src.ai.portfolio_qa import PortfolioQA

__all__ = [
    "OllamaClient",
    "PortfolioQA",
    "enrich_signal_rationale",
    "generate_cycle_summary",
    "generate_llm_cycle_summary",
    "generate_llm_explanation",
    "generate_llm_weekly_digest",
    "generate_signal_explanation",
    "generate_weekly_digest",
]
