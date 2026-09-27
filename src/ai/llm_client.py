"""Ollama HTTP client with health check and graceful fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from src.config import LLMConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Thin wrapper around the Ollama REST API (``/api/generate``).

    All external calls are wrapped so that failures return ``None``
    instead of raising. This keeps LLM features optional.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(float(config.timeout_seconds)),
        )

    @property
    def enabled(self) -> bool:
        """Whether LLM features are turned on in config."""
        return self._config.enabled

    def is_available(self) -> bool:
        """Return ``True`` when Ollama responds to a lightweight probe."""
        if not self._config.enabled:
            return False
        try:
            r = self._client.get("/api/tags", timeout=5.0)
            return r.status_code == 200
        except (httpx.HTTPError, OSError, ValueError):
            return False

    def generate(self, prompt: str, system: str | None = None) -> str | None:
        """Send a completion request and return the response text, or ``None`` on failure."""
        if not self._config.enabled:
            logger.debug("LLM disabled; skipping generate call")
            return None
        payload: dict = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self._config.max_tokens,
                "temperature": self._config.temperature,
            },
        }
        if system is not None:
            payload["system"] = system
        logger.debug(
            "Ollama generate: model=%s prompt_len=%d",
            self._config.model,
            len(prompt),
        )
        try:
            r = self._client.post("/api/generate", json=payload)
            r.raise_for_status()
            body = r.json()
            text = str(body.get("response", "")).strip()
            return text if text else None
        except (httpx.HTTPError, OSError, ValueError) as exc:
            logger.warning("Ollama generate failed: %s", exc)
            return None

    def embed(self, text: str, *, model: str | None = None) -> list[float] | None:
        """Call ``POST /api/embeddings``; return vector or ``None`` on failure."""
        if not self._config.enabled:
            logger.debug("LLM disabled; skipping embed call")
            return None
        m = (model or "").strip() or self._config.model
        payload = {"model": m, "prompt": text}
        logger.debug("Ollama embed: model=%s text_len=%d", m, len(text))
        try:
            r = self._client.post("/api/embeddings", json=payload)
            r.raise_for_status()
            body = r.json()
            emb = body.get("embedding")
            if isinstance(emb, list) and emb:
                return [float(x) for x in emb]
            return None
        except (httpx.HTTPError, OSError, ValueError) as exc:
            logger.warning("Ollama embed failed: %s", exc)
            return None
