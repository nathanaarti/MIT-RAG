"""Model clients. The only place chat and embedding clients are built, so the model policy
(config.assert_model_allowed) and the provider settings apply to every call in the app."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from . import config


def make_llm(model: str, *, max_tokens: int | None = None, temperature: float | None = None) -> ChatOpenAI:
    config.assert_model_allowed(model)
    kwargs: dict[str, Any] = {"timeout": config.LLM_TIMEOUT_S, "max_retries": config.LLM_MAX_RETRIES}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(model=model, api_key=config.require_api_key(), base_url=config.OPENROUTER_BASE_URL, **kwargs)


class CachedEmbeddings(Embeddings):
    """Wraps an embedding model with a small LRU cache on embed_query and a call counter.

    One user request embeds the same text several times (FAQ match, scope gate, vector
    search, agent searches); the cache makes that one API call. The counter lets evaluation
    report embedding calls next to LLM calls."""

    def __init__(self, inner: Embeddings, max_items: int = 2048):
        self.inner = inner
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.max_items = max_items
        self.query_calls = 0
        self.document_calls = 0

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            hit = self._cache.get(text)
            if hit is not None:
                self._cache.move_to_end(text)
                return hit
        vec = self.inner.embed_query(text)
        with self._lock:
            self.query_calls += 1
            self._cache[text] = vec
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.document_calls += 1
        return self.inner.embed_documents(texts)


_EMBEDDINGS: CachedEmbeddings | None = None
_EMB_LOCK = threading.Lock()


def get_embeddings() -> CachedEmbeddings:
    global _EMBEDDINGS
    with _EMB_LOCK:
        if _EMBEDDINGS is None:
            config.assert_model_allowed(config.EMBEDDING_MODEL)
            _EMBEDDINGS = CachedEmbeddings(OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL, api_key=config.require_api_key(),
                base_url=config.OPENROUTER_BASE_URL,
                check_embedding_ctx_length=False,  # OpenRouter needs raw text, not pre-tokenized input
            ))
    return _EMBEDDINGS
