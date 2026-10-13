"""Per-request LLM accounting by role (Checkpoint 6.1 CostMeter).

A LangChain callback records every chat-model call made inside a `metered()` block: role
(from a `role:<name>` tag), model, input/output/cached/reasoning tokens, the dollar cost
OpenRouter billed (`usage.cost`), and latency. Roles used: planner, answer, filter (Lab 7.1
persona filter), metadata (ingestion), judge (evaluation only; excluded from totals).
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

from . import config


class CostMeter(BaseCallbackHandler):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: dict[Any, tuple[float, str]] = {}
        self.calls: list[dict] = []
        self.errors = 0

    def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, **kwargs) -> None:
        role = next((t[5:] for t in (tags or []) if t.startswith("role:")), "other")
        with self._lock:
            self._open[run_id] = (time.perf_counter(), role)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        with self._lock:
            t0, role = self._open.pop(run_id, (time.perf_counter(), "other"))
        msg = getattr(response.generations[0][0], "message", None)
        usage = getattr(msg, "usage_metadata", None) or {}
        meta = getattr(msg, "response_metadata", None) or {}
        token_usage = meta.get("token_usage") or (response.llm_output or {}).get("token_usage") or {}
        model = meta.get("model_name") or (response.llm_output or {}).get("model_name") or "?"
        inp, out = int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)
        cost = token_usage.get("cost")
        estimated = cost is None
        if estimated:
            key = next((k for k in config.FALLBACK_PRICE_PER_MTOK if model.startswith(k)), None)
            p_in, p_out = config.FALLBACK_PRICE_PER_MTOK.get(key, (0.0, 0.0))
            cost = inp / 1e6 * p_in + out / 1e6 * p_out
        rec = {"role": role, "model": model, "input": inp, "output": out,
               "cached": int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0),
               "reasoning": int((usage.get("output_token_details") or {}).get("reasoning", 0) or 0),
               "cost": float(cost or 0.0), "estimated": estimated, "latency": time.perf_counter() - t0}
        with self._lock:
            self.calls.append(rec)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        with self._lock:
            self._open.pop(run_id, None)
            self.errors += 1

    def totals(self, exclude: tuple[str, ...] = ("judge",)) -> dict[str, Any]:
        rows = [c for c in self.calls if c["role"] not in exclude]
        t: dict[str, Any] = {"calls": len(rows), "input": 0, "output": 0, "cached": 0, "reasoning": 0, "cost": 0.0}
        for c in rows:
            for k in ("input", "output", "cached", "reasoning", "cost"):
                t[k] += c[k]
            r = c["role"]
            t[f"{r}_calls"] = t.get(f"{r}_calls", 0) + 1
            t[f"{r}_input"] = t.get(f"{r}_input", 0) + c["input"]
            t[f"{r}_output"] = t.get(f"{r}_output", 0) + c["output"]
            t[f"{r}_cost"] = t.get(f"{r}_cost", 0.0) + c["cost"]
        t["estimated"] = any(c["estimated"] for c in rows)
        return t


_METER: ContextVar[CostMeter | None] = ContextVar("avt411_cost_meter", default=None)
register_configure_hook(_METER, inheritable=True)


@contextmanager
def metered():
    meter = CostMeter()
    token = _METER.set(meter)
    try:
        yield meter
    finally:
        _METER.reset(token)
