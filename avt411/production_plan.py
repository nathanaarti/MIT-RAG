"""The production plan (Checkpoint 7.1 starter Step 2, `my_production_plan()`), for this system."""
from __future__ import annotations

from typing import Any


def production_plan() -> dict[str, Any]:
    return {
        "defenses": [
            "Deterministic input screening before any model call: normalise (NFKC, zero-width, literal \\n), "
            "strip pasted source blocks, pasted e-mail header blocks (Lab 7.1) and forged role turns, drop "
            "directive sentences, block prompt/secret extraction and off-task writing requests, 2,000-char cap",
            "Lab 7.1 persona filter (gpt-5.4-nano, temperature 0, fail-open) invoked only when the deterministic "
            "guard sees a weak roleplay/persistence signal",
            "XML/nonce-structured prompts on BOTH answer paths: user text escaped (Lab 7.1) inside "
            "<user_question-nonce>, evidence inside <evidence-nonce>, hardened system prompts with a trust hierarchy",
            "Least-privilege agent: read-only tools, validated planner JSON, no repeated actions, step and token caps",
            "Evidence sanitiser at ingestion (scan + record) and at query time (redact text addressed to the AI)",
            "Output guard: canary/prompt-fingerprint leak check, URL allow-list, secret redaction, citation check",
            "Memory quarantine, a JSONL security-event log, and a per-request route/cost log",
        ],
        "cost_optimizations": [
            "Zero-LLM quick lookups: KG-computed authorship answers and curated FAQ answers from testInputs.json",
            "Scope gate declines off-domain questions without an LLM call",
            "Deterministic router: single-hop questions take a 1-call standard RAG path; the agent runs only for "
            "vague references, multi-paper, author+content and follow-up questions, or when the standard answer "
            "reports insufficient evidence (escalation seeded with the evidence already retrieved)",
            "Answer cache for repeated history-free questions",
            "Retrieval that scales: BM25 built from the stored chunks (startup 97 s -> 5 s), incremental ingestion, "
            "catalog shortlist in the planner prompt beyond 40 papers",
        ],
        "measurable_gains": [
            "LLM calls, agent runs, planner calls, tokens, billed $ and latency per question: agent_only (6.1) vs routed",
            "Pass rate on the 83 eval questions for both systems, with FAQ hits reported separately",
            "Security audit outcomes: unhardened 5.1 agent vs the final pipeline",
            "Route check: one probe per route, LLM calls per route",
        ],
        "residual_risks": [
            "A curated FAQ answer can be served for a near-paraphrase that asks a different fact (threshold + margin + "
            "distinctive-term check, calibrated to 0 false matches on 94 probes, but not proven impossible)",
            "Deterministic routing misroutes questions whose wording hides that they need several papers; the "
            "insufficiency escalation catches only answers that admit missing evidence",
            "Stealth corpus poisoning (a false 'erratum' with no imperative) is not detectable by any filter",
            "Paraphrased policy-disclosure requests and staged multi-turn attacks that avoid the guard's phrases",
            "The persona filter fails open: an outage silently removes that layer",
        ],
    }
