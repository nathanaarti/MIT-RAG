"""Central configuration for the AVT-411 Research Paper Navigator.

Every tunable lives here and can be overridden with an environment variable (or a line in
`.env`), so nothing else in the package hard-codes a path, a model id, or a threshold.
Paths default to locations next to the project root, never to a machine-specific path.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()  # also honour a .env in the current working directory


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ─── Paths ─────────────────────────────────────────────────────────────────────
PAPERS_DIR = _path("AVT411_PAPERS_DIR", "Research Papers")
CHROMA_DIR = _path("AVT411_CHROMA_DIR", "chroma_db_papers")
CHROMA_COLLECTION = os.getenv("AVT411_CHROMA_COLLECTION", "langchain")  # name used by the 3.1 index
KG_PATH = _path("AVT411_KG_PATH", "avt411_knowledge_graph.json")
KG_SUPPLEMENT_PATH = _path("AVT411_KG_SUPPLEMENT_PATH", "data/kg_supplement.json")
FAQ_PATH = _path("AVT411_FAQ_PATH", "data/faq.json")
FAQ_EMBEDDINGS_PATH = _path("AVT411_FAQ_EMBEDDINGS_PATH", "data/faq_embeddings.json")
TEST_INPUTS = _path("AVT411_TEST_INPUTS", "testInputs.json")
TEST_INPUTS_INCOMPLETE = _path("AVT411_TEST_INPUTS_INCOMPLETE", "testInputs_incomplete.json")
FAQ_PARAPHRASES = _path("AVT411_FAQ_PARAPHRASES", "tests/data/faq_paraphrases.json")
OUTPUT_DIR = _path("AVT411_OUTPUT_DIR", "results")
LOG_DIR = _path("AVT411_LOG_DIR", "logs")
REQUEST_LOG = LOG_DIR / "requests.jsonl"
SECURITY_LOG = LOG_DIR / "security_events.jsonl"

# ─── Provider + models (OpenRouter, OpenAI-compatible API) ─────────────────────
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY_ENV = "OPENROUTER_API_KEY"
ANSWER_MODEL = os.getenv("AVT411_ANSWER_MODEL", "openai/gpt-5.4-mini")
PLANNER_MODEL = os.getenv("AVT411_PLANNER_MODEL", "openai/gpt-5.4-mini")
FILTER_MODEL = os.getenv("AVT411_FILTER_MODEL", "openai/gpt-5.4-nano")      # Lab 7.1 persona filter
METADATA_MODEL = os.getenv("AVT411_METADATA_MODEL", "openai/gpt-5.4-mini")  # ingestion-time title/author extraction
JUDGE_MODEL = os.getenv("AVT411_JUDGE_MODEL", "openai/gpt-5.4-mini")
EMBEDDING_MODEL = os.getenv("AVT411_EMBEDDING_MODEL", "openai/text-embedding-3-small")
LLM_TIMEOUT_S = _float("AVT411_LLM_TIMEOUT_S", 120)
LLM_MAX_RETRIES = _int("AVT411_LLM_MAX_RETRIES", 2)
ANSWER_MAX_TOKENS = _int("AVT411_ANSWER_MAX_TOKENS", 1500)
PLANNER_MAX_TOKENS = _int("AVT411_PLANNER_MAX_TOKENS", 600)

# Used only if OpenRouter omits usage.cost (USD per million input/output tokens).
FALLBACK_PRICE_PER_MTOK = {
    "openai/gpt-4o-mini": (0.15, 0.60), "openai/gpt-5.4-nano": (0.20, 1.25),
    "openai/gpt-5.4-mini": (0.75, 4.50), "openai/gpt-5.4": (2.50, 15.00),
}

# ─── Ingestion + retrieval ─────────────────────────────────────────────────────
CHUNK_SIZE = _int("AVT411_CHUNK_SIZE", 1500)          # characters (unchanged from Checkpoint 3.1)
CHUNK_OVERLAP = _int("AVT411_CHUNK_OVERLAP", 200)
EMBED_BATCH = _int("AVT411_EMBED_BATCH", 128)         # chunks per embedding request at ingestion
NUM_RETRIEVED = _int("AVT411_NUM_RETRIEVED", 4)       # chunks per search (agent tool + 3.1 baseline)
STANDARD_NUM_RETRIEVED = _int("AVT411_STANDARD_NUM_RETRIEVED", 6)  # chunks for the 1-call standard path
CANDIDATE_POOL = _int("AVT411_CANDIDATE_POOL", 30)    # per-retriever candidates before fusion (was 10)
WEIGHT_BM25 = _float("AVT411_WEIGHT_BM25", 0.5)
WEIGHT_VECTOR = _float("AVT411_WEIGHT_VECTOR", 0.5)

# ─── Agent ─────────────────────────────────────────────────────────────────────
AGENT_MAX_STEPS = _int("AVT411_AGENT_MAX_STEPS", 4)
CATALOG_FULL_LIMIT = _int("AVT411_CATALOG_FULL_LIMIT", 40)   # above this, the planner sees a shortlist
CATALOG_SHORTLIST = _int("AVT411_CATALOG_SHORTLIST", 25)

# ─── Screening ─────────────────────────────────────────────────────────────────
MAX_INPUT_CHARS = _int("AVT411_MAX_INPUT_CHARS", 2000)
# "off" | "on_signal" (only when the deterministic guard sees a weak signal) | "always" (Lab 7.1)
PERSONA_FILTER_MODE = os.getenv("AVT411_PERSONA_FILTER", "on_signal")
# Out-of-scope gate: best cosine similarity between the query and any indexed chunk.
# Calibrated by `faq-eval`: 83 in-domain eval questions >= 0.243, 8 off-domain probes <= 0.208.
SCOPE_MIN_SIMILARITY = _float("AVT411_SCOPE_MIN_SIMILARITY", 0.22)

# ─── FAQ / cache ───────────────────────────────────────────────────────────────
FAQ_ENABLED = _bool("AVT411_FAQ_ENABLED", True)
# Calibrated by `faq-eval`: 0 wrong/false matches over 94 should-miss probes; the closest
# should-miss probe without a distinctive-term mismatch scored 0.757, so 0.80 keeps headroom.
FAQ_MIN_SIMILARITY = _float("AVT411_FAQ_MIN_SIMILARITY", 0.80)
FAQ_MIN_MARGIN = _float("AVT411_FAQ_MIN_MARGIN", 0.05)
FAQ_DISTINCTIVE_DF = _float("AVT411_FAQ_DISTINCTIVE_DF", 0.02)  # a term in <2% of chunks is "distinctive"
CACHE_ENABLED = _bool("AVT411_CACHE_ENABLED", True)
CACHE_TTL_S = _float("AVT411_CACHE_TTL_S", 6 * 3600)
CACHE_MAX_ITEMS = _int("AVT411_CACHE_MAX_ITEMS", 512)


# ─── Model policy: no models from Chinese developers, anywhere in the stack ───
# Checked every time a chat or embedding client is constructed (llm.py), so a model id
# supplied through an environment variable or a CLI flag cannot bypass it.
_BLOCKED_VENDORS = {
    "qwen", "alibaba", "alibaba-nlp", "deepseek", "deepseek-ai", "z-ai", "zhipuai", "thudm", "moonshotai",
    "minimax", "baichuan", "baichuan-inc", "01-ai", "baidu", "tencent", "bytedance", "bytedance-seed",
    "stepfun", "stepfun-ai", "internlm", "shanghai-ai-lab", "baai", "iflytek", "sensetime", "xiaomi",
    "meituan", "kwaipilot", "inclusionai", "thudm", "openbmb", "infly", "arcee-ai-cn",
}
_BLOCKED_MODEL_TOKENS = re.compile(
    r"(qwen|qwq|deepseek|\bglm\b|glm-|chatglm|kimi|moonshot|minimax|abab|baichuan|\byi-|ernie|hunyuan|doubao|"
    r"seed-oss|step-\d|internlm|intern-|\bbge\b|bge-|gte-|ling-|ring-|mimo)", re.I)


class ModelPolicyError(ValueError):
    pass


def assert_model_allowed(model_id: str) -> str:
    """Raise ModelPolicyError for a model from a Chinese developer. Returns the id unchanged."""
    vendor = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
    if vendor in _BLOCKED_VENDORS or _BLOCKED_MODEL_TOKENS.search(model_id):
        raise ModelPolicyError(f"model {model_id!r} is not allowed by the deployment model policy")
    return model_id


def configured_models() -> dict[str, str]:
    return {"answer": ANSWER_MODEL, "planner": PLANNER_MODEL, "filter": FILTER_MODEL,
            "metadata": METADATA_MODEL, "judge": JUDGE_MODEL, "embedding": EMBEDDING_MODEL}


def require_api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise SystemExit(
            f"\n[setup] {API_KEY_ENV} is not set.\n"
            f"  Put a line '{API_KEY_ENV}=sk-or-...' in a .env file in {PROJECT_ROOT}\n"
            "  or export it in your shell, then rerun.\n")
    return key
