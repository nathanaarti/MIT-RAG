"""AVT-411 Research Paper Navigator — a screened, cost-routed RAG application over the AVT-411 papers.

Request flow: input screening -> answer cache -> KG authorship tier -> curated FAQ -> scope
gate -> deterministic router -> standard RAG (1 LLM call) or agentic RAG -> output guard.
See README.md. Entry point: `python -m avt411 -h`.
"""
