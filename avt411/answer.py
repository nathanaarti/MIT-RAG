"""The grounded answer step, shared by the standard RAG path and the agent.

This is Checkpoint 6.1's answer node, extracted so both paths get the same protections:
nonce-tagged ("spotlighted") data blocks with the user's text XML-escaped (Lab 7.1), the
hardened system prompt with a canary, and the output guard.
"""
from __future__ import annotations

import uuid

from . import guardrails as guard
from . import prompts
from .knowledge_graph import GraphIndex, paper_key


def evidence_paper_ids(keys, index: GraphIndex) -> set[str]:
    """Catalog ids of the papers behind evidence keys ('kg:<paper_id>' or '<file>#nnn')."""
    ids = set()
    for k in keys:
        if k.startswith("kg:"):
            doc = index.paper_by_id.get(k[3:])
            if doc:
                ids.add(paper_key(doc["source"]))
        else:
            ids.add(paper_key(k.split("#")[0]))
    return ids


class AnswerWriter:
    def __init__(self, llm, index: GraphIndex, *, harden: bool = True, output_guard: bool = True,
                 base_system: str = prompts.ANSWER_SYSTEM):
        self.llm = llm
        self.index = index
        self.harden = harden
        self.output_guard = output_guard
        self.canary = f"cnry-{uuid.uuid4().hex[:10]}"
        self.system = base_system + (prompts.SECURITY_RULES_ANSWER.format(canary=self.canary) if harden else "")
        self.known_ids = index.catalog_ids()

    def build_user_message(self, question: str, items: list[tuple[str, str]], history_text: str = "",
                           clarifications: list[str] | tuple = ()) -> str:
        if self.harden:
            nonce = guard.new_nonce()
            parts = [guard.spotlight("prior_conversation", history_text or "(none)", nonce, user_supplied=True)]
            if clarifications:
                parts.append("Clarifications:\n" + guard.escape_xml("\n".join(clarifications)))
            parts.append("Context (paper excerpts and knowledge-graph records):\n" + guard.spotlight_documents(items, nonce))
            parts.append(guard.spotlight("user_question", question, nonce, user_supplied=True))
        else:
            parts = [f"Prior conversation:\n{history_text}"] if history_text else []
            if clarifications:
                parts.append("Clarifications:\n" + "\n".join(clarifications))
            parts.append("Context (paper excerpts and knowledge-graph records):\n" +
                         ("\n\n---\n\n".join(v for _, v in items) or "(no evidence retrieved)"))
            parts.append(f"Question: {question}")
        return "\n\n".join(parts)

    def write(self, question: str, items: list[tuple[str, str]], history_text: str = "",
              clarifications: list[str] | tuple = ()) -> tuple[str, list[str]]:
        """Returns (answer, output-guard findings)."""
        from langchain_core.messages import HumanMessage, SystemMessage
        msg = self.build_user_message(question, items, history_text, clarifications)
        reply = self.llm.invoke([SystemMessage(content=self.system), HumanMessage(content=msg)],
                                config={"tags": ["role:answer"]}).content or ""
        return self.check(reply, {k for k, _ in items})

    def check(self, reply: str, evidence_keys) -> tuple[str, list[str]]:
        if not self.output_guard:
            return reply, []
        v = guard.inspect_output(reply, canary=self.canary, prompt_fingerprints=prompts.PROMPT_FINGERPRINTS,
                                 evidence_ids=evidence_paper_ids(evidence_keys, self.index), known_ids=self.known_ids)
        return v.text, v.findings
