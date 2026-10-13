"""Deterministic routing between the standard RAG path and the agent.

No LLM decides the route: a model call to choose between models would cost a call on every
question to save one on some. The rules below send a question to the agent only when it
needs something the 1-call path cannot do — resolve a vague reference against the catalog,
gather evidence from several papers, combine authorship with content, or follow up on an
earlier turn. Everything else takes the standard path, which escalates to the agent (seeded
with the evidence it already retrieved) when its answer says the evidence was insufficient.

Evidence for the split (Checkpoint 5.1, 70 questions): the 1-call hybrid pipeline passed
60/70; its failures were authorship aggregation (now answered by the KG tier) and a few
retrieval misses — exactly the cases the escalation rule catches.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .knowledge_graph import GraphIndex

VAGUE_REFERENCE = re.compile(
    r"\b(that|this|those|these|the same|the other)\s+(\w+[\s-]+){0,2}?(paper|papers|study|studies|gan|method|methods|"
    r"approach|approaches|trick|one|ones|model|models|work|technique|framework|network|surrogate|test|case)\b", re.I)
MULTI_PAPER = re.compile(
    r"\b(which|what|list|all|both|each|across|other|several|multiple|different|compare|between)\s+(\w+[\s-]+){0,3}?"
    r"(papers|(?<!case )studies|documents|works|articles)\b|\bacross the (corpus|collection|papers)\b|"
    r"\b(in|of) (the )?corpus\b.*\b(all|every|which)\b", re.I)
CROSS_COMPARE = re.compile(r"\b(compare|comparison|contrast|versus|vs\.?|similarit\w+|differences?)\b", re.I)
_PAPER_REF = re.compile(r"\b(paper|study|report)\b", re.I)

# The answer prompt tells the model to "say so explicitly" when the excerpts are not enough;
# these are the ways it says so. The evidence must be the subject ("the excerpts do not
# mention…"), so a factual negative ("the dataset does not include submarines") does not match.
_EVIDENCE = r"(excerpts?|context|evidence|documents?|sources?|retrieved (text|material|chunks?)|provided (text|material|information))"
INSUFFICIENT = re.compile(
    rf"{_EVIDENCE}\s+(provided\s+|retrieved\s+|available\s+|given\s+|here\s+)?(do(es)?\s+not|don['’]t|doesn['’]t)\s+"
    r"(\w+\s+)?(contain|include|provide|mention|specify|state|say|describe|give|report|cover|discuss|indicate|list)\b|"
    r"not enough (information|detail|evidence)|insufficient (information|evidence|detail)|"
    r"\bno (information|details?|mention) (about|on|of|regarding)\b|"
    # not "confirm/verify": the security rules make the model say it "cannot verify" pasted claims,
    # which is a correct answer, not missing evidence (audit: it caused a needless escalation)
    r"\b(i|we) (can ?not|can['’]t|am unable to|was unable to|could not|couldn['’]t)\s+(find|determine|answer|identify)|"
    rf"not (specified|mentioned|stated|provided|reported|described|included|given|available) in the (provided |retrieved |available )?{_EVIDENCE}|"
    rf"{_EVIDENCE} (provided )?(is|are) (insufficient|silent)", re.I)


@dataclass
class RouteDecision:
    route: str                                # "standard" | "agent"
    reasons: list[str] = field(default_factory=list)


def decide(question: str, index: GraphIndex, has_history: bool = False, named_authors: int = 0) -> RouteDecision:
    """`named_authors` = known authors the question names (resolved by the KG answerer,
    including partial/misspelled names)."""
    reasons = []
    if VAGUE_REFERENCE.search(question):
        reasons.append("vague_reference")
    if MULTI_PAPER.search(question):
        reasons.append("multi_paper")
    referenced = {d["source"] for k, d in index.paper_by_key.items()
                  if re.search(rf"(?<![\w-]){re.escape(k)}(?![\w-])", question.lower())}
    if len(referenced) >= 2:
        reasons.append("names_multiple_papers")
    if len(referenced) >= 1 and CROSS_COMPARE.search(question) and len(_PAPER_REF.findall(question)) >= 2:
        reasons.append("cross_paper_comparison")
    if named_authors or index.authors_mentioned(question):
        reasons.append("author_plus_content")   # the KG tier declined it, so it asks more than a list
    if has_history and re.search(r"\b(it|its|they|them|their|he|she|his|her|that|those|the same|above|previous)\b",
                                 question, re.I):
        reasons.append("follow_up_on_history")
    return RouteDecision("agent" if reasons else "standard", reasons or ["single_hop"])


def needs_escalation(answer: str) -> bool:
    """True when the standard answer reports that the retrieved evidence was not enough."""
    return bool(INSUFFICIENT.search(answer or ""))
