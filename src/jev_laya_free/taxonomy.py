"""Dependency-free catalog of the typed advisory questions used by Hermes."""
from .client import Choice, Noul, Score

CAPABILITIES = ("model_routing", "confidence_action", "tool_screening", "progress", "completion", "skill_selection", "compaction", "citation", "rag_filter", "semantic_find", "composite", "intent_routing")
CATALOG = {name: {"state": {}, "policy": "advisory only; deterministic code owns thresholds and fail-open behavior"} for name in CAPABILITIES}

def questions():
    return {
        "model_routing": Choice(instructions="Which execution tier fits this turn?", options=["deterministic-code", "cheap-LLM", "frontier-LLM", "human"]),
        "confidence_action": Score(instructions="How confident is the proposed answer?", criteria=["low", "medium", "high"]),
        "tool_screening": Score(instructions="How harmful is this prompt, reply, or tool call?", criteria=["safe", "suspicious", "harmful"]),
        "progress": Choice(instructions="What is the trajectory status?", options=["continue", "warn", "replan", "halt"]),
        "completion": Noul(instructions="Should completion be held for review?"),
        "skill_selection": Score(instructions="How well does the best catalog skill match?", criteria=["weak", "possible", "strong"]),
        "compaction": Score(instructions="How useful is retaining this context item?", criteria=["drop", "keep"]),
        "citation": Noul(instructions="Does the cited context support the claim?"),
        "rag_filter": Choice(instructions="How should this passage be treated?", options=["keep", "flag", "drop"]),
        "semantic_find": Noul(instructions="Does a matching answer exist among candidates?"),
        "composite": Score(instructions="What is the composite quality/risk score?", criteria=["low", "medium", "high"]),
        "intent_routing": Choice(instructions="Which intent handler should receive this request?", options=["logic", "specialist-LLM", "human"]),
        "next_hand": Choice(instructions="Suggest the next evidence step", options=["inspect_code", "run_test", "extract_pdf_text", "render_pdf_page", "inspect_image", "synthesize", "review", "stop"]),
        "needs_review": Noul(instructions="Should a human review the evidence?"),
    }