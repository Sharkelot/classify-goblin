"""Dependency-free catalog of the typed advisory questions used by Hermes."""
from .client import Choice, Noul, Score

CAPABILITIES = ("model_routing", "confidence_action", "tool_screening", "progress", "completion", "skill_selection", "compaction", "citation", "rag_filter", "semantic_find", "composite", "intent_routing")
CATALOG = {name: {"state": {}, "policy": "advisory only; deterministic code owns thresholds and fail-open behavior"} for name in CAPABILITIES}

# JEV-MM-14: single source of truth for modality validation. The workflow
# guard accepts exactly these modalities (the empty/text path plus every
# capability with an implemented evidence path). Unknown modalities fail
# closed. This map is the authoritative definition; the guard and the
# per-capability evidence questions are both derived from it.
EVIDENCE_HANDS = {
    '': ('synthesize', 'review'),
    'code': ('inspect_code', 'run_test', 'synthesize'),
    'pdf': ('extract_pdf_text', 'render_pdf_page', 'synthesize'),
    'image': ('inspect_image', 'synthesize'),
    'video': ('inspect_video', 'synthesize'),
    'audio': ('transcribe_audio', 'synthesize'),
}


def evidence_questions(modality):
    """Per-capability typed evidence questions with stable ordered labels.

    Returns a dict with an ``evidence_next_hand`` choice (the capability's
    ordered evidence-step labels) and a ``needs_review`` noul. The labels
    are stable and ordered; the broken undifferentiated global 8-option
    ``next_hand`` never appears here. Unknown modalities fail closed.
    """
    from . import schema
    schema.require(modality in EVIDENCE_HANDS, 'unsupported modality')
    return {
        'evidence_next_hand': {'type': 'choice',
                               'instructions': 'Suggest the next evidence step',
                               'criteria': list(EVIDENCE_HANDS[modality])},
        'needs_review': {'type': 'noul',
                         'instructions': 'Should a human review the evidence?'},
    }


def profile_fit_question(ordered_assignees):
    """A profile-fit/delegation recommendation question with ordered labels.

    The options are the roster's ordered valid assignees followed by
    ``abstain``. This is advisory only: it suggests a profile but never
    authorizes an assignment (the deterministic assessor + revalidation
    own that). Empty or duplicate rosters fail closed.
    """
    from . import schema
    schema.require(isinstance(ordered_assignees, (list, tuple)), 'assignees must be a sequence')
    schema.require(len(ordered_assignees) >= 1, 'at least one assignee required')
    schema.require(len(set(ordered_assignees)) == len(ordered_assignees), 'duplicate assignee')
    return {'type': 'choice',
            'instructions': 'Which profile best fits this task?',
            'criteria': [*ordered_assignees, 'abstain']}


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