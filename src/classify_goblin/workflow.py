"""Stateless deterministic Hermes/Qwen guard; model outputs never authorize work."""
import hashlib
from . import schema
from .client import Choice, Noul, ClientError
from .taxonomy import questions as catalog_questions, EVIDENCE_HANDS, evidence_questions

BOOLS = {'terminal', 'artifact_delta', 'board_delta', 'result_changed', 'same_action',
         'evidence_sufficient', 'source_grounded', 'test_needed'}
COUNTS = {'same_action_streak', 'same_tool_failures', 'context_compactions'}
TEXTS = {'status', 'modality', 'source_digest', 'location', 'extraction_quality',
         'artifact_digest', 'previous_artifact_digest', 'board_digest', 'previous_board_digest',
         'task_identity'}
VALUES = {'action', 'previous_action', 'result', 'previous_result'}


def fingerprint(value):
    schema.json_value(value)
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def guard(state):
    schema.require(isinstance(state, dict) and set(state) <= BOOLS | COUNTS | TEXTS | VALUES, 'invalid workflow state fields')
    schema.require(len(schema.dumps(state).encode()) <= 32768, 'workflow state exceeds 32 KiB')
    for key, value in state.items():
        if key in BOOLS:
            schema.require(type(value) is bool, 'workflow flags must be booleans')
        elif key in COUNTS:
            schema.require(type(value) is int and 0 <= value <= 1000000, 'invalid workflow counter')
        elif key in TEXTS:
            schema.require(isinstance(value, str) and len(value) <= 512, 'invalid workflow text')
    schema.require(state.get('modality', '') in EVIDENCE_HANDS, 'unsupported modality')
    progress = any(state.get(k, False) for k in ('artifact_delta', 'board_delta', 'result_changed'))
    for key in ('artifact_digest', 'board_digest', 'result'):
        previous = 'previous_' + key
        if key in state and previous in state:
            progress |= fingerprint(state[key]) != fingerprint(state[previous])
    repeated = state.get('same_action_streak', 0)
    if state.get('same_action') or ('action' in state and 'previous_action' in state and fingerprint(state['action']) == fingerprint(state['previous_action'])):
        repeated = max(2, repeated)
    if state.get('terminal') or state.get('status', '').lower() in ('completed', 'done', 'blocked', 'review'):
        decision = 'terminal'
    elif not progress and (repeated >= 3 or state.get('context_compactions', 0) >= 2 or state.get('same_tool_failures', 0) >= 3):
        decision = 'stop'
    elif not progress and repeated >= 2:
        decision = 'review'
    else:
        decision = 'allow'
    return {'decision': decision, 'progress': progress,
            'fingerprints': {k: fingerprint(state.get(k)) for k in ('action', 'result')},
            'same_action_streak': repeated}


def route(state, gate):
    decision = gate['decision']
    if decision != 'allow':
        return {'hand': 'stop' if decision == 'terminal' else decision, 'executor': None}
    if all(state.get(k) for k in ('evidence_sufficient', 'source_grounded', 'source_digest', 'location')):
        hand = 'synthesize'
    elif state.get('modality') == 'code':
        hand = 'run_test' if state.get('test_needed') else 'inspect_code'
    elif state.get('modality') == 'pdf':
        hand = 'render_pdf_page' if state.get('extraction_quality') in ('partial', 'empty', 'scanned') else 'extract_pdf_text'
    elif state.get('modality') == 'image':
        hand = 'inspect_image'
    elif state.get('modality') == 'video':
        hand = 'inspect_video'
    elif state.get('modality') == 'audio':
        hand = 'transcribe_audio'
    else:
        hand = None
    return {'hand': hand, 'executor': 'qwen' if hand else None}


def decide(state, client=None, assessor=None, roster=None):
    gate = guard(state)
    capability = state.get('modality', '') or 'text'
    task_identity = fingerprint({'state': state})
    output = {'gate': gate, 'route': route(state, gate),
              'capability': capability, 'task_identity': task_identity,
              'advisory': {'authoritative': False, 'status': 'disabled', 'answers': {}}}
    if gate['decision'] != 'allow':
        output['advisory']['status'] = 'skipped_by_gate'
    elif client is not None:
        try:
            # Only routing metadata is sent. No source content or action/result text.
            summary = {k: state[k] for k in ('modality', 'extraction_quality', 'test_needed',
                                           'evidence_sufficient', 'source_grounded') if k in state}
            if state.get('task_identity'):
                summary['task_identity'] = state['task_identity']
            # Per-capability typed questions when a capability is present; the
            # undifferentiated global catalog is the text-only fallback.
            questions = evidence_questions(state.get('modality', '')) if state.get('modality') else catalog_questions()
            if roster is not None:
                from .taxonomy import profile_fit_question
                questions['profile_fit'] = profile_fit_question(roster.valid_assignees)
            result = client.system_one(state=summary, questions=questions)
            # Fail closed on a label-order mismatch: the model's answer set
            # must name exactly the questions asked, with conforming labels.
            schema.answers(result['answers'], questions)
            output['advisory'].update(status='ok', answers=result['answers'])
        except (ClientError, schema.ValidationError):
            output['advisory']['status'] = 'unavailable'
    # Deterministic, advisory-only profile fit. The model's profile answer
    # never authorizes an assignment; the assessor + revalidation own it.
    profile = {'authoritative': False, 'status': 'disabled', 'verdict': None,
               'candidates': [], 'selected': None}
    if assessor is not None:
        if gate['decision'] != 'allow':
            profile['status'] = 'skipped_by_gate'
        else:
            try:
                assessment = assessor(_request_from_state(state), roster)
                profile.update(status='ok', verdict=assessment['verdict'],
                              candidates=[c['profile'] for c in assessment['candidates']],
                              selected=validate_profile_selection(assessment, roster))
            except Exception:
                profile.update(status='unavailable', verdict=None, candidates=[], selected=None)
    output['advisory']['profile'] = profile
    return output


def _request_from_state(state):
    from .profile_assessor.schema import AssessmentRequest
    return AssessmentRequest(task_kind=state.get('modality', '') or 'text')


def validate_profile_selection(assessment, fresh_snapshot):
    """Deterministic revalidation gate before any caller invokes Kanban.

    Only a ``recommend`` verdict is eligible. The selected profile is the
    first ranked candidate that revalidates to ``ok`` against the fresh
    roster snapshot. Any mismatch (stale, missing, invalid, or a
    non-recommend verdict) fails closed with ``None`` — no fallback.
    """
    from .profile_assessor.revalidation import revalidate
    if not isinstance(assessment, dict) or assessment.get('verdict') != 'recommend':
        return None
    for candidate in assessment.get('candidates', []):
        name = candidate['profile']
        expected = candidate.get('entry_digest')
        if expected is None:
            continue
        if revalidate(name, fresh_snapshot, expected).status == 'ok':
            return name
    return None
