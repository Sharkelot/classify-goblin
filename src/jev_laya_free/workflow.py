"""Stateless deterministic Hermes/Qwen guard; model outputs never authorize work."""
import hashlib
from . import schema
from .client import Choice, Noul, ClientError

BOOLS = {'terminal', 'artifact_delta', 'board_delta', 'result_changed', 'same_action',
         'evidence_sufficient', 'source_grounded', 'test_needed'}
COUNTS = {'same_action_streak', 'same_tool_failures', 'context_compactions'}
TEXTS = {'status', 'modality', 'source_digest', 'location', 'extraction_quality',
         'artifact_digest', 'previous_artifact_digest', 'board_digest', 'previous_board_digest'}
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
    schema.require(state.get('modality', '') in ('', 'code', 'pdf', 'image'), 'unsupported modality')
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
    else:
        hand = None
    return {'hand': hand, 'executor': 'qwen' if hand else None}


def decide(state, client=None):
    gate = guard(state)
    output = {'gate': gate, 'route': route(state, gate),
              'advisory': {'authoritative': False, 'status': 'disabled', 'answers': {}}}
    if gate['decision'] != 'allow':
        output['advisory']['status'] = 'skipped_by_gate'
    elif client is not None:
        try:
            # Only routing metadata is sent. No source content or action/result text.
            summary = {k: state[k] for k in ('modality', 'extraction_quality', 'test_needed',
                                           'evidence_sufficient', 'source_grounded') if k in state}
            result = client.system_one(state=summary, questions={
                'next_hand': Choice(instructions='Suggest the next evidence step', options=[
                    'inspect_code', 'run_test', 'extract_pdf_text', 'render_pdf_page',
                    'inspect_image', 'synthesize', 'review', 'stop']),
                'needs_review': Noul(instructions='Should a human review the evidence?')})
            output['advisory'].update(status='ok', answers=result['answers'])
        except (ClientError, schema.ValidationError):
            output['advisory']['status'] = 'unavailable'
    return output
