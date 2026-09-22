"""CG-MM-14: deterministic multimodal workflow + advisory typed routing.

Regression tests for the deterministic/advisory boundary:

* modality validation from a single source of truth (implemented evidence
  paths only; unknown modalities fail closed);
* per-capability typed evidence questions with stable ordered labels and a
  full server round-trip that preserves label order;
* capability/task identity in Qwen prompts and workflow outputs;
* guard precedence (terminal/repeated/no-progress/failed-tool/compaction)
  and model non-override of route, guard, and profile assignment;
* fail-closed backend/assessor paths with no fallback decision;
* deterministic profile revalidation before any caller invokes Hermes Kanban.
"""
import json
import threading
import unittest
from datetime import datetime, timezone
from contextlib import contextmanager
from http.client import HTTPConnection

from classify_goblin import schema
from classify_goblin.client import ClientError
from classify_goblin.server import LocalServer
from classify_goblin.taxonomy import (
    EVIDENCE_HANDS, evidence_questions, profile_fit_question)
from classify_goblin.workflow import (
    decide, guard, validate_profile_selection)
from classify_goblin.profile_assessor import (
    RosterSnapshot, AssessmentRequest, assess)
from classify_goblin.multimodal.qwen_service import (
    OpenAICompatClient, QwenArtifactBackend, QwenCapabilityUnavailable,
    _system_prompt)
from classify_goblin.artifacts import ResolvedArtifact


@contextmanager
def running(backend=None):
    server = LocalServer(('127.0.0.1', 0), backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def post(server, payload, headers=None):
    conn = HTTPConnection('127.0.0.1', server.server_port, timeout=2)
    conn.request('POST', '/v1/systemone', payload,
                 headers or {'Content-Type': 'application/json'})
    reply = conn.getresponse()
    status, body = reply.status, reply.read()
    conn.close()
    return status, (json.loads(body) if body else {})


def _artifact(kind, data, mime):
    import hashlib
    digest = hashlib.sha256(data).hexdigest()
    return ResolvedArtifact('a1', kind, mime, digest, (), {}, data)


def _roster(names, valid=None, overrides=None):
    overrides = overrides or {}
    entries = []
    for name in names:
        entry = {
            'name': name,
            'on_disk': True,
            'valid_assignee': True,
            'description': 'test profile',
            'description_auto': False,
            'model_label': 'M1',
            'provider_label': 'p1',
            'context_length': 200000,
            'modalities': ['text', 'vision'],
            'skill_names': ['s'],
            'capabilities': ['code'],
            'workspace': None,
            'status': 'ok',
            'snapshot_time': '2026-09-21T00:00:00Z',
        }
        entry.update(overrides.get(name, {}))
        entries.append(entry)
    return RosterSnapshot.from_dict({
        'snapshot_time': '2026-09-21T00:00:00Z',
        'valid_assignees': valid if valid is not None else list(names),
        'entries': entries})


class AdversarialClient:
    """A model client whose answers may try to override deterministic policy."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = 0

    def system_one(self, **kwargs):
        self.calls += 1
        return {'answers': self.answers}


def _conforming_answers(questions, overrides=None):
    """Build a schema-conforming answer set for normalized questions."""
    out = {}
    for name, q in questions.items():
        kind = q['type']
        if kind == 'noul':
            out[name] = {'type': 'noul', 'noul': 0.5}
        elif kind == 'choice':
            keys = list(q['criteria'])
            p = {k: 1.0 / len(keys) for k in keys}
            out[name] = {'type': 'choice', 'confidence': 0.5,
                        'probabilities': p, 'choice': keys[0]}
        else:
            keys = [str(i) for i in range(len(q['criteria']))]
            p = {k: 1.0 / len(keys) for k in keys}
            out[name] = {'type': 'score', 'confidence': 0.5,
                        'probabilities': p,
                        'score': sum(int(k) * p[k] for k in keys),
                        'legend': {str(i): schema.text(v)
                                   for i, v in enumerate(q['criteria'])}}
    for name, answer in (overrides or {}).items():
        out[name] = answer
    return out


class ModalityValidationTests(unittest.TestCase):
    def test_single_source_of_truth_for_modalities(self):
        # The workflow accepts exactly the modalities with an implemented
        # evidence path (plus the empty/text path). Nothing else.
        self.assertEqual(tuple(sorted(EVIDENCE_HANDS)),
                         ('', 'audio', 'code', 'image', 'pdf', 'video'))
        for modality in ('', 'code', 'pdf', 'image', 'video', 'audio'):
            guard({'modality': modality})
        with self.assertRaises(schema.ValidationError):
            guard({'modality': 'hologram'})

    def test_unknown_modality_fails_closed_in_questions(self):
        with self.assertRaises(schema.ValidationError):
            evidence_questions('hologram')


class EvidenceQuestionTests(unittest.TestCase):
    EXPECTED = {
        '': ('synthesize', 'review'),
        'code': ('inspect_code', 'run_test', 'synthesize'),
        'pdf': ('extract_pdf_text', 'render_pdf_page', 'synthesize'),
        'image': ('inspect_image', 'synthesize'),
        'video': ('inspect_video', 'synthesize'),
        'audio': ('transcribe_audio', 'synthesize'),
    }

    def test_stable_ordered_labels_per_capability(self):
        for modality, labels in self.EXPECTED.items():
            questions = evidence_questions(modality)
            self.assertEqual(list(questions['evidence_next_hand']['criteria']),
                             list(labels))
            # Deterministic across calls: same labels, same order.
            self.assertEqual(evidence_questions(modality), questions)

    def test_no_undifferentiated_global_next_hand_in_evidence_questions(self):
        # The broken global 8-option next_hand must not leak into the
        # per-capability evidence surface.
        for modality in self.EXPECTED:
            labels = set(evidence_questions(modality)['evidence_next_hand']['criteria'])
            self.assertNotIn('stop', labels)
            self.assertNotIn('review', labels - {'synthesize', 'review'})

    def test_profile_fit_question_ordered_labels_with_abstention(self):
        question = profile_fit_question(('developer-goblin', 'video-goblin'))
        self.assertEqual(list(question['criteria']),
                         ['developer-goblin', 'video-goblin', 'abstain'])
        with self.assertRaises(schema.ValidationError):
            profile_fit_question(())
        with self.assertRaises(schema.ValidationError):
            profile_fit_question(('a', 'a'))

    def test_round_trip_preserves_label_order(self):
        # Full server round-trip: per-capability evidence questions + a
        # profile-fit question keep their label order byte-for-byte.
        from classify_goblin.backends import RulesBackend
        questions = evidence_questions('pdf')
        questions['profile_fit'] = profile_fit_question(
            ('developer-goblin', 'video-goblin'))
        body = schema.request({'model': 'local-default', 'state': {'modality': 'pdf'},
                              'questions': questions})
        with running(RulesBackend()) as (server, _):
            status, resp = post(server, json.dumps(body).encode())
        self.assertEqual(status, 200)
        self.assertEqual(list(resp['answers']['evidence_next_hand']['probabilities']),
                         ['extract_pdf_text', 'render_pdf_page', 'synthesize'])
        self.assertEqual(list(resp['answers']['profile_fit']['probabilities']),
                         ['developer-goblin', 'video-goblin', 'abstain'])


class GuardPrecedenceAndOverrideTests(unittest.TestCase):
    def test_guard_precedence_unchanged(self):
        for state, decision in [({'terminal': True, 'same_action_streak': 9}, 'terminal'),
                                ({'same_action_streak': 3}, 'stop'),
                                ({'same_action_streak': 2}, 'review'),
                                ({'context_compactions': 2}, 'stop'),
                                ({'same_tool_failures': 3}, 'stop'),
                                ({'same_action_streak': 9, 'artifact_delta': True}, 'allow')]:
            self.assertEqual(guard(state)['decision'], decision)

    def test_model_cannot_override_route(self):
        # The model answers a valid but conflicting evidence hand; the
        # deterministic route wins.
        questions = evidence_questions('code')
        normalized = schema.request({'model': 'workflow', 'state': {},
                                    'questions': questions})['questions']
        client = AdversarialClient(_conforming_answers(normalized, {
            'evidence_next_hand': {'type': 'choice', 'choice': 'synthesize',
                                   'confidence': 0.9,
                                   'probabilities': {'inspect_code': 0.0,
                                                    'run_test': 0.1,
                                                    'synthesize': 0.9}}}))
        out = decide({'modality': 'code'}, client)
        self.assertEqual(out['gate']['decision'], 'allow')
        self.assertEqual(out['route']['hand'], 'inspect_code')
        self.assertEqual(out['advisory']['status'], 'ok')

    def test_model_cannot_override_guard(self):
        client = AdversarialClient(_conforming_answers(
            schema.request({'model': 'workflow', 'state': {},
                           'questions': evidence_questions('code')})['questions']))
        out = decide({'same_action_streak': 3}, client)
        self.assertEqual(out['gate']['decision'], 'stop')
        self.assertEqual(out['route']['hand'], 'stop')
        self.assertIsNone(out['route']['executor'])
        self.assertEqual(client.calls, 0)  # gate skips the model entirely

    def test_label_order_mismatch_fails_closed(self):
        # The model answers with a label set that does not match the
        # per-capability question (an old undifferentiated global label):
        # the advisory fails closed; the route is untouched.
        client = AdversarialClient({'next_hand': {'type': 'choice', 'choice': 'stop',
                                                 'confidence': 0.9,
                                                 'probabilities': {'stop': 1.0}}})
        out = decide({'modality': 'code'}, client)
        self.assertEqual(out['advisory']['status'], 'unavailable')
        self.assertEqual(out['route']['hand'], 'inspect_code')

    def test_unavailable_client_fails_closed_with_no_fallback(self):
        class Offline:
            def system_one(self, **kwargs):
                raise ClientError('offline')
        out = decide({'modality': 'video'}, Offline())
        self.assertEqual(out['advisory']['status'], 'unavailable')
        self.assertEqual(out['route']['hand'], 'inspect_video')


class ProfileAdvisoryTests(unittest.TestCase):
    def test_safe_recommendation_selected_deterministically(self):
        # video-goblin is distinguishable (auto description -> lower confidence)
        # so the assessor can rank developer-goblin first and recommend.
        roster = _roster(['developer-goblin', 'video-goblin'],
                         overrides={'video-goblin': {'description_auto': True}})
        out = decide({'modality': 'code'}, None, assessor=assess, roster=roster)
        profile = out['advisory']['profile']
        self.assertIs(profile['authoritative'], False)
        self.assertEqual(profile['status'], 'ok')
        self.assertEqual(profile['verdict'], 'recommend')
        self.assertEqual(profile['candidates'], ['developer-goblin', 'video-goblin'])
        self.assertEqual(profile['selected'], 'developer-goblin')

    def test_model_profile_answer_cannot_authorize_assignment(self):
        # The model says video-goblin; the deterministic assessor selection
        # wins (or nothing is selected).
        roster = _roster(['developer-goblin', 'video-goblin'],
                         overrides={'video-goblin': {'description_auto': True}})
        questions = evidence_questions('code')
        questions['profile_fit'] = profile_fit_question(roster.valid_assignees)
        normalized = schema.request({'model': 'workflow', 'state': {},
                                    'questions': questions})['questions']
        client = AdversarialClient(_conforming_answers(normalized, {
            'profile_fit': {'type': 'choice', 'choice': 'video-goblin',
                           'confidence': 0.9,
                           'probabilities': {'developer-goblin': 0.0,
                                            'video-goblin': 0.8,
                                            'abstain': 0.2}}}))
        out = decide({'modality': 'code'}, client, assessor=assess, roster=roster)
        self.assertEqual(out['advisory']['profile']['selected'], 'developer-goblin')

    def test_ambiguous_candidates_review_selects_nothing(self):
        # Two identical candidates: top-two delta 0 < epsilon -> review.
        roster = _roster(['developer-goblin', 'video-goblin'])
        out = decide({'modality': 'code'}, None, assessor=assess, roster=roster)
        # Distinct names but identical sanitized fields => identical scores.
        self.assertEqual(out['advisory']['profile']['verdict'], 'review')
        self.assertIsNone(out['advisory']['profile']['selected'])

    def test_low_confidence_review_selects_nothing(self):
        # Direct assessor: penalties 0.25 (auto description) + 0.15 (status
        # unknown) + 0.15 (workspace unknown) + 0.10 (stale snapshot, via
        # explicit now > 600s after the roster snapshot) = confidence 0.35 < 0.40.
        roster = _roster(['weak-goblin'],
                         overrides={'weak-goblin': {'description_auto': True,
                                                   'status': 'unknown',
                                                   'context_length': 2000}})
        request = AssessmentRequest(task_kind='code', min_context=1000,
                                   workspace_repo='some-repo')
        # Roster snapshot_time is 2026-09-21T00:00:00Z; now 1200s later -> stale.
        now = datetime(2026, 9, 21, 0, 20, 0, tzinfo=timezone.utc)
        assessment = assess(request, roster, now=now)
        self.assertEqual(assessment['verdict'], 'review')
        self.assertEqual(assessment['reason'], 'low_confidence')
        self.assertIsNone(validate_profile_selection(assessment, roster))

    def test_no_candidate_abstains(self):
        roster = _roster(['gone-goblin'],
                         valid=[],
                         overrides={'gone-goblin': {'on_disk': False, 'valid_assignee': False}})
        out = decide({'modality': 'code'}, None, assessor=assess, roster=roster)
        self.assertEqual(out['advisory']['profile']['verdict'], 'abstain')
        self.assertIsNone(out['advisory']['profile']['selected'])

    def test_invalid_profile_never_selected(self):
        # A profile that is not a valid assignee is excluded by F2 and can
        # never be selected, even if it is the only entry.
        roster = _roster(['ghost-goblin'], valid=[])
        out = decide({'modality': 'code'}, None, assessor=assess, roster=roster)
        self.assertEqual(out['advisory']['profile']['verdict'], 'abstain')
        self.assertIsNone(out['advisory']['profile']['selected'])

    def test_assessor_unavailable_fails_closed(self):
        def broken(request, snapshot):
            raise ValueError('roster vanished')
        out = decide({'modality': 'code'}, None, assessor=broken,
                     roster=_roster(['developer-goblin']))
        profile = out['advisory']['profile']
        self.assertEqual(profile['status'], 'unavailable')
        self.assertIsNone(profile['selected'])
        # No fallback: the gate and route are still deterministic.
        self.assertEqual(out['gate']['decision'], 'allow')
        self.assertEqual(out['route']['hand'], 'inspect_code')

    def test_revalidation_gate_before_kanban(self):
        # Distinguishable candidates so the assessor can 'recommend'
        # (video-goblin's auto description lowers its confidence).
        roster = _roster(['developer-goblin', 'video-goblin'],
                         overrides={'video-goblin': {'description_auto': True}})
        request = AssessmentRequest(task_kind='code')
        assessment = assess(request, roster)
        self.assertEqual(assessment['verdict'], 'recommend')
        # Fresh identical snapshot: the deterministic revalidation passes.
        self.assertEqual(validate_profile_selection(assessment, roster),
                         'developer-goblin')
        # Stale snapshot (both profiles modified): no candidate revalidates,
        # so the deterministic gate fails closed with no selection.
        stale = _roster(['developer-goblin', 'video-goblin'],
                        overrides={'developer-goblin': {'description': 'changed after assessment'},
                                  'video-goblin': {'description': 'also changed'}})
        self.assertIsNone(validate_profile_selection(assessment, stale))
        # Missing profile in the fresh roster: the top candidate is absent
        # and the fallback candidate is no longer a valid assignee, so the
        # gate fails closed with no selection.
        missing = _roster(['video-goblin'], valid=[])
        self.assertIsNone(validate_profile_selection(assessment, missing))
        # Non-recommend verdict: fail closed.
        self.assertIsNone(validate_profile_selection(
            {'verdict': 'review', 'candidates': assessment['candidates']}, roster))
        self.assertIsNone(validate_profile_selection({}, roster))


class BackendFailClosedTests(unittest.TestCase):
    def _backend(self, transport):
        return QwenArtifactBackend(OpenAICompatClient(
            base_url='http://127.0.0.1:9999', transport=transport))

    def test_audio_fails_closed_before_transport(self):
        calls = []

        def transport(url, encoded, headers, timeout):
            calls.append(url)
            return b'{}'

        backend = self._backend(transport)
        artifact = _artifact('audio', b'RIFF....', 'audio/wav')
        with self.assertRaises(QwenCapabilityUnavailable):
            backend.predict_artifacts({'modality': 'audio'}, {}, [artifact])
        self.assertEqual(calls, [])  # no fake success, no HTTP call

    def test_unknown_modality_fails_closed(self):
        calls = []

        def transport(url, encoded, headers, timeout):
            calls.append(url)
            return b'{}'

        backend = self._backend(transport)
        artifact = _artifact('hologram', b'x', 'application/octet-stream')
        with self.assertRaises(QwenCapabilityUnavailable):
            backend.predict_artifacts({'modality': 'hologram'}, {}, [artifact])
        self.assertEqual(calls, [])

    def test_verified_video_proceeds_to_transport(self):
        calls = []

        def transport(url, encoded, headers, timeout):
            calls.append(url)
            inner = {
                'answers': {
                    'evidence_next_hand': {
                        'type': 'choice', 'choice': 'inspect_video',
                        'confidence': 1.0,
                        'probabilities': {'inspect_video': 1.0, 'synthesize': 0.0}},
                    'needs_review': {'type': 'noul', 'noul': 0.1},
                },
            }
            return json.dumps({
                'choices': [{'message': {'content': json.dumps(inner)}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
            }).encode()

        backend = self._backend(transport)
        artifact = _artifact('video', b'\x00\x00\x00\x14ftypmp42', 'video/mp4')
        raw = backend.predict_artifacts({'modality': 'video'},
                                       evidence_questions('video'), [artifact])
        self.assertEqual(calls, ['http://127.0.0.1:9999/v1/chat/completions'])
        self.assertEqual(raw['answers']['evidence_next_hand']['choice'],
                         'inspect_video')


class PromptIdentityTests(unittest.TestCase):
    def test_prompt_carries_capability_identity(self):
        for modality in ('code', 'pdf', 'image', 'video', 'audio'):
            prompt = _system_prompt({'modality': modality},
                                    evidence_questions(modality))
            self.assertIn(f'Capability under evaluation: {modality}.', prompt)

    def test_prompt_default_capability_is_text(self):
        prompt = _system_prompt('plain text state', evidence_questions(''))
        self.assertIn('Capability under evaluation: text.', prompt)

    def test_prompt_carries_task_identity_when_present(self):
        prompt = _system_prompt({'modality': 'code', 'task_identity': 'abc123'},
                                evidence_questions('code'))
        self.assertIn('Task identity: abc123', prompt)


class OutputIdentityTests(unittest.TestCase):
    def test_decide_output_carries_capability_and_task_identity(self):
        out = decide({'modality': 'video'})
        self.assertEqual(out['capability'], 'video')
        self.assertTrue(out['task_identity'])
        self.assertIs(out['advisory']['authoritative'], False)
        self.assertEqual(out['advisory']['status'], 'disabled')
        self.assertEqual(out['route'], {'hand': 'inspect_video', 'executor': 'qwen'})

    def test_text_state_has_no_hand_and_still_has_identity(self):
        out = decide({})
        self.assertEqual(out['capability'], 'text')
        self.assertTrue(out['task_identity'])
        self.assertIsNone(out['route']['hand'])

    def test_text_only_wire_compatibility_preserved(self):
        # A text-only /v1/systemone request with the new typed questions
        # still round-trips through the real server (no artifacts).
        from classify_goblin.backends import RulesBackend
        questions = evidence_questions('')
        body = schema.request({'model': 'local-default', 'state': 'hello',
                              'questions': questions})
        with running(RulesBackend()) as (server, _):
            status, resp = post(server, json.dumps(body).encode())
        self.assertEqual(status, 200)
        self.assertEqual(set(resp['answers']), set(questions))
        self.assertEqual(resp['answers']['evidence_next_hand']['type'], 'choice')


if __name__ == '__main__':
    unittest.main()
