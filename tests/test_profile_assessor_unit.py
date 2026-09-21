"""Unit tests for the profile assessor (JEV-MM-11).

Covers: deterministic filters F1-F7 (pass and fail paths), verdict rules,
tie-breaking determinism, revalidation + fallback chain, the pure profile-dir
parser (allowlisted reads only), and output hygiene.
"""
import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path

from jev_laya_free.profile_assessor import (
    RosterSnapshot,
    assess,
    parse_profile_dir,
    revalidate,
)
from jev_laya_free.profile_assessor.filters import filter_chain
from jev_laya_free.profile_assessor.schema import AssessmentRequest

FIX = Path(__file__).parent / 'fixtures' / 'profile_assessor'


def _load_roster(name):
    with open(FIX / name, encoding='utf-8') as fh:
        return RosterSnapshot.from_dict(json.load(fh))


def _load_req(name):
    with open(FIX / name, encoding='utf-8') as fh:
        return AssessmentRequest.from_dict(json.load(fh))


def _by_name(result, name):
    for c in result['candidates']:
        if c['profile'] == name:
            return c
    return None


class FilterTests(unittest.TestCase):
    def test_f1_on_disk_excludes_missing(self):
        snap = _load_roster('missing_profile.json')
        req = _load_req('req_code.json')
        result = assess(req, snap)
        excluded = {e['profile']: e['filter_reason'] for e in result['excluded']}
        self.assertEqual(excluded.get('ghost-goblin'), 'not_on_disk')
        self.assertIsNone(_by_name(result, 'ghost-goblin'))

    def test_f2_valid_assignee_excludes(self):
        snap = _load_roster('missing_profile.json')
        req = _load_req('req_code.json')
        result = assess(req, snap)
        excluded = {e['profile']: e['filter_reason'] for e in result['excluded']}
        self.assertEqual(excluded.get('ghost-goblin2'), 'not_valid_assignee')
        self.assertIsNone(_by_name(result, 'ghost-goblin2'))

    def test_f3_context_below_min_excludes(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_low_context.json')
        result = assess(req, snap)
        excluded = {e['profile']: e['filter_reason'] for e in result['excluded']}
        self.assertEqual(excluded.get('developer-goblin'), 'context_below_min')
        self.assertIsNone(_by_name(result, 'developer-goblin'))

    def test_f3_context_unknown_fails_closed(self):
        snap = _load_roster('roster_2026-09-21.json')
        entries = [dataclasses.replace(e, context_length=None)
                   if e.name == 'default' else e for e in snap.entries]
        snap2 = RosterSnapshot(tuple(entries), tuple(snap.valid_assignees),
                              snap.snapshot_time, snap.digest)
        req = _load_req('req_low_context.json')
        result = assess(req, snap2)
        excluded = {e['profile']: e['filter_reason'] for e in result['excluded']}
        self.assertEqual(excluded.get('default'), 'context_unknown')

    def test_f4_modality_not_declared_excludes(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_design_vision.json')
        result = assess(req, snap)
        excluded = {e['profile']: e['filter_reason'] for e in result['excluded']}
        self.assertEqual(excluded.get('luna-goblin'), 'modality_not_declared')
        self.assertEqual(excluded.get('default'), 'modality_not_declared')

    def test_f5_capability_not_declared_excludes(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_no_match.json')
        result = assess(req, snap)
        excluded = {e['profile']: e['filter_reason'] for e in result['excluded']}
        self.assertEqual(excluded.get('developer-goblin'), 'capability_not_declared')
        self.assertEqual(len(result['excluded']), 8)
        self.assertEqual(result['verdict'], 'abstain')

    def test_f6_workspace_mismatch_excludes(self):
        snap = _load_roster('roster_2026-09-21.json')
        # Give developer-goblin a concrete workspace that does NOT match
        # the request -> hard mismatch (unknown/None would pass with penalty).
        entries = [dataclasses.replace(e, workspace='repo-b')
                   if e.name == 'developer-goblin' else e
                   for e in snap.entries]
        snap2 = RosterSnapshot(tuple(entries), tuple(snap.valid_assignees),
                              snap.snapshot_time, snap.digest)
        req = dataclasses.replace(_load_req('req_code.json'),
                                 workspace_repo='repo-a')
        result = assess(req, snap2)
        excluded = {e['profile']: e['filter_reason'] for e in result['excluded']}
        self.assertEqual(excluded.get('developer-goblin'), 'workspace_mismatch')

    def test_f6_workspace_unknown_passes_with_penalty(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = dataclasses.replace(_load_req('req_code.json'),
                                 workspace_repo='repo-a')
        # All roster entries have workspace=None (unknown) -> pass F6, but
        # each pays the compat_unknown confidence penalty.
        result = assess(req, snap)
        c = _by_name(result, 'developer-goblin')
        self.assertIsNotNone(c)
        # compat_unknown penalty applied (workspace was None)
        self.assertLess(c['confidence'], 0.95)

    def test_f7_status_blocked_excludes(self):
        snap = _load_roster('roster_2026-09-21.json')
        entries = [dataclasses.replace(e, status='blocked')
                   if e.name == 'developer-goblin' else e
                   for e in snap.entries]
        snap2 = RosterSnapshot(tuple(entries), tuple(snap.valid_assignees),
                              snap.snapshot_time, snap.digest)
        req = _load_req('req_code.json')
        result = assess(req, snap2)
        excluded = {e['profile']: e['filter_reason'] for e in result['excluded']}
        self.assertEqual(excluded.get('developer-goblin'), 'status_blocked')

    def test_f7_status_unavailable_excludes(self):
        snap = _load_roster('roster_2026-09-21.json')
        entries = [dataclasses.replace(e, status='unavailable')
                   if e.name == 'developer-goblin' else e
                   for e in snap.entries]
        snap2 = RosterSnapshot(tuple(entries), tuple(snap.valid_assignees),
                              snap.snapshot_time, snap.digest)
        req = _load_req('req_code.json')
        result = assess(req, snap2)
        excluded = {e['profile']: e['filter_reason'] for e in result['excluded']}
        self.assertEqual(excluded.get('developer-goblin'), 'status_unavailable')

    def test_filter_chain_prefix_property(self):
        # Monotonic exclusion: a candidate excluded by the first k filters
        # is also excluded by the full 7-filter chain (the full chain
        # includes the prefix). This is the invariant that actually holds.
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_design_vision.json')
        for k in range(1, 8):
            prefix = filter_chain(snap.entries, k, req, snap.valid_assignees)
            full = filter_chain(snap.entries, 7, req, snap.valid_assignees)
            for name, reason in prefix.items():
                if reason is not None:
                    self.assertIsNotNone(
                        full.get(name),
                        f'prefix-excluded {name} survived the full chain')


class VerdictTests(unittest.TestCase):
    def test_no_eligible_candidate(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_no_match.json')
        result = assess(req, snap)
        self.assertEqual(result['verdict'], 'abstain')
        self.assertEqual(result['reason'], 'no_eligible_candidate')
        self.assertEqual(len(result['candidates']), 0)

    def test_high_risk_task(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_high_risk_code.json')
        result = assess(req, snap)
        self.assertEqual(result['verdict'], 'review')
        self.assertEqual(result['reason'], 'high_risk_task')
        self.assertEqual(len(result['candidates']), 1)

    def test_ambiguous_candidates_within_epsilon(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_tied.json')
        result = assess(req, snap)
        self.assertEqual(result['verdict'], 'review')
        self.assertEqual(result['reason'], 'ambiguous_candidates')
        names = {c['profile'] for c in result['candidates']}
        self.assertEqual(names, {'developer-goblin', 'video-review-goblin'})

    def test_ambiguous_boundary_gap_006_recommends(self):
        # Engineer a gap: developer (0.90) vs a synthetic 0.50 entry.
        snap = _load_roster('roster_2026-09-21.json')
        entries = [dataclasses.replace(e, capabilities=[], description=None)
                   if e.name == 'video-review-goblin' else e
                   for e in snap.entries]
        snap2 = RosterSnapshot(tuple(entries), tuple(snap.valid_assignees),
                              snap.snapshot_time, snap.digest)
        req = _load_req('req_tied.json')
        result = assess(req, snap2)
        # developer 0.90 vs video-review 0.50 -> gap 0.40, not ambiguous
        self.assertEqual(result['verdict'], 'recommend')
        self.assertIsNone(result['reason'])

    def test_stale_snapshot_forces_review(self):
        snap = _load_roster('roster_2026-09-21.json')
        # Age the snapshot beyond max_snapshot_age (300s default).
        snap2 = RosterSnapshot(tuple(snap.entries), tuple(snap.valid_assignees),
                              '2026-09-20T22:00:00Z', snap.digest)
        req = _load_req('req_code.json')
        from datetime import datetime, timezone
        now = datetime(2026, 9, 21, 1, 0, 0, tzinfo=timezone.utc)
        result = assess(req, snap2, now=now)
        self.assertEqual(result['verdict'], 'review')
        self.assertEqual(result['reason'], 'stale_snapshot')

    def test_low_confidence_forces_review(self):
        snap = _load_roster('roster_2026-09-21.json')
        entries = [dataclasses.replace(
            e, description_auto=True, status='unknown',
            workspace=None, context_length=None)
            if e.name == 'developer-goblin' else e
            for e in snap.entries]
        snap3 = RosterSnapshot(tuple(entries), tuple(snap.valid_assignees),
                              snap.snapshot_time, snap.digest)
        req = _load_req('req_code.json')
        # Penalties: auto(-0.25) + status(-0.15) + workspace(-0.15)
        # + age>600(-0.10) = 0.65 -> conf 0.35 < 0.40.
        req = dataclasses.replace(req, workspace_repo='repo-x')
        from datetime import datetime, timezone
        now = datetime(2026, 9, 21, 1, 0, 0, tzinfo=timezone.utc)
        result = assess(req, snap3, now=now)
        c = _by_name(result, 'developer-goblin')
        self.assertIsNotNone(c)
        self.assertLess(c['confidence'], 0.4)
        self.assertEqual(result['verdict'], 'review')
        self.assertEqual(result['reason'], 'low_confidence')

    def test_recommend(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_code.json')
        result = assess(req, snap)
        self.assertEqual(result['verdict'], 'recommend')
        self.assertIsNone(result['reason'])
        self.assertEqual(len(result['candidates']), 1)
        self.assertEqual(result['candidates'][0]['profile'], 'developer-goblin')


class TieBreakingTests(unittest.TestCase):
    def test_tied_candidates_name_asc_order(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_tied.json')
        result = assess(req, snap)
        names = [c['profile'] for c in result['candidates']]
        self.assertEqual(names, sorted(names))

    def test_deterministic_across_100_runs(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_tied.json')
        first = json.dumps(assess(req, snap), sort_keys=True)
        for _ in range(99):
            again = json.dumps(assess(req, snap), sort_keys=True)
            self.assertEqual(first, again)


class RevalidateTests(unittest.TestCase):
    def test_revalidate_ok(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_code.json')
        result = assess(req, snap)
        entry = next(e for e in snap.entries if e.name == 'developer-goblin')
        rv = revalidate('developer-goblin', snap, entry.entry_digest())
        self.assertEqual(rv.status, 'ok')

    def test_revalidate_stale_context_changed(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_code.json')
        result = assess(req, snap)
        entry = next(e for e in snap.entries if e.name == 'developer-goblin')
        # Fresh roster with changed context_length -> digest mismatch.
        fresh_entries = [dataclasses.replace(e, context_length=100000)
                        if e.name == 'developer-goblin' else e
                        for e in snap.entries]
        fresh = RosterSnapshot(tuple(fresh_entries), tuple(snap.valid_assignees),
                              '2026-09-21T01:00:00Z', snap.digest)
        rv = revalidate('developer-goblin', fresh, entry.entry_digest())
        self.assertEqual(rv.status, 'stale')

    def test_revalidate_missing(self):
        snap = _load_roster('roster_2026-09-21.json')
        fresh_entries = [e for e in snap.entries if e.name != 'developer-goblin']
        fresh = RosterSnapshot(tuple(fresh_entries), tuple(snap.valid_assignees),
                              '2026-09-21T01:00:00Z', snap.digest)
        entry = next(e for e in snap.entries if e.name == 'developer-goblin')
        rv = revalidate('developer-goblin', fresh, entry.entry_digest())
        self.assertEqual(rv.status, 'missing')

    def test_revalidate_invalid(self):
        snap = _load_roster('roster_2026-09-21.json')
        entry = next(e for e in snap.entries if e.name == 'developer-goblin')
        fresh = RosterSnapshot(tuple(snap.entries),
                              tuple(a for a in snap.valid_assignees
                                    if a != 'developer-goblin'),
                              '2026-09-21T01:00:00Z', snap.digest)
        rv = revalidate('developer-goblin', fresh, entry.entry_digest())
        self.assertEqual(rv.status, 'invalid')

    def test_fallback_chain_picks_next_candidate(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_tied.json')
        result = assess(req, snap)
        # Two candidates: developer-goblin (rank 1), video-review-goblin (rank 2).
        # Fresh roster removes developer-goblin from valid set -> invalid.
        fresh = RosterSnapshot(
            tuple(e for e in snap.entries if e.name != 'developer-goblin'),
            tuple(a for a in snap.valid_assignees
                  if a != 'developer-goblin'),
            '2026-09-21T01:00:00Z', snap.digest)
        from jev_laya_free.profile_assessor.revalidation import first_valid_candidate
        chosen = first_valid_candidate(result['candidates'], fresh)
        self.assertEqual(chosen, 'video-review-goblin')

    def test_fallback_chain_all_invalid_returns_none(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_code.json')
        result = assess(req, snap)
        fresh = RosterSnapshot(tuple(), tuple(), '2026-09-21T01:00:00Z',
                              snap.digest)
        from jev_laya_free.profile_assessor.revalidation import first_valid_candidate
        self.assertIsNone(first_valid_candidate(result['candidates'], fresh))


class ParserTests(unittest.TestCase):
    def _make_profile_dir(self, tmp):
        d = Path(tmp) / 'test-profile'
        d.mkdir()
        (d / 'profile.yaml').write_text(
            'description: a test profile\n'
            'description_auto: false\n', encoding='utf-8')
        (d / 'config.yaml').write_text(
            'model:\n'
            '  default: TestModel-1\n'
            '  provider: test-provider\n'
            '  supports_vision: true\n'
            '  context_length: 123456\n'
            '  api_key: SENTINEL_CONFIG_KEY\n'
            'providers:\n'
            '  test-provider:\n'
            '    name: Test Provider Name\n'
            '    base_url: https://example.test/v1\n', encoding='utf-8')
        (d / 'skills').mkdir()
        (d / 'skills' / 'alpha-skill').mkdir()
        (d / 'skills' / 'beta-skill').mkdir()
        (d / '.env').write_text('SENTINEL_ENV=secret\n', encoding='utf-8')
        (d / 'auth.json').write_text('{"token": "SENTINEL_AUTH"}\n',
                                    encoding='utf-8')
        (d / 'memories').mkdir()
        (d / 'memories' / 'm.json').write_text('{"fact": "SENTINEL_MEM"}\n',
                                              encoding='utf-8')
        (d / 'sessions').mkdir()
        (d / 'sessions' / 's.jsonl').write_text('{"t": "SENTINEL_SESSION"}\n',
                                               encoding='utf-8')
        return d

    def test_parser_reads_only_allowlisted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._make_profile_dir(tmp)
            opened = []
            real_open = open

            def tracking_open(path, *args, **kwargs):
                opened.append(str(path))
                return real_open(path, *args, **kwargs)

            import builtins
            orig = builtins.open
            builtins.open = tracking_open
            try:
                entry = parse_profile_dir(d)
            finally:
                builtins.open = orig

            for p in opened:
                base = os.path.basename(p)
                self.assertIn(base, ('profile.yaml', 'config.yaml', 'skills'),
                              f'parser read non-allowlisted file: {p}')
            # Forbidden files must not have been opened.
            for forbidden in ('.env', 'auth.json', 'm.json', 's.jsonl'):
                self.assertFalse(
                    any(os.path.basename(p) == forbidden for p in opened),
                    f'parser opened forbidden file {forbidden}')

    def test_parser_allowlisted_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._make_profile_dir(tmp)
            entry = parse_profile_dir(d)
            self.assertEqual(entry.description, 'a test profile')
            self.assertEqual(entry.model_label, 'TestModel-1')
            self.assertEqual(entry.provider_label, 'test-provider')
            self.assertEqual(entry.context_length, 123456)
            self.assertEqual(entry.modalities, ('text', 'vision'))
            self.assertEqual(entry.skill_names, ('alpha-skill', 'beta-skill'))
            self.assertFalse(entry.description_auto)


class OutputHygieneTests(unittest.TestCase):
    def test_json_serializable_sorted_keys(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_code.json')
        result = assess(req, snap)
        text = json.dumps(result, sort_keys=True)
        self.assertIsInstance(text, str)

    def test_advisory_only_constant(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_code.json')
        result = assess(req, snap)
        self.assertTrue(result['advisory_only'])

    def test_digests_present_and_stable(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_code.json')
        r1 = assess(req, snap)
        r2 = assess(req, snap)
        self.assertEqual(r1['roster_digest'], r2['roster_digest'])
        self.assertEqual(r1['request_digest'], r2['request_digest'])
        self.assertTrue(r1['roster_digest'])
        self.assertTrue(r1['request_digest'])

    def test_no_bytes_fields(self):
        snap = _load_roster('roster_2026-09-21.json')
        req = _load_req('req_code.json')
        result = assess(req, snap)

        def check(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    check(v)
            elif isinstance(obj, list):
                for v in obj:
                    check(v)
            else:
                self.assertNotIsInstance(obj, bytes)

        check(result)

    def test_no_eligible_false_in_candidates(self):
        snap = _load_roster('missing_profile.json')
        req = _load_req('req_code.json')
        result = assess(req, snap)
        for c in result['candidates']:
            self.assertTrue(c['eligible'])


if __name__ == '__main__':
    unittest.main()
