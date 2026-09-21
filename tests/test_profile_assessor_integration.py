"""Integration tests for the profile assessor CLI (JEV-MM-11).

Covers the pure profile-dir parser on a realistic profile layout, the
end-to-end CLI (roster + request JSON -> JSON result), CLI error paths
(exit 2 on missing/invalid inputs), and digest consistency between the
module API and the CLI output.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jev_laya_free.profile_assessor import (
    RosterSnapshot,
    assess,
    parse_profile_dir,
)

ROOT = Path(__file__).parent.parent
FIX = Path(__file__).parent / 'fixtures' / 'profile_assessor'
CLI = ROOT / 'src' / 'jev_laya_free' / 'profile_assessor' / 'cli.py'


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True, text=True, timeout=60)


def _load_roster(name):
    with open(FIX / name, encoding='utf-8') as fh:
        return RosterSnapshot.from_dict(json.load(fh))


class PureParserIntegrationTests(unittest.TestCase):
    def _make_profile(self, root, name, **overrides):
        d = root / name
        d.mkdir()
        desc = overrides.get('description', 'integration profile')
        (d / 'profile.yaml').write_text(
            f'description: {desc}\n'
            'description_auto: false\n', encoding='utf-8')
        (d / 'config.yaml').write_text(
            'model:\n'
            '  default: TestModel-1\n'
            '  provider: test-provider\n'
            '  supports_vision: true\n'
            '  context_length: 123456\n', encoding='utf-8')
        skills = overrides.get('skills', [])
        skills_dir = d / 'skills'
        skills_dir.mkdir()
        for s in skills:
            (skills_dir / s).mkdir()
        # Files the parser must NEVER read (contain sentinel secrets).
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

    def test_parser_on_realistic_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = self._make_profile(root, 'test-profile',
                                   skills=['alpha-skill', 'beta-skill'])
            entry = parse_profile_dir(d)
            self.assertEqual(entry.name, 'test-profile')
            self.assertEqual(entry.description, 'integration profile')
            self.assertEqual(entry.model_label, 'TestModel-1')
            self.assertEqual(entry.provider_label, 'test-provider')
            self.assertEqual(entry.context_length, 123456)
            self.assertEqual(entry.modalities, ('text', 'vision'))
            self.assertEqual(entry.skill_names, ('alpha-skill', 'beta-skill'))
            # No sentinel secret may appear anywhere in the entry.
            blob = json.dumps(entry.to_dict())
            for sentinel in ('SENTINEL_ENV', 'SENTINEL_AUTH', 'SENTINEL_MEM',
                            'SENTINEL_SESSION'):
                self.assertNotIn(sentinel, blob)

    def test_parser_missing_config_is_unavailable_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'bare-profile'
            d.mkdir()
            (d / 'profile.yaml').write_text('description: bare\n',
                                           encoding='utf-8')
            entry = parse_profile_dir(d)
            self.assertEqual(entry.status, 'unavailable')
            self.assertIn('config', entry.status_reason)


class CliEndToEndTests(unittest.TestCase):
    def test_cli_recommend(self):
        proc = run_cli('--roster', str(FIX / 'roster_2026-09-21.json'),
                       '--request', str(FIX / 'req_code.json'))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result['verdict'], 'recommend')
        self.assertEqual(result['candidates'][0]['profile'], 'developer-goblin')
        self.assertTrue(result['advisory_only'])

    def test_cli_abstain(self):
        proc = run_cli('--roster', str(FIX / 'roster_2026-09-21.json'),
                       '--request', str(FIX / 'req_no_match.json'))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result['verdict'], 'abstain')
        self.assertEqual(result['reason'], 'no_eligible_candidate')

    def test_cli_review_high_risk(self):
        proc = run_cli('--roster', str(FIX / 'roster_2026-09-21.json'),
                       '--request', str(FIX / 'req_high_risk_code.json'))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result['verdict'], 'review')
        self.assertEqual(result['reason'], 'high_risk_task')

    def test_cli_missing_roster_exit_2(self):
        proc = run_cli('--roster', str(FIX / 'nope.json'),
                       '--request', str(FIX / 'req_code.json'))
        self.assertEqual(proc.returncode, 2)

    def test_cli_invalid_json_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'bad.json'
            bad.write_text('{not json', encoding='utf-8')
            proc = run_cli('--roster', str(bad),
                           '--request', str(FIX / 'req_code.json'))
            self.assertEqual(proc.returncode, 2)

    def test_cli_invalid_request_schema_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'badreq.json'
            bad.write_text(json.dumps({'task_kind': 42}), encoding='utf-8')
            proc = run_cli('--roster', str(FIX / 'roster_2026-09-21.json'),
                           '--request', str(bad))
            self.assertEqual(proc.returncode, 2)

    def test_cli_digest_matches_module(self):
        snap = _load_roster('roster_2026-09-21.json')
        with open(FIX / 'req_code.json', encoding='utf-8') as fh:
            from jev_laya_free.profile_assessor.schema import AssessmentRequest
            req = AssessmentRequest.from_dict(json.load(fh))
        module_result = assess(req, snap)
        proc = run_cli('--roster', str(FIX / 'roster_2026-09-21.json'),
                       '--request', str(FIX / 'req_code.json'))
        cli_result = json.loads(proc.stdout)
        self.assertEqual(module_result['roster_digest'],
                         cli_result['roster_digest'])
        self.assertEqual(module_result['request_digest'],
                         cli_result['request_digest'])

    def test_cli_no_bytes_in_output(self):
        proc = run_cli('--roster', str(FIX / 'roster_2026-09-21.json'),
                       '--request', str(FIX / 'req_code.json'))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # JSON round-trips cleanly; the text is pure str (no bytes).
        result = json.loads(proc.stdout)
        self.assertIsInstance(proc.stdout, str)
        self.assertNotIn('\x00', proc.stdout)


class CliProfilesRootTests(unittest.TestCase):
    def test_cli_parses_profile_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, skills in (('alpha-goblin', ['s1']),
                                ('beta-goblin', [])):
                d = root / name
                d.mkdir()
                (d / 'profile.yaml').write_text(
                    f'description: {name} desc\n', encoding='utf-8')
                (d / 'config.yaml').write_text(
                    'model:\n'
                    '  default: M-1\n'
                    '  provider: p-1\n'
                    '  supports_vision: false\n'
                    '  context_length: 5000\n', encoding='utf-8')
                skills_dir = d / 'skills'
                skills_dir.mkdir()
                for s in skills:
                    (skills_dir / s).mkdir()
                (d / '.env').write_text('SENTINEL_ENV=x\n', encoding='utf-8')
            proc = run_cli('--profiles-root', str(root))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(proc.stdout)
            self.assertIn('entries', result)
            names = {e['name'] for e in result['entries']}
            self.assertEqual(names, {'alpha-goblin', 'beta-goblin'})
            alpha = next(e for e in result['entries']
                        if e['name'] == 'alpha-goblin')
            self.assertEqual(alpha['skill_names'], ['s1'])
            self.assertEqual(alpha['modalities'], ['text'])
            self.assertNotIn('SENTINEL_ENV', json.dumps(result))


if __name__ == '__main__':
    unittest.main()
