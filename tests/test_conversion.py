import json
from pathlib import Path
import tempfile
import unittest

from jev_laya_free.training.conversion import build, eto, hazard, hermes, observable
from jev_laya_free.training.data import normalize_row, split_examples


class ConversionTests(unittest.TestCase):
    def test_gold_loop_overrides_stale_metadata(self):
        for label in ('progress', 'repeat_without_progress', 'blocked', 'terminal'):
            row = {'id': 'board:task:1', 'state': {}, 'questions': {},
                   'gold': {'loop_state': label}, 'repeat_without_progress': True}
            example = normalize_row(row, source='local', ordinal=0)
            self.assertEqual(example.metadata['repeat_without_progress'], label == 'repeat_without_progress')
            self.assertEqual(example.metadata['loop_state'], label)

    def test_single_episode_cannot_be_split(self):
        rows = [{'id': f'b:t:{i}', 'state': {}, 'gold': {'loop_state': 'blocked'}} for i in range(5)]
        examples = list(hermes(rows))
        train, validation = split_examples(examples)
        self.assertEqual(len({e.group for e in examples}), 1)
        self.assertTrue(not train or not validation)

    def test_hazard_causal_prefix_and_official_split(self):
        row = {'id': 'train_1', 'scaffold': 'tool', 'n_edits': 3,
               'resolved': True, 'edit_outcomes': [False, True, False]}
        examples = list(hazard([row], 'train'))
        self.assertEqual(examples[1].state['edit_outcomes'], [False])
        self.assertEqual(examples[1].targets['next_edit_outcome']['label_index'], 1)
        self.assertNotIn('resolved', json.dumps(examples[1].state))
        self.assertEqual(examples[1].metadata['official_split'], 'train')

    def test_eto_drops_thought_and_keeps_action(self):
        turns = [('human', 'prompt'), ('gpt', 'OK'), ('human', 'Observation: room'),
                 ('gpt', 'Thought: SECRET_RATIONALE\nAction: look'),
                 ('human', 'Observation: room'), ('gpt', 'Thought: more private\nAction: look')]
        examples = list(eto([{'id': 'x', 'conversations': [{'from': r, 'value': v} for r, v in turns]}], 'test'))
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[1].targets['repeated_action_observation']['label_index'], 1)
        self.assertNotIn('SECRET_RATIONALE', json.dumps([e.to_dict() for e in examples]))
        self.assertEqual(observable({'rationale': 'private', 'result': 'ok'}), {'result': 'ok'})

    def test_build_reproducible_and_license_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'train.json').write_text(json.dumps([{'id': 'train_1', 'scaffold': 'tool', 'edit_outcomes': [False]}]))
            config = root / 'config.json'
            config.write_text(json.dumps({'sources': [{'id': 'hazard', 'adapter': 'hazard', 'glob': '*.json', 'policy': 'public', 'license': 'CC-BY-4.0', 'license_file': 'README.md', 'license_marker': 'license: cc-by-4.0'}, {'id': 'absent', 'adapter': 'manifest-only', 'glob': 'missing/*'}]}))
            result = build(root, root/'missing', root/'out', config)
            self.assertEqual(result['sources'][0]['status'], 'excluded-license-unverified')
            self.assertEqual(result['sources'][1]['status'], 'unavailable')
            (root/'README.md').write_text('license: cc-by-4.0')
            first = build(root, root/'missing', root/'out', config)
            second = build(root, root/'missing', root/'out', config)
            self.assertEqual(first, second)
            self.assertEqual(first['splits']['train']['examples'], 1)


if __name__ == '__main__':
    unittest.main()
