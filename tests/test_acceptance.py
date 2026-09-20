import unittest
from jev_laya_free.training.acceptance import AcceptanceConfig, annotations, evaluate_acceptance
from jev_laya_free.training.data import normalize_row
from jev_laya_free.training.engine import flatten_examples
from jev_laya_free.training.metrics import evaluate_predictions


class AcceptanceTests(unittest.TestCase):
    def test_hermes_gold_survives_normalization(self):
        row = {'state': {'modality': 'code'}, 'questions': {'loop_state': {
            'type': 'choice', 'criteria': {'progress': 'ok', 'repeat_without_progress': 'repeat'}}},
            'gold': {'loop_state': 'repeat_without_progress'}}
        example = normalize_row(row, source='local', ordinal=0)
        flat = flatten_examples([example])[0]
        record = dict(flat, **flat['target'], kind='choice')
        report = evaluate_predictions([record])
        self.assertEqual(report['repeat_without_progress']['support'], 1)
        self.assertEqual(report['repeat_without_progress']['recall'], 1)
        raw = annotations(dict(row, predictions={'loop_state': 'progress'}, metadata={'repeat_without_progress': False}))
        self.assertTrue(raw['expected_repeat'])
        self.assertFalse(raw['predicted_repeat'])

    def test_unrelated_boolean_is_not_repeat(self):
        result = annotations({'question_name': 'needs_review', 'kind': 'noul', 'label_index': 1,
                              'probabilities': [0, 1], 'metadata': {'repeat_without_progress': True}})
        self.assertNotIn('expected_repeat', result)

    def fixtures(self):
        rows = []
        for modality, hands in [('code', ['inspect_code', 'run_test']),
                                ('pdf', ['extract_pdf_text', 'render_pdf_page']),
                                ('image', ['inspect_image', 'synthesize'])]:
            for index, hand in enumerate(hands):
                gold = {'next_hand': hand, 'loop_state': 'repeat_without_progress' if index else 'progress'}
                if modality == 'code': gold['high_risk_code'] = bool(index)
                rows.append({'modality': modality, 'gold': gold, 'predictions': dict(gold),
                             'kind': 'choice', 'probabilities': [1, 0], 'label_index': 0})
        return rows

    def test_separate_modality_gates_and_confusion(self):
        rows = self.fixtures()
        report = evaluate_acceptance(rows)
        self.assertTrue(report['passed'])
        self.assertFalse(report['authorizes_execution'])
        self.assertEqual(report['gates']['pdf_extract_vs_render']['metrics']['confusion_matrix']['render_pdf_page']['render_pdf_page'], 1)
        rows[3]['predictions']['next_hand'] = 'extract_pdf_text'
        report = evaluate_acceptance(rows)
        self.assertFalse(report['gates']['pdf_extract_vs_render']['passed'])
        self.assertFalse(report['gates']['next_hand_pdf']['passed'])
        self.assertTrue(report['gates']['next_hand_image']['passed'])

    def test_missing_evidence_does_not_pass(self):
        report = evaluate_acceptance([])
        self.assertFalse(report['passed'])
        self.assertTrue(report['gates']['deterministic_guard_precedence']['passed'])
        self.assertIsNone(report['gates']['latency_ms']['passed'])
        self.assertFalse(report['gates']['repeat_risk']['passed'])

    def test_false_positives_and_missing_predictions(self):
        rows = self.fixtures()
        rows[0]['predictions']['loop_state'] = 'repeat_without_progress'
        rows[0]['predictions']['high_risk_code'] = True
        rows[4]['predictions'].pop('next_hand')
        report = evaluate_acceptance(rows)
        for name in ['repeat_risk', 'high_risk_code', 'image_visual_inspection']:
            self.assertFalse(report['gates'][name]['passed'])

    def test_calibration_resources_and_validation(self):
        rows = self.fixtures()
        for row in rows:
            row.update(probabilities=[.6, .4], latency_ms=20, vram_mb=100)
        report = evaluate_acceptance(rows, AcceptanceConfig(max_latency_p95_ms=10, max_vram_mb=200))
        self.assertFalse(report['gates']['calibration']['passed'])
        self.assertFalse(report['gates']['latency_ms']['passed'])
        self.assertTrue(report['gates']['vram_mb']['passed'])
        with self.assertRaises(ValueError): AcceptanceConfig(repeat_recall=2)
        rows[0]['probabilities'] = [float('nan'), 1]
        with self.assertRaises(ValueError): evaluate_acceptance(rows)
