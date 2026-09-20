import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_laya_free import ValidationError
from jev_laya_free.backends import DistilBertBackend
from jev_laya_free.training.data import (
    DecisionExample,
    make_synthetic_cases,
    normalize_row,
    split_examples,
    write_jsonl,
    load_examples,
)
from jev_laya_free.training.engine import flatten_examples
from jev_laya_free.training.metrics import (
    evaluate_predictions,
    evaluate_workflow_precedence,
    fit_temperature,
)


class DataTests(unittest.TestCase):
    def test_synthetic_fixtures_are_deterministic_and_grouped(self):
        first = [case.to_dict() for case in make_synthetic_cases(16)]
        second = [case.to_dict() for case in make_synthetic_cases(16)]
        self.assertEqual(first, second)
        train, validation = split_examples(make_synthetic_cases(32))
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertTrue({case.group for case in train}.isdisjoint({case.group for case in validation}))

    def test_local_row_redacts_obvious_credentials(self):
        row = {
            "task_id": "t-1",
            "state": {"summary": "Bearer super-secret-token", "api_key": "do-not-save"},
            "questions": {"ready": {"type": "noul", "instructions": "ready?"}},
            "gold": {"ready": True},
        }
        example = normalize_row(row, source="local", ordinal=0)
        encoded = json.dumps(example.to_dict())
        self.assertNotIn("super-secret-token", encoded)
        self.assertNotIn("do-not-save", encoded)
        self.assertEqual(example.targets["ready"]["label_index"], 1)

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            self.assertEqual(write_jsonl(path, make_synthetic_cases(3)), 3)
            loaded = load_examples(path)
            self.assertEqual(len(loaded), 3)
            self.assertIsInstance(loaded[0], DecisionExample)


class MetricsTests(unittest.TestCase):
    def test_metrics_and_temperature(self):
        records = [
            {"kind": "choice", "probabilities": [0.9, 0.1], "label_index": 0},
            {"kind": "score", "probabilities": [0.1, 0.8, 0.1], "label_index": 1},
            {"kind": "noul", "probabilities": [0.2, 0.8], "label_index": 1,
             "expected_repeat": True, "predicted_repeat": True},
        ]
        report = evaluate_predictions(records)
        self.assertEqual(report["overall"]["count"], 3)
        self.assertEqual(report["overall"]["accuracy"], 1.0)
        self.assertEqual(report["repeat_without_progress"]["recall"], 1.0)
        calibration = fit_temperature([[2.0, 0.0], [0.0, 2.0]], [0, 1])
        self.assertLessEqual(calibration["nll_after"], calibration["nll_before"])

    def test_deterministic_precedence_is_complete(self):
        report = evaluate_workflow_precedence()
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["passed"], report["total"])


class OptionalBackendTests(unittest.TestCase):
    def test_distilbert_requires_explicit_local_checkpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError):
                DistilBertBackend()

    def test_optional_typed_loss_runs_when_torch_is_present(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional dependency is not installed")
        from jev_laya_free.training.engine import typed_loss
        examples = flatten_examples(make_synthetic_cases(1))[:2]
        logits = torch.zeros((len(examples), 10), requires_grad=True)
        loss, components = typed_loss(logits, examples)
        self.assertTrue(float(loss.detach()) > 0)
        self.assertIn("brier", components)
        loss.backward()


if __name__ == "__main__":
    unittest.main()
