import json
import math
import tempfile
import unittest
from pathlib import Path

from jev_laya_free.distilbert_model import (
    calibration_temperature,
    save_calibration,
)
from jev_laya_free.training.calibrate import (
    calibrate,
    fit_stability_bootstrap,
)
from jev_laya_free.training.cli import _rescale, _resolve_calibration
from jev_laya_free.training.metrics import fit_temperature


def _log_softmax(logits, temperature=1.0):
    values = [float(v) / temperature for v in logits]
    pivot = max(values)
    exps = [math.exp(v - pivot) for v in values]
    total = sum(exps)
    return [math.log(e / total) for e in exps]


class CalibrationPersistenceTests(unittest.TestCase):
    def _write_checkpoint(self, directory):
        config = {
            "format": "jev-laya-distilbert-v1",
            "model": "distilbert-base",
            "head_dim": 64,
            "max_options": 16,
        }
        config_path = Path(directory) / "jev_laya_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def test_missing_calibration_defaults_to_one(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_checkpoint(directory)
            self.assertEqual(calibration_temperature(directory), 1.0)

    def test_save_calibration_merges_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_checkpoint(directory)
            save_calibration(directory, {
                "temperature": 0.5,
                "nll_before": 1.2,
                "nll_after": 1.1,
                "split": ["a.jsonl", "b.jsonl"],
            })
            self.assertEqual(calibration_temperature(directory), 0.5)
            stored = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["calibration"]["temperature"], 0.5)
            self.assertEqual(stored["calibration"]["fitted_on"], "a.jsonl+b.jsonl")
            self.assertEqual(stored["model"], "distilbert-base")

    def test_save_calibration_rejects_non_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_checkpoint(directory)
            with self.assertRaises(ValueError):
                save_calibration(directory, {"temperature": 0.0})
            with self.assertRaises(ValueError):
                save_calibration(directory, {"temperature": float("nan")})


class FitStabilityTests(unittest.TestCase):
    def test_bootstrap_reports_distribution(self):
        log_probs = [_log_softmax([2.0, 0.0]) for _ in range(20)]
        labels = [0] * 20
        result = fit_stability_bootstrap(log_probs, labels, iterations=50, seed=3)
        self.assertEqual(result["iterations"], 50)
        self.assertGreaterEqual(result["max"], result["min"])
        self.assertGreaterEqual(result["p90"], result["p10"])
        self.assertGreaterEqual(result["std"], 0.0)

    def test_bootstrap_empty(self):
        self.assertEqual(fit_stability_bootstrap([], [], iterations=10)["iterations"], 0)


class CalibrateCommandTests(unittest.TestCase):
    def test_calibrate_end_to_end_on_held_out_split(self):
        from jev_laya_free.training.data import load_examples
        full = load_examples("data/public-test-a.jsonl")
        self.assertTrue(full)
        report = calibrate(
            checkpoint="checkpoints/local-distilbert",
            data=["data/public-test-a.jsonl"],
            device="cpu",
            batch_size=8,
            bootstrap_iterations=20,
            apply=False,
        )
        self.assertEqual(report["counts"]["examples"], len(full))
        self.assertGreater(report["counts"]["questions"], 0)
        self.assertGreater(report["calibration"]["temperature"], 0.0)
        self.assertEqual(report["applied"], False)
        self.assertEqual(report["deterministic_precedence"]["passed"],
                       report["deterministic_precedence"]["total"])


class RescaleFlagTests(unittest.TestCase):
    def test_rescale_identity_at_one(self):
        record = {"probabilities": [0.1, 0.9], "label_index": 1}
        out = _rescale([record], 1.0)[0]
        self.assertAlmostEqual(sum(out["probabilities"]), 1.0)
        self.assertEqual(out["predicted_index"], 1)
        for a, b in zip(record["probabilities"], out["probabilities"]):
            self.assertAlmostEqual(a, b, places=9)

    def test_rescale_cools_confident_predictions(self):
        record = {"probabilities": [0.98, 0.02], "label_index": 0}
        out = _rescale([record], 0.5)[0]
        # Lowering temperature sharpens: the confident class gets more mass.
        self.assertGreater(out["probabilities"][0], 0.98)
        self.assertAlmostEqual(sum(out["probabilities"]), 1.0)
        self.assertEqual(out["predicted_index"], 0)

    def test_rescale_heats_spreads_predictions(self):
        record = {"probabilities": [0.98, 0.02], "label_index": 0}
        out = _rescale([record], 2.0)[0]
        # Raising temperature flattens: the confident class loses mass.
        self.assertLess(out["probabilities"][0], 0.98)
        self.assertAlmostEqual(sum(out["probabilities"]), 1.0)

    def test_rescale_preserves_non_probability_fields(self):
        record = {"probabilities": [0.25, 0.75], "label_index": 1, "kind": "choice"}
        out = _rescale([record], 1.5)[0]
        self.assertEqual(out["kind"], "choice")
        self.assertEqual(out["label_index"], 1)

    def test_resolve_calibration_numeric_and_checkpoint(self):
        self.assertEqual(_resolve_calibration(None), 1.0)
        self.assertEqual(_resolve_calibration("1.25"), 1.25)
        with tempfile.TemporaryDirectory() as directory:
            save_calibration(directory, {"temperature": 0.75})
            self.assertEqual(_resolve_calibration(directory), 0.75)
            # A valid config with no calibration block falls back to 1.0.
            config_path = Path(directory) / "jev_laya_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            del config["calibration"]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(_resolve_calibration(directory), 1.0)


if __name__ == "__main__":
    unittest.main()
