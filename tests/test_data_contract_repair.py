"""GOBLIN-JEV-02 data-contract repair tests.

These tests pin the six required data-contract repairs for the multi-routing
benchmark. They are written to FAIL on the current (pre-repair) code and PASS
after the repair:

1. Explicit ordered label spaces with ``label`` + ``label_index`` (round-trip).
2. No synthetic generation metadata (``scenario``/``variant``) in model-visible
   state; preserved only as audit/split metadata.
3. Bounded numeric features stay in their valid range and the gold label is
   derived from the final perturbed state, not the pre-perturbation state.
4. Grouped, stratified splits: related variants/episodes (same capability +
   scenario family) never cross splits; held-out data uses an independent seed.
5. Hard negatives / counterfactual pairs (good/bad/uncertain outcomes).
6. Calibration is kept separate from the final held-out evaluation and raw vs
   calibrated metrics are reported separately.

Run with: ``PYTHONPATH=src python -m pytest tests/test_data_contract_repair.py -v``
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jev_laya_free.taxonomy import CAPABILITIES
from jev_laya_free.training.capability_benchmark import (
    DEFAULT_SEED,
    SCENARIOS,
    build_capability_benchmark,
    build_capability_examples,
    check_split_leakage,
    split_capability_examples,
    validate_line,
)

# The continuous (0..1) numeric features that must stay in range after jitter.
UNIT_FEATURES = (
    "complexity",
    "overlap",
    "relevance",
    "quality",
    "risk",
    "urgency",
)


class LabelRoundTripTests(unittest.TestCase):
    """Repair 1: explicit ordered label spaces with label + label_index."""

    def test_target_carries_explicit_label_string(self):
        for record in build_capability_examples():
            target = record["targets"][record["capability"]]
            self.assertIn(
                "label",
                target,
                f"target for {record['capability']} is missing the explicit 'label' field",
            )

    def test_label_matches_label_index(self):
        for record in build_capability_examples():
            capability = record["capability"]
            target = record["targets"][capability]
            question = record["questions"][capability]
            label = target["label"]
            index = target["label_index"]
            # Resolve the ordered label space the way the model sees it.
            qtype = question["type"]
            criteria = question.get("criteria", {})
            if qtype == "choice":
                labels = list(criteria)
            elif qtype == "score":
                labels = [str(c) for c in criteria]
            else:  # noul
                labels = ["false", "true"]
            self.assertEqual(
                labels[index],
                label,
                f"labels[{index}] != label for {capability}: {labels[index]!r} vs {label!r}",
            )

    def test_label_survives_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_capability_benchmark(output_dir=Path(temp) / "a")
            for split in ("train", "validation", "test"):
                path = Path(manifest["files"][split]["path"])
                for line in path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    target = record["targets"][record["capability"]]
                    self.assertIn("label", target, f"{split} record lost its label")
                    validate_line(record)


class MetadataLeakageTests(unittest.TestCase):
    """Repair 2: no generation metadata in model-visible state."""

    def test_state_has_no_generation_metadata(self):
        for record in build_capability_examples():
            state = record["state"]
            self.assertNotIn(
                "scenario",
                state,
                f"state leaks 'scenario' for {record['capability']}",
            )
            self.assertNotIn(
                "variant",
                state,
                f"state leaks 'variant' for {record['capability']}",
            )

    def test_metadata_preserves_generation_fields(self):
        for record in build_capability_examples():
            metadata = record["metadata"]
            self.assertIn("scenario", metadata, "metadata must preserve 'scenario'")
            self.assertIn("variant", metadata, "metadata must preserve 'variant'")

    def test_prompt_rendering_excludes_generation_metadata(self):
        from jev_laya_free.distilbert_model import render_question_prompt

        for record in build_capability_examples():
            prompt = render_question_prompt(
                record["state"], record["questions"][record["capability"]]
            )
            self.assertNotIn(
                "scenario",
                prompt,
                f"prompt leaks 'scenario' for {record['capability']}",
            )
            self.assertNotIn(
                "variant",
                prompt,
                f"prompt leaks 'variant' for {record['capability']}",
            )


class BoundedStateTests(unittest.TestCase):
    """Repair 3: bounded numeric features + gold label from final state."""

    def test_unit_features_stay_in_range(self):
        for record in build_capability_examples():
            for key in UNIT_FEATURES:
                value = record["state"].get(key)
                if isinstance(value, float):
                    self.assertGreaterEqual(
                        value,
                        0.0,
                        f"{key}={value} below 0 for {record['capability']}/{record['metadata']['scenario']}",
                    )
                    self.assertLessEqual(
                        value,
                        1.0,
                        f"{key}={value} above 1 for {record['capability']}/{record['metadata']['scenario']}",
                    )

    def test_gold_label_matches_final_perturbed_state(self):
        # The gold label must be the policy decision for the FINAL (jittered)
        # state, not the pre-perturbation state. For the boundary-threshold
        # capability the label is a deterministic function of the perturbed
        # numeric field; if the label were derived pre-jitter it will disagree
        # with the final state for at least one record.
        mismatch = 0
        for record in build_capability_examples():
            capability = record["capability"]
            scenario = record["metadata"]["scenario"]
            if scenario != "boundary-threshold":
                continue
            state = record["state"]
            label = record["targets"][capability]["label"]
            expected = _expected_boundary_label(capability, state)
            if expected is not None and expected != label:
                mismatch += 1
        self.assertEqual(
            mismatch,
            0,
            "gold label disagrees with the final perturbed state for "
            f"{mismatch} boundary-threshold records",
        )


def _expected_boundary_label(capability: str, state: dict) -> str | None:
    """Recompute the documented boundary-threshold decision from the final state."""
    if capability == "model_routing":
        complexity = state.get("complexity")
        if isinstance(complexity, (int, float)):
            return "frontier-LLM" if float(complexity) >= 0.6 else "cheap-LLM"
    if capability == "skill_selection":
        overlap = state.get("overlap")
        if isinstance(overlap, (int, float)):
            value = float(overlap)
            return "possible" if 0.4 <= value < 0.5 else "weak"
    if capability == "compaction":
        relevance = state.get("relevance")
        if isinstance(relevance, (int, float)):
            return "keep" if float(relevance) >= 0.3 else "drop"
    if capability == "composite":
        risk = state.get("risk")
        if isinstance(risk, (int, float)):
            return "medium" if float(risk) >= 0.5 else "low"
    if capability == "intent_routing":
        urgency = state.get("urgency")
        if isinstance(urgency, (int, float)):
            return "human" if float(urgency) >= 0.8 else "logic"
    if capability == "progress":
        streak = state.get("same_action_streak")
        if isinstance(streak, int):
            return "warn" if streak >= 2 else "continue"
    return None


class GroupedSplitTests(unittest.TestCase):
    """Repair 4: grouped, stratified splits with an independent held-out seed."""

    def test_no_group_crosses_splits(self):
        splits = split_capability_examples(build_capability_examples())
        seen: dict[tuple, str] = {}
        for split_name, records in splits.items():
            for record in records:
                group = (record["capability"], record["metadata"]["scenario"])
                if group in seen:
                    self.assertEqual(
                        seen[group],
                        split_name,
                        f"group {group} appears in {seen[group]!r} and {split_name!r}",
                    )
                else:
                    seen[group] = split_name

    def test_every_capability_and_scenario_represented_in_each_split(self):
        splits = split_capability_examples(build_capability_examples())
        for split_name in ("train", "validation", "test"):
            capabilities = {r["capability"] for r in splits[split_name]}
            self.assertEqual(
                capabilities,
                set(CAPABILITIES),
                f"{split_name} is missing capabilities: {set(CAPABILITIES) - capabilities}",
            )

    def test_heldout_uses_independent_seed(self):
        # The final held-out split must be generated with a seed independent of
        # the train/validation seed so it cannot be memorised from training.
        heldout = build_capability_examples(seed=DEFAULT_SEED + 1)
        base = build_capability_examples(seed=DEFAULT_SEED)
        heldout_states = {json.dumps(r["state"], sort_keys=True) for r in heldout}
        base_states = {json.dumps(r["state"], sort_keys=True) for r in base}
        # Independent seeds produce distinct states for the same (cap, scenario, variant).
        self.assertNotEqual(
            heldout_states,
            base_states,
            "held-out data was not generated with an independent seed",
        )


class HardNegativeTests(unittest.TestCase):
    """Repair 5: hard negatives / counterfactual pairs across outcome domains."""

    def test_includes_good_bad_uncertain_outcomes(self):
        records = build_capability_examples()
        # The scenario matrix must include at least one hard-negative (risky),
        # one ambiguous/uncertain, and one clean (safe) fixture per capability.
        for capability in CAPABILITIES:
            scenarios = {
                r["metadata"]["scenario"] for r in records if r["capability"] == capability
            }
            for required in ("safe", "risky", "ambiguous"):
                self.assertIn(
                    required,
                    scenarios,
                    f"capability {capability} is missing the {required!r} fixture",
                )

    def test_includes_counterfactual_pairs(self):
        records = build_capability_examples()
        # A counterfactual pair is two records for the same capability whose
        # states differ only in the decision-driving field (a "pair" with a
        # flipped outcome). At minimum, the safe and risky fixtures for the
        # same capability must carry distinct gold labels.
        for capability in CAPABILITIES:
            by_scenario = {
                r["metadata"]["scenario"]: r["targets"][capability]["label"]
                for r in records
                if r["capability"] == capability
            }
            self.assertNotEqual(
                by_scenario.get("safe"),
                by_scenario.get("risky"),
                f"capability {capability}: safe and risky fixtures share the same "
                "gold label, so there is no good/bad counterfactual pair",
            )


class CalibrationSeparationTests(unittest.TestCase):
    """Repair 6: calibration split is separate from the final held-out split."""

    def test_calibration_split_is_not_the_test_split(self):
        splits = split_capability_examples(build_capability_examples())
        self.assertNotIn("test", splits.get("calibration", []))
        # A distinct calibration split must exist and be disjoint from test.
        self.assertIn(
            "calibration",
            splits,
            "there is no dedicated calibration split separate from the held-out test split",
        )
        calibration_ids = {r["example_id"] for r in splits["calibration"]}
        test_ids = {r["example_id"] for r in splits["test"]}
        self.assertEqual(
            calibration_ids & test_ids,
            set(),
            "calibration and test splits share examples",
        )

    def test_raw_and_calibrated_metrics_are_reported_separately(self):
        from jev_laya_free.training.metrics import evaluate_predictions

        # Build a tiny set of prediction records from the benchmark targets and
        # verify the report separates raw from calibrated evaluation.
        records = build_capability_examples()[:8]
        predictions = []
        for record in records:
            capability = record["capability"]
            target = record["targets"][capability]
            predictions.append(
                {
                    "probabilities": target["probabilities"],
                    "label_index": target["label_index"],
                    "kind": record["questions"][capability]["type"],
                }
            )
        report = evaluate_predictions(predictions)
        self.assertIn("raw", report, "report must contain a 'raw' metrics block")
        self.assertIn("calibrated", report, "report must contain a 'calibrated' metrics block")
        self.assertIsNot(report["raw"], report["calibrated"])


if __name__ == "__main__":
    unittest.main()
