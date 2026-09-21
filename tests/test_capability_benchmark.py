import json
import tempfile
import unittest
from pathlib import Path

from jev_laya_free.taxonomy import CAPABILITIES
from jev_laya_free.training.capability_benchmark import (
    FIXTURE_FULL,
    FIXTURE_SMOKE,
    FULL_VARIANTS_PER_SCENARIO,
    MIN_HELDOUT_PER_CAPABILITY,
    SCHEMA_VERSION,
    VARIANTS_PER_SCENARIO,
    build_capability_benchmark,
    build_capability_examples,
    check_split_leakage,
    coverage_report,
    fixture_class,
    heldout_per_capability,
    split_capability_examples,
    validate_line,
)


class CapabilityBenchmarkTests(unittest.TestCase):
    def test_example_count_is_families_times_scenarios_times_variants(self):
        for variants in (VARIANTS_PER_SCENARIO, FULL_VARIANTS_PER_SCENARIO):
            records = build_capability_examples(variants=variants)
            # 12 capability families x 7 scenarios x variants
            self.assertEqual(len(records), len(CAPABILITIES) * 7 * variants)

    def test_every_record_validates_against_schema(self):
        for record in build_capability_examples():
            validate_line(record)

    def test_split_assignment_is_deterministic_and_reconciles(self):
        records = build_capability_examples()
        splits = split_capability_examples(records)
        total = sum(len(v) for v in splits.values())
        self.assertEqual(total, len(records))
        self.assertEqual(set(splits), {"train", "validation", "test", "calibration"})
        # Re-running produces identical split sizes.
        again = split_capability_examples(build_capability_examples())
        self.assertEqual({k: len(v) for k, v in splits.items()},
                         {k: len(v) for k, v in again.items()})

    def test_no_split_leakage(self):
        splits = split_capability_examples(build_capability_examples())
        # Should not raise.
        check_split_leakage(splits)

    def test_states_are_globally_distinct(self):
        # Distinctness must hold even at the full fixture size.
        for variants in (VARIANTS_PER_SCENARIO, FULL_VARIANTS_PER_SCENARIO):
            records = build_capability_examples(variants=variants)
            seen = set()
            for record in records:
                key = json.dumps(record["state"], sort_keys=True)
                self.assertNotIn(key, seen, f"duplicate state: {key}")
                seen.add(key)

    def test_count_fields_stay_integers(self):
        # Jitter must not convert integer counts into floats.
        for record in build_capability_examples():
            for key, value in record["state"].items():
                if key in {"unverified_claims", "same_action_streak", "evidence_count",
                          "contradicting_facts", "variant"}:
                    self.assertIsInstance(value, int, f"{key} became {value!r}")

    def test_coverage_reports_all_families_with_support(self):
        splits = split_capability_examples(build_capability_examples())
        report = coverage_report(splits)
        self.assertEqual(set(report), set(CAPABILITIES))
        all_splits = ("train", "validation", "test", "calibration")
        for capability, entry in report.items():
            for split in all_splits:
                self.assertGreaterEqual(entry[split]["support"], 0)
            # 35 examples per capability across all splits (7 scenarios x 5 variants).
            total = sum(entry[s]["support"] for s in all_splits)
            self.assertEqual(total, 7 * VARIANTS_PER_SCENARIO)

    def test_fixture_class_smoke(self):
        self.assertEqual(fixture_class(VARIANTS_PER_SCENARIO), FIXTURE_SMOKE)
        self.assertLess(heldout_per_capability(VARIANTS_PER_SCENARIO), MIN_HELDOUT_PER_CAPABILITY)

    def test_fixture_class_full(self):
        self.assertEqual(fixture_class(FULL_VARIANTS_PER_SCENARIO), FIXTURE_FULL)
        self.assertGreaterEqual(heldout_per_capability(FULL_VARIANTS_PER_SCENARIO),
                               MIN_HELDOUT_PER_CAPABILITY)

    def test_full_fixture_has_enough_heldout_per_capability(self):
        # The quality gate: the full fixture must give >= 100 held-out examples
        # per capability for trustworthy accuracy/calibration claims.
        self.assertGreaterEqual(heldout_per_capability(FULL_VARIANTS_PER_SCENARIO), 100)

    def test_build_writes_deterministic_files_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            first = build_capability_benchmark(output_dir=Path(temp) / "a")
            second = build_capability_benchmark(output_dir=Path(temp) / "b")
            # Deterministic: identical bytes per split file.
            for split in ("train", "validation", "test"):
                a_bytes = Path(first["files"][split]["path"]).read_bytes()
                b_bytes = Path(second["files"][split]["path"]).read_bytes()
                self.assertEqual(a_bytes, b_bytes)
            # Manifest records the expected total and schema version.
            self.assertEqual(first["total_examples"], len(CAPABILITIES) * 7 * VARIANTS_PER_SCENARIO)
            self.assertEqual(first["schema_version"], SCHEMA_VERSION)
            # Manifest records the fixture class distinction.
            self.assertEqual(first["fixture_class"], FIXTURE_SMOKE)
            self.assertEqual(first["heldout_per_capability"], heldout_per_capability(VARIANTS_PER_SCENARIO))

    def test_build_full_fixture_records_full_class(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_capability_benchmark(
                output_dir=Path(temp) / "a", variants=FULL_VARIANTS_PER_SCENARIO)
            self.assertEqual(manifest["fixture_class"], FIXTURE_FULL)
            self.assertGreaterEqual(manifest["heldout_per_capability"], MIN_HELDOUT_PER_CAPABILITY)
            self.assertEqual(manifest["total_examples"],
                             len(CAPABILITIES) * 7 * FULL_VARIANTS_PER_SCENARIO)

    def test_build_is_reproducible_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as temp:
            build_capability_benchmark(output_dir=Path(temp) / "a")
            build_capability_benchmark(output_dir=Path(temp) / "b")
            a = (Path(temp) / "a" / "capability-benchmark-train.jsonl").read_bytes()
            b = (Path(temp) / "b" / "capability-benchmark-train.jsonl").read_bytes()
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
