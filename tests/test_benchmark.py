import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]

class BenchmarkCLITests(unittest.TestCase):
    def run_cli(self, *args):
        env = {"PYTHONPATH": str(ROOT / "src")}
        import os
        env.update(os.environ)
        return subprocess.run([sys.executable, "scripts/benchmark.py", *args], cwd=ROOT,
                              env=env, text=True, capture_output=True)

    def test_offline_smoke_records_all_capabilities_and_metadata(self):
        p = self.run_cli("--backend", "offline")
        self.assertEqual(p.returncode, 0, p.stderr)
        report = json.loads(p.stdout)
        self.assertEqual(report["requests"], 12)
        self.assertEqual(report["schema_validity"], 1.0)
        self.assertEqual(set(report["per_capability"]), {
            "model_routing", "confidence_action", "tool_screening", "progress", "completion",
            "skill_selection", "compaction", "citation", "rag_filter", "semantic_find",
            "composite", "intent_routing"})
        self.assertTrue(report["metadata"]["commit"])
        self.assertTrue(report["metadata"]["fixture_sha256"])

    def test_malformed_offline_row_is_reported(self):
        path = ROOT / "fixtures" / "_bad-benchmark.jsonl"
        path.write_text(json.dumps({"prediction": {}}) + "\n")
        try:
            p = self.run_cli("--backend", "offline", "--fixture", str(path))
            self.assertEqual(p.returncode, 2)
            self.assertEqual(json.loads(p.stdout)["errors"], 1)
        finally:
            path.unlink()

    def test_backend_outage_is_fail_closed_and_transport_stays_loopback(self):
        p = self.run_cli("--backend", "rules", "--url", "http://127.0.0.1:1")
        self.assertEqual(p.returncode, 2)
        self.assertEqual(json.loads(p.stdout)["errors"], 12)
        sys.path.insert(0, str(ROOT / "src"))
        from jev_laya_free.client import TypeSafeClient
        with self.assertRaises(Exception):
            TypeSafeClient(base_url="https://example.com")

if __name__ == "__main__": unittest.main()
