"""Check source-panel acceptance denominators independently of neural inference."""

from collections import Counter
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("endgame_summary", Path(__file__).parents[1] / "tools/summarize_endgame_referee.py")
summary_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary_tool)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def panel(self, resolved, winner_errors=0):
        source_ids = [f"fixture:{index}" for index in range(108)]
        passes = [{"source_url": source_ids[index], "pre_move_index": 100} for index in range(24)]
        self.write("panel.json", {"ending_source_ids": source_ids, "passes": passes})
        self.write("run.json", {"fixtures_only": False})
        self.write("fixtures.json", [])
        self.write("authorized_evidence.json", {})
        for index, source_id in enumerate(source_ids):
            accepted = index < resolved
            score = (-3 if index < winner_errors else 3) if accepted else None
            status = "resolved" if accepted else "unresolved"
            evidence = {"profile_sha256": "fixture-profile", "runs": [
                {"cleanup": [], "terminal": {"white_minus_black_halfpoints": score}}],
                "cost": {"cpu_seconds": 1, "wall_seconds": 1}}
            self.write(f"ending-{index:03d}.json", {
                "source_url": source_id, "recorded_result": {"kind": "scored", "margin": "1.5", "winner": "W"},
                "decision": {"status": status, "reason": status, "white_minus_black_halfpoints": score,
                             "evidence": json.dumps(evidence)}, "evidence_sha256": "fixture-evidence"})
        for index, selected in enumerate(passes):
            self.write(f"pass-{index:03d}.json", selected | {"color": "B", "prior_pass_streak": 0,
                "decision": {"verdict": "acceptable", "reason": "fixture", "gains_points": [0, 0]}})
        statuses = Counter(["resolved"] * resolved + ["unresolved"] * (108 - resolved))
        self.write("summary.json", {"profile_sha256": "fixture-profile", "endings": 108, "passes": 24,
            "ending_statuses": dict(statuses), "ending_reasons": dict(statuses),
            "pass_verdicts": {"acceptable": 24}, "pass_reasons": {"fixture": 24},
            "trace_sha256": "fixture-trace", "trace_bytes": 0, "cost": {"cpu_seconds": 108}})

    def test_missing_outcomes_remain_in_resolution_denominator(self):
        self.panel(97)
        result = summary_tool.summarize(self.directory)
        self.assertFalse(result["readiness_check"]["resolution_passes"])
        self.assertTrue(result["readiness_check"]["winner_agreement_passes"])
        self.assertEqual(result["ending_counts"]["unresolved"], 11)
        self.assertEqual(len(result["endings"]), 108)

    def test_winner_gate_uses_resolved_scored_comparisons(self):
        self.panel(98, winner_errors=5)
        result = summary_tool.summarize(self.directory)
        self.assertTrue(result["readiness_check"]["resolution_passes"])
        self.assertFalse(result["readiness_check"]["winner_agreement_passes"])
        self.assertEqual(result["ending_counts"]["known_scored_comparisons"], 98)
        self.assertEqual(result["ending_counts"]["winner_agrees"], 93)
        self.assertEqual(result["endings"][0]["margin_delta_halfpoints"], -6)
        self.panel(98, winner_errors=4)
        self.assertTrue(summary_tool.summarize(self.directory)["readiness_check"]["winner_agreement_passes"])

    def test_a_missing_or_replaced_case_cannot_disappear(self):
        self.panel(108)
        path = self.directory / "ending-050.json"
        item = json.loads(path.read_text())
        item["source_url"] = "replacement"
        self.write(path.name, item)
        with self.assertRaises(ValueError):
            summary_tool.summarize(self.directory)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            summary_tool.summarize(self.directory)
