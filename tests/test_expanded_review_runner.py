import copy
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kenggo import expanded_review as review
from kenggo.base_pilot import _ref
from kenggo.checkpoints import file_hash, save_json
from kenggo.expanded_pilot import ARMS, ROOT

import test_expanded_finish as fixtures


class ExpandedReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExpandedFinishTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.old_output = self.fixture.root, self.fixture.output
        self.plan = json.loads((ROOT / "configs/e2_review_rerun_v1.json").read_text())
        refs = self.plan["inputs"]
        prior = dict(self.fixture.failure, completion_status="completed", matched_training=True,
                     training_decision=self.fixture.config["inputs"]["training_decision"])
        save_json(self.old_output / "completion.json", prior)
        refs.update(original_plan=self.fixture.plan_ref, previous_result=_ref(self.old_output / "completion.json"),
                    training_decision=prior["training_decision"])
        self.plan["parameters"]["prior_ledger_sha256"] = file_hash(self.fixture.ledger_path)
        self.path = self.root / "review.json"
        save_json(self.path, self.plan)
        self.output = self.root / self.plan["parameters"]["output"]
        self.patcher = patch.object(review, "ROOT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def save_plan(self):
        save_json(self.path, self.plan, replace=True)
        return file_hash(self.path)

    def test_frozen_rerun_keeps_models_reference_rules_and_total_gpu_cap(self):
        original = review.validate_review(self.plan)
        stages, bundles = review.saved_models(self.plan, original)
        self.assertEqual(set(stages), set(ARMS))
        self.assertEqual(set(bundles), set(ARMS))
        for arm, bundle in bundles.items():
            self.assertIn(str(self.old_output / arm), bundle["checkpoint_manifest"])
        for section, key, value in (("limits", "wall_seconds", 604801),
                                    ("selection", "games_per_rank", 20),
                                    ("match_parameters", "conditioning_rank", "1d"),
                                    ("reference_budget", "queries", 59183)):
            changed = copy.deepcopy(self.plan)
            changed["parameters"][section][key] = value
            with self.subTest(section=section), self.assertRaises(ValueError):
                review.validate_review(changed)

    def test_changed_saved_training_is_rejected_before_any_job(self):
        original = review.validate_review(self.plan)
        path = self.old_output / ARMS[0] / "worker/checkpoint.json"
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "checkpoint manifest changed"):
            review.saved_models(self.plan, original)

    def test_only_evaluation_jobs_run_and_analysis_and_games_precede_reference(self):
        original = review.validate_review(self.plan)
        training, bundles = review.saved_models(self.plan, original)
        calls, stages = [], {}
        sets = []
        for identity, counts in review.EXPECTED_SETS.items():
            role, komi = identity.split("_komi")
            manifest = self.root / (identity + ".json")
            save_json(manifest, {})
            sets.append({"spec": {"id": identity, "role": role, "komi_halfpoints": int(komi),
                "source_ids": [f"{identity}-{index}" for index in range(counts["games"])],
                "aggregate_ranks": list(review.WEEK_REVIEW_RANKS)}, "rows": counts["rows"], "manifest": _ref(manifest)})

        def stage(name, config, kind, **options):
            calls.append((name, config, kind, options))
            worker = self.output / name / "worker"
            worker.mkdir(parents=True)
            stages[name] = {"directory": str(worker.parent)}
            if name == "preparation":
                result = {"common_sets": sets}
            elif name.startswith("evaluation-"):
                reports = []
                for item in sets:
                    path = worker / (item["spec"]["id"] + ".npz")
                    path.write_bytes(b"synthetic")
                    reports.append({"set": item["spec"]["id"], "variant": "ordinary",
                        "artifacts": {"predictions.npz": {"path": path.name, "sha256": file_hash(path)}}})
                result = {"reports": reports}
            else:
                result = {}
            return stages[name], result

        review.execute(self.plan, _ref(self.path), SimpleNamespace(stages=stages, stage=stage), training, bundles)
        self.assertEqual([name for name, *_ in calls], list(review.STAGES))
        self.assertTrue(all("device" not in options and "register" not in options for *_, options in calls))
        for name, config, kind, _ in calls:
            if name.startswith("evaluation-"):
                self.assertEqual(config["parameters"]["evaluation_scope"], "week_review_v1")
                self.assertEqual(len(config["parameters"]["common_sets"]), 2 if name.endswith("human-reference") else 4)
            if name.startswith("matches-"):
                self.assertEqual(config["parameters"]["recipe"], "week_review_matches_v1")
        self.assertEqual(calls[-1][1]["parameters"]["reference_budget"]["queries"], 59182)

    def test_later_failure_preserves_old_evidence_and_records_a_new_clock_and_charge(self):
        before = self.fixture.ledger.snapshot()["state"]["entries"]
        original_bytes = (self.old_output / "completion.json").read_bytes()
        def fail(plan, plan_ref, session, training, bundles):
            self.assertEqual(session.limits["cpu_seconds"], 518400 - 1800)
            session.failed_stage = {"name": "evaluation-human-reference", "artifacts": {}}
            raise RuntimeError("reference failed")
        with patch.object(review, "capture_provenance", return_value=({"source": {}}, b"")), \
                patch.object(review, "execute", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "reference failed"):
                review.run(self.path, file_hash(self.path), self.output)
        result = json.loads((self.output / "failure.json").read_text())
        self.assertEqual((self.old_output / "completion.json").read_bytes(), original_bytes)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["failed_stage"]["name"], "evaluation-human-reference")
        self.assertTrue(result["matched_training"])
        costs = result["costs"]
        elapsed_cap = datetime.fromisoformat(costs["deadline_utc"]) - datetime.fromisoformat(costs["launch_utc"])
        self.assertEqual(elapsed_cap.total_seconds(), 604800)
        state = costs["ledger"]["state"]
        self.assertEqual(state["entries"][:len(before)], before)
        self.assertFalse(state["reservations"])
        self.assertEqual(costs["ledger"]["charged_gpu_seconds"], 0)
        with self.assertRaisesRegex(ValueError, "ledger changed"):
            review.run(self.path, file_hash(self.path), self.output)

    def test_changed_or_reserved_ledger_cannot_create_a_run(self):
        self.fixture.ledger.reserve("foreign", estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
        with self.assertRaisesRegex(ValueError, "ledger changed"):
            review.run(self.path, file_hash(self.path), self.output)
        self.plan["parameters"]["prior_ledger_sha256"] = file_hash(self.fixture.ledger_path)
        with self.assertRaisesRegex(ValueError, "previous allocation"):
            review.run(self.path, self.save_plan(), self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
