"""The owner page must retain interruptions alongside completed comparisons."""

from pathlib import Path
import tempfile
import unittest

from kenggo.checkpoints import save_json
from kenggo.continuation_data import reference
from kenggo.continuation_data import read
from kenggo import learning_review, progressive_review


def interrupted_fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    def saved(name, value):
        target = root / (name + ".json")
        save_json(target, value)
        return reference(target)

    stages, selected = [], {}
    for arm in ("control", "candidate"):
        for epoch in range(2):
            diagnostics = {name: {"nll": 3. - epoch * .2, "top1": .3, "value_mse": .7,
                "rows": 12, "policy_count": 12, "value_count": 8, "aggregate_ranks": ["5k"],
                "diagnostic_sha256": pin * 64} for name, pin in (("training_monitor", "a"), ("validation", "b"))}
            metrics = saved(f"{arm}-{epoch}-metrics", {"round": epoch, "sampler_step": epoch * 4,
                "lifetime_updates": 31250 + epoch * 4, "new_raw_best": True, **diagnostics})
            checkpoint = saved(f"{arm}-{epoch}-checkpoint", {"arm": arm, "epoch": epoch})
            commit = saved(f"{arm}-{epoch}-commit", {"metrics": metrics, "checkpoint": checkpoint})
            progress = {"sampler_step": epoch * 4, "epoch_metrics": []}
            summary = saved(f"{arm}-{epoch}", {"status": "completed", "training_rows": 128,
                "progress": progress, "commits": [commit], "best_checkpoint": checkpoint})
            stages.append({"arm": arm, "epoch": epoch,
                           "stage": {"summary": summary, "status": "completed", "job": {"reason": None}}})
            selected[arm] = checkpoint
    partial = saved("candidate-partial", {"status": "partial", "training_rows": 256,
        "progress": {"sampler_step": 6, "epoch_metrics": []}, "commits": [],
        "best_checkpoint": selected["candidate"]})
    stages.append({"arm": "candidate", "epoch": 2,
                   "stage": {"summary": partial, "status": "partial", "job": {"reason": "CPU limit"}}})
    comparison = saved("comparison", {"comparisons": {}})
    return saved("summary", {"status": "budget_or_evidence_stop", "reason": "Interrupted interval",
        "deadline_utc": "2026-11-28T00:00:00+00:00", "failed_attempts": [{"name": "saved helper", "reason": "failure retained"}],
        "costs": {"elapsed_seconds": 100, "cpu_seconds": 50, "ledger": {"charged_gpu_seconds": 40}},
        "tiers": [{"tier": 1, "previous_rows": 128, "target_rows": 256, "training_rows": 256,
            "status": "partial_training", "training_stop": "interrupted_interval", "completed_intervals": 1,
            "comparison_status": "completed", "comparison": comparison, "selected": selected,
            "decision": {"continue_growth": False, "reasons": ["Interrupted training; stop growth"]},
            "incomplete_interval": {"epoch": 2, "arm": "candidate", "reason": "CPU limit", "stage": partial},
            "training": stages, "assessments": {}}]})


class ProgressiveReviewTests(unittest.TestCase):
    def test_summaryless_failure_is_visible_without_claiming_an_unfinished_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = interrupted_fixture(root)
            saved = read(summary)
            tier = saved["tiers"][0]
            tier["training"].pop()
            tier["incomplete_interval"].update(status="failed", reason="worker killed before summary")
            tier.pop("comparison_status")
            tier.pop("comparison")
            save_json(root / "failed.json", saved)
            ref = reference(root / "failed.json")
            data = progressive_review.readout(ref)
            self.assertEqual(data["training"]["tier1-candidate"]["status"], "failed")
            self.assertEqual(data["training"]["tier1-candidate"]["selected_sampler_step"], 4)
            page_ref = progressive_review.export(ref, root / "review.html", history_path=None)
            _, page = learning_review.read_review(page_ref)
            self.assertIn("worker killed before summary", page)
            self.assertIn("The fixed comparison did not finish.", page)
            self.assertNotIn("The fixed comparison uses the last completed pair.", page)

    def test_interrupted_attempt_and_selected_complete_pair_remain_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = interrupted_fixture(root)
            data = progressive_review.readout(summary)
            tier = data["tiers"][0]
            self.assertEqual(tier["status"], "partial_training")
            self.assertEqual(tier["comparison_status"], "completed")
            self.assertEqual(tier["incomplete_interval"]["reason"], "CPU limit")
            self.assertEqual(tier["attempts"][-1]["status"], "partial")
            self.assertEqual(tier["attempts"][-1]["epoch"], 2)
            self.assertEqual(data["failed_attempts"][0]["reason"], "failure retained")
            candidate = learning_review.build(data)["runs"][0]["training"]["tier1-candidate"]
            self.assertEqual(candidate["status"], "partial")
            self.assertEqual(candidate["updates"], 6)
            self.assertEqual(candidate["selected_step"], 4)
            self.assertEqual([point["step"] for point in candidate["checkpoints"]], [0, 4])
            page_ref = progressive_review.export(summary, root / "review.html", history_path=None)
            _, page = learning_review.read_review(page_ref)
            self.assertIn("Interrupted interval 2", page)
            self.assertIn("CPU limit", page)
            self.assertIn("The fixed comparison uses the last completed pair", page)
            with self.assertRaises(FileExistsError):
                progressive_review.export(summary, root / "review.html", history_path=None)


if __name__ == "__main__":
    unittest.main()
