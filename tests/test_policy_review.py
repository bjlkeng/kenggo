"""Policy review pages retain measured checkpoints, missingness and study scope."""

import json
from pathlib import Path
import tempfile
import unittest

from kenggo import learning_review, policy_review
from kenggo.checkpoints import save_json
from kenggo.continuation_data import read, reference


def saved(root, name, value):
    target = Path(root) / (name + ".json")
    save_json(target, value)
    return reference(target)


def metric(value):
    return {"status": "available", "estimate": {"status": "finite", "value": value},
        "counts": {"moves": 12, "games": 2, "accounts": 3}, "masked_moves": 4,
        "aggregate_ranks": ["5k"], "missing_ranks": [],
        "uncertainty": {"status": "inconclusive", "interval95": None, "reasons": ["few_accounts"]},
        "side_weights": ["large raw weights must not enter the page"]}


def fixture(root):
    root = Path(root)
    metrics = {"nll": metric(2.5), "top1": metric(.3), "value_mse": metric(.7)}
    report = {"aggregate_ranks": ["5k"], "expected_outputs": 12, "overall": {"metrics": metrics}}
    previous = {"bundle_id": "previous", "training": {}, "evaluations": {
        "validation_development_komi13": {"prior-policy": {"metrics": report}}}}
    history_page = root / "previous.html"
    history_page.write_text('<html><main></main><script id="expanded-data" type="application/json">'
                            + json.dumps(previous) + '</script></html>')
    history = saved(root, "history", {"schema": "e2_review_history_v1", "runs": [{
        "id": "previous", "bundle_id": "previous", "source": reference(history_page),
        "title": "Earlier run", "short": "Earlier", "description": "Earlier model; different sample.",
        "notes": ["Evaluation rerun; no new training."]}]})
    plan = saved(root, "plan", {"inputs": {"review_history": history}, "parameters": {"training_rows": 16}})
    parent = saved(root, "parent", {"model": "parent"})
    trials = []
    for identity, rate, selected_round in (("baseline", .001, 1), ("small-rate", .0003, 0)):
        stages, checkpoints, online = [], [], []
        for epoch in (0, 1):
            diagnostics = {name: {"nll": 3. - epoch * .2, "top1": .3 + epoch * .02,
                "value_mse": .7 + epoch * .1, "rows": 12, "policy_count": 12, "value_count": 8,
                "aggregate_ranks": ["5k"], "diagnostic_sha256": pin * 64, "evaluated_rows": 14,
                "excluded_rank_rows": 2, "weighting": "equal ranks, game sides and valid moves"}
                for name, pin in (("training_monitor", "a"), ("validation", "b"))}
            point = saved(root, f"{identity}-{epoch}-metrics", {"round": epoch, "sampler_step": epoch * 4,
                "lifetime_updates": 100 + epoch * 4, "new_raw_best": epoch <= selected_round,
                "learning_rate_used": rate, "learning_rate_next": rate / (2 if epoch else 1),
                "value_loss_weight": 0, **diagnostics})
            checkpoints.append(saved(root, f"{identity}-{epoch}-checkpoint", {"trial": identity, "epoch": epoch}))
            best = checkpoints[min(epoch, selected_round)]
            commit = saved(root, f"{identity}-{epoch}-commit", {"version": "continuation_epoch_commit_v1",
                "round": epoch, "checkpoint": checkpoints[-1], "metrics": point, "best_checkpoint": best})
            if epoch:
                online.append({"epoch": 1, "next_step": 4, "updates": 4, "presented_examples": 32,
                               "policy_count": 32, "value_count": 24, "policy_loss": 2.9, "value_loss": .72})
            worker = saved(root, f"{identity}-{epoch}-worker", {"status": "completed", "training_rows": 16,
                "completed_updates": epoch * 4, "presented_examples": epoch * 32,
                "progress": {"sampler_step": epoch * 4, "epoch_metrics": online,
                             "partial_round": {"updates": 0, "presented_examples": 0}},
                "commits": [commit], "best_checkpoint": best})
            stages.append({"epoch": epoch, "stage": {"status": "completed", "summary": worker}})
        partial = saved(root, f"{identity}-partial", {"status": "partial", "reason": "CPU limit",
            "training_rows": 16, "completed_updates": 2, "presented_examples": 16,
            "progress": {"sampler_step": 6, "epoch_metrics": online,
                         "partial_round": {"updates": 2, "presented_examples": 16}},
            "commits": [], "best_checkpoint": checkpoints[selected_round]})
        stages.append({"epoch": 2, "stage": {"status": "partial", "summary": partial,
                                              "job": {"reason": "CPU limit"}}})
        trials.append({"id": identity, "parameters": {"learning_rate": rate, "weight_decay": .01},
            "training": stages, "selected": checkpoints[selected_round], "status": "partial",
            "stop_reason": "budget", "unused_predictions": ["raw marker"]})
    diagnostic_refs = {}
    for name, role in (("train_monitor", "train"), ("epoch_validation", "validation_development")):
        diagnostic_refs[name] = saved(root, name, {"purpose": name, "role": role,
            "aggregate_ranks": ["5k"], "source_ids": [name + "-game-1", name + "-game-2"],
            "rows": [{"row_key": f"raw-position-{index}"} for index in range(14)]})
    selection = saved(root, "selection", {"schema": "policy_stopping_selection_v1", "per_cell": 4,
        "unique_games": 2, "expected_rows": 14, "cells": [{"rank": "5k", "komi_halfpoints": 13,
            "available_games": 2, "selected_games": 2, "target_games": 2,
            "selected_mover_accounts": ["account1", "account2"], "below_requested_quota": True}]})
    preparation = saved(root, "preparation", {"schema": "e2_policy_preparation_v1", "status": "completed",
        "coverage": {"5k": {"games": 2}}, "selection": selection, "assessments": {}, **diagnostic_refs})
    assessment_report = saved(root, "assessment-report", {"report": report, "all_rank_report": report,
        "excluded_rank_rows": 2, "evaluated_rows": 14,
        "predictions": [{"row_key": "raw prediction must not enter the page", "nll": 1.}]})
    assessment = saved(root, "assessment", {"schema": "progressive_assessment_v1", "status": "completed",
        "reports": {"validation_development_komi13": assessment_report}})
    paired = {"kind": "candidate_minus_control", "aggregate_ranks": ["5k"], "rows": 12,
        "excluded_rows": 2, "metrics": {"nll": metric(-.12), "top1": metric(.02), "value_mse": metric(.4)}}
    paired["metrics"]["nll"]["uncertainty"] = {"status": "available", "interval95": [-.2, -.04]}
    comparison = saved(root, "comparison", {"status": "completed", "selected_trial_id": "small-rate",
        "baseline_trial_id": "baseline",
        "comparisons": {"small-rate": {"against_parent": {"validation_development_komi13": paired},
                                      "against_baseline": {"validation_development_komi13": paired}}},
        "notes": ["Saved paired comparison"]})
    return saved(root, "study", {"schema": "e2_policy_study_v1", "status": "budget_stop",
        "reason": "Training hit its allowance", "plan": plan, "parent_checkpoint": parent,
        "launch_utc": "2026-11-25T00:00:00+00:00", "deadline_utc": "2026-11-28T00:00:00+00:00",
        "limits": {"wall_seconds": 259200}, "costs": {"elapsed_seconds": 100, "cpu_seconds": 50,
            "ledger": {"charged_gpu_seconds": 40}}, "trials": trials, "preparation": preparation,
        "assessments": {"parent": assessment, "small-rate": assessment}, "comparison": comparison,
        "selected_trial_id": "small-rate", "failed_attempts": [{"reason": "Earlier worker failed"}]})


class PolicyReviewTests(unittest.TestCase):
    def test_committed_curves_keep_epoch_zero_selection_and_partial_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = fixture(temporary)
            data = policy_review.readout(summary)
            run = learning_review.build(data)["runs"][0]
            trial = run["training"]["small-rate"]
            self.assertEqual([point["step"] for point in trial["checkpoints"]], [0, 4])
            self.assertEqual([point["step"] for point in trial["points"]], [4])
            self.assertEqual(trial["selected_step"], 0)
            self.assertEqual(trial["selected_round"], 0)
            self.assertEqual(trial["selected_lifetime_updates"], 100)
            self.assertEqual(trial["updates"], 6)
            self.assertEqual(trial["presentations"], 48)
            self.assertEqual(trial["checkpoints"][1]["training_monitor"]["top1"], .32)
            self.assertEqual(trial["checkpoints"][1]["validation"]["value_count"], 8)
            details = data["trials"][1]
            self.assertEqual(details["attempts"][-1]["reason"], "CPU limit")
            self.assertEqual(details["learning_rates"][-1]["learning_rate_next"], .00015)
            self.assertEqual(details["learning_rates"][0]["value_loss_weight"], 0)

    def test_page_keeps_pinned_history_fixed_metrics_and_diagnostic_only_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = fixture(root)
            page_ref = policy_review.export(summary, root / "review.html")
            data, page = learning_review.read_review(page_ref)
            self.assertIn('id="learning-review"', page)
            self.assertIn('id="lr-policy-chart"', page)
            self.assertIn('id="lr-agreement-chart"', page)
            self.assertIn('id="lr-value-chart"', page)
            self.assertIn("Earlier model; different sample.", page)
            self.assertIn("diagnostic only", page)
            self.assertIn("do not select, guard, or promote", page)
            self.assertIn("0.00015", page)
            self.assertIn("against_parent", page)
            self.assertIn("against_baseline", page)
            self.assertIn("Baseline trial: baseline", page)
            self.assertIn("-0.1200", page)
            self.assertIn("[-0.2000, -0.0400]", page)
            self.assertIn("CPU limit", page)
            self.assertIn("Earlier worker failed", page)
            self.assertIn("<ol>", page)
            self.assertIn("kenggo-ydo.19", page)
            self.assertNotIn("raw prediction must not enter", page)
            self.assertNotIn("large raw weights", page)
            self.assertNotIn("raw-position-", page)
            self.assertLess(len(page), 100_000)
            self.assertEqual(data["preparation"]["diagnostics"]["epoch_validation"]["rows"], 14)
            self.assertEqual(data["preparation"]["selection"]["cells"][0]["selected_mover_account_count"], 2)
            self.assertIn("Stopping-validation coverage", page)
            self.assertIn("Below requested quota", page)
            self.assertEqual(data["comparison"]["comparisons"]["small-rate"]["against_parent"]
                ["validation_development_komi13"]["metrics"]["nll"]["estimate"]["value"], -.12)
            manifest = read(reference(root / "review.json"))
            expected = read(read(summary)["plan"])["inputs"]["review_history"]
            self.assertEqual(manifest["history"], expected)
            self.assertEqual(len(manifest["history_sources"]), 1)
            self.assertEqual(len(manifest["code"]), 3)

    def test_publication_snapshot_cost_scope_links_the_final_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = read(fixture(root))
            summary["costs_scope"] = "through training and comparison; final publication costs are in summary.json"
            ref = saved(root, "review-input", summary)
            page_ref = policy_review.export(ref, root / "snapshot-review.html", history_path=None)
            data, page = learning_review.read_review(page_ref)
            self.assertEqual(data["costs_scope"], summary["costs_scope"])
            self.assertIn("Cost scope:</strong> through training and comparison", page)
            self.assertIn("Final study cost record, including publication", page)
            self.assertIn(f'href="{root / "summary.json"}"', page)

    def test_failed_preparation_and_summaryless_attempt_remain_reviewable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = read(fixture(root))
            summary.update(trials=[], assessments={}, status="failed", reason="Validation coverage is insufficient")
            for key in ("preparation", "comparison", "selected_trial_id"):
                summary.pop(key)
            failed = saved(root, "failed", summary)
            page_ref = policy_review.export(failed, root / "failure.html", history_path=None)
            data, page = learning_review.read_review(page_ref)
            self.assertEqual(data["training"], {})
            self.assertIn("Validation coverage is insufficient", page)
            self.assertIn("No new training updates are recorded", page)
            self.assertIn("No fixed comparison measurements are available", page)
            summary["trials"] = [{"id": "failed-trial", "parameters": {"learning_rate": .001},
                "status": "failed", "stop_reason": "Worker killed", "training": [{"epoch": 0,
                    "stage": {"status": "failed", "job": {"reason": "Killed before summary"}}}]}]
            no_worker = saved(root, "no-worker", summary)
            data = policy_review.readout(no_worker)
            self.assertEqual(data["training"], {})
            self.assertEqual(data["trials"][0]["attempts"][0]["reason"], "Killed before summary")
            policy_review.export(no_worker, root / "no-worker-review.html", history_path=None)

    def test_partial_trial_without_committed_epoch_does_not_create_a_curve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = read(fixture(root))
            worker = saved(root, "early-partial", {"status": "partial", "training_rows": 16,
                "progress": {"sampler_step": 2, "epoch_metrics": [],
                             "partial_round": {"updates": 2, "presented_examples": 16}}, "commits": []})
            summary.update(trials=[{"id": "partial", "parameters": {}, "status": "partial",
                "training": [{"epoch": 0, "stage": {"status": "partial", "summary": worker}}]}], assessments={})
            for key in ("comparison", "selected_trial_id"):
                summary.pop(key)
            ref = saved(root, "early", summary)
            data = policy_review.readout(ref)
            self.assertEqual(data["training"], {})
            self.assertEqual(data["trials"][0]["completed_updates"], 2)
            self.assertEqual(data["trials"][0]["presented_examples"], 16)
            page_ref = policy_review.export(ref, root / "partial.html", history_path=None)
            _, page = learning_review.read_review(page_ref)
            self.assertIn("No committed checkpoint measurements", page)
            self.assertNotIn("No new training updates are recorded", page)

    def test_missing_selected_measurement_and_mutated_history_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref = fixture(root)
            summary = read(ref)
            summary["trials"][0]["selected"] = summary["parent_checkpoint"]
            invalid = saved(root, "invalid", summary)
            with self.assertRaisesRegex(ValueError, "selected checkpoint"):
                policy_review.readout(invalid)
            (root / "history.json").write_text('{"schema":"e2_review_history_v1","runs":[]}')
            with self.assertRaises(ValueError):
                policy_review.export(ref, root / "changed-history.html")
            policy_review.export(ref, root / "no-history.html", history_path=None)

    def test_sealed_pages_and_existing_manifests_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = fixture(root)
            target = root / "review.html"
            ref = policy_review.export(summary, target, history_path=None)
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                policy_review.export(summary, target, history_path=None)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(reference(target), ref)
            target.unlink()
            with self.assertRaises(FileExistsError):
                policy_review.export(summary, target, history_path=None)
            self.assertFalse(target.exists())

    def test_unavailable_comparison_is_not_a_zero_gain_and_html_is_escaped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = read(fixture(root))
            comparison = read(summary["comparison"])
            point = comparison["comparisons"]["small-rate"]["against_parent"]["validation_development_komi13"]
            point["metrics"]["nll"] = {"status": "unavailable", "missing_ranks": ["5k"]}
            summary["comparison"] = saved(root, "unavailable-comparison", comparison)
            summary["reason"] = '</script><img src=x onerror="alert(1)">'
            ref = saved(root, "escaped", summary)
            page_ref = policy_review.export(ref, root / "escaped-review.html", history_path=None)
            data, page = learning_review.read_review(page_ref)
            self.assertIn("Unavailable", page)
            self.assertNotIn('<img src=x', page)
            self.assertEqual(data["reason"], summary["reason"])
            missing = data["comparison"]["comparisons"]["small-rate"]["against_parent"]
            self.assertNotIn("estimate", missing["validation_development_komi13"]["metrics"]["nll"])


if __name__ == "__main__":
    unittest.main()
