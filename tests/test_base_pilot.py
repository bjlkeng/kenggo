import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from kenggo.base_pilot import ROOT, _ref, _report_metrics, _stage, _worker_reports, run
from kenggo.checkpoints import json_hash, save_checkpoint, save_json
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import Experiment
from test_tracking import CONFIG, PROVENANCE, Sink, components


class BasePilotTests(unittest.TestCase):
    def test_supervisor_exception_keeps_settled_costs_without_counting_them_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = BudgetLedger.create(root / "ledger.json", budget_id="fixture", gpu_seconds=10)

            def failed_launch(*args, **kwargs):
                time.sleep(.2)
                raise OSError("synthetic launch failure")

            def experiment(*args, **kwargs):
                return Experiment(*args, **kwargs, sink_factory=Sink)

            started = time.monotonic()
            with mock.patch("kenggo.base_pilot.capture_provenance", return_value=(PROVENANCE, b"")), \
                 mock.patch("kenggo.base_pilot.Experiment", side_effect=experiment), \
                 mock.patch("kenggo.pilot_jobs.subprocess.Popen", side_effect=failed_launch):
                with self.assertRaisesRegex(OSError, "synthetic launch failure"):
                    _stage(root, "stage", CONFIG,
                        {"wall_seconds": 3, "cpu_seconds": 3, "output_bytes": 1000000, "shutdown_seconds": 1},
                        ledger, module="kenggo.baseline_matches", device="synthetic-gpu")
            elapsed = time.monotonic() - started
            snapshot = ledger.snapshot()
            self.assertLess(sum(item["wall_seconds"] for item in snapshot["state"]["entries"]), elapsed + .05)
            self.assertGreater(snapshot["charged_gpu_seconds"], .19)
            metrics = [json.loads(line)["metrics"] for line in (root / "stage/metrics.jsonl").read_text().splitlines()]
            self.assertEqual(metrics[0]["job/status"], "failed")
            self.assertEqual(metrics[0]["job/gpu_seconds"], snapshot["charged_gpu_seconds"])

    def stage(self, root, *, status="completed", training=True, fail_shutdown=False, evaluation=False):
        config = copy.deepcopy(CONFIG)
        config["purpose"] = "baseline-training" if training else "baseline-evaluation" if evaluation else "baseline-matches"
        ledger = BudgetLedger.create(root / "ledger.json", budget_id="fixture", gpu_seconds=10)

        def job(command, output, **kwargs):
            output.mkdir()
            kwargs["ledger"].reserve(kwargs["run_id"], estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
            kwargs["ledger"].record(kwargs["run_id"], allocations=[], outcome="completed",
                                   arm_shares={"fixture": 1}, cpu_seconds=0, wall_seconds=0)
            if training:
                ref = save_checkpoint(output / "checkpoints", *components(), step=1, progress={"next_step": 1},
                    config_sha256=json_hash(config), provenance_sha256=json_hash(PROVENANCE))
                summary = {"status": status, "checkpoints": [{"path": str(Path(ref["path"]).relative_to(output)),
                           "sha256": ref["sha256"], "step": 1, "epoch": 1}],
                           "epoch_metrics": [{"epoch": 1, "policy_loss": 1.0, "value_count": 2}]}
            elif evaluation:
                metrics = {"overall": {"metrics": {"nll": {"estimate": {"status": "finite", "value": .25},
                           "uncertainty": {"status": "inconclusive", "interval95": None}}}}}
                artifacts = {}
                for name, value in (("metrics.json", metrics), ("sampling.json", {"overall": {"pairwise_disagreement": .5}}),
                                    ("timing.json", {"measurements": [{"seconds": .1}], "unit": "learned_batch"})):
                    artifacts[name] = {"path": name, "sha256": save_json(output / name, value)}
                summary = {"status": status, "reports": [{"set": "population", "variant": "ordinary", "artifacts": artifacts}]}
            else:
                summary = {"status": status, "scheduled_games": 12, "recorded_games": 0}
            save_json(output / ("training_summary.json" if training else "summary.json"), summary)
            return {"status": "completed", "costs": {"wall_seconds": 0, "cpu_seconds": 0},
                    "parent_cpu_seconds": 0, "child_cpu_seconds": 0}

        class FailingSink(Sink):
            def finish(self, status, summary):
                raise RuntimeError("tracking shutdown failed")

        def experiment(*args, **kwargs):
            return Experiment(*args, **kwargs, sink_factory=FailingSink if fail_shutdown else Sink)

        with mock.patch("kenggo.base_pilot.capture_provenance", return_value=(PROVENANCE, b"")), \
             mock.patch("kenggo.base_pilot.Experiment", side_effect=experiment), \
             mock.patch("kenggo.base_pilot.run_job", side_effect=job):
            result = _stage(root, "stage", config,
                {"wall_seconds": 3, "cpu_seconds": 3, "output_bytes": 1000000, "shutdown_seconds": 1},
                ledger, module="kenggo.baseline_training" if training else "kenggo.baseline_evaluation" if evaluation
                else "kenggo.baseline_matches", register=training)
        return result, ledger.snapshot()

    def test_evaluation_metrics_and_support_status_reach_the_tracking_sink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.stage(root, training=False, evaluation=True)
            records = [json.loads(line)["metrics"] for line in (root / "stage/metrics.jsonl").read_text().splitlines()]
            metrics = {key: value for record in records for key, value in record.items()}
            prefix = "reports/population/ordinary/metrics/overall/metrics/nll/"
            self.assertEqual(metrics[prefix + "estimate/value"], .25)
            self.assertEqual(metrics[prefix + "uncertainty/status"], "inconclusive")
            self.assertEqual(metrics["reports/population/ordinary/sampling/overall/pairwise_disagreement"], .5)
            self.assertEqual(metrics["job/gpu_seconds"], 0)
            state = json.loads((root / "stage/run.json").read_text())
            self.assertIn("population/ordinary", state["summary"]["worker_reports"])

    def test_paired_and_match_reports_retain_unresolved_counts_and_verify_result_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paired = root / "paired.json"
            save_json(paired, {"overall": {"metrics": {"nll": {
                "estimate": {"status": "positive_infinity", "value": None},
                "uncertainty": {"status": "inconclusive"}}}}})
            summary = {"comparisons": {"validation": {"conditioned_minus_control": {"report": _ref(paired)}}}}
            config = {"purpose": "baseline-analysis"}
            reports = _worker_reports(config, root, summary)
            metrics = _report_metrics(reports)
            prefix = "reports/validation/conditioned_minus_control/overall/metrics/nll/"
            self.assertEqual(metrics[prefix + "estimate/status"], "positive_infinity")
            self.assertNotIn(prefix + "estimate/value", metrics)
            pin = save_json(root / "scores.json", {"arms": {"conditioned": {"counts": {"U": 4, "W": 0}}}})
            matches = _worker_reports({"purpose": "baseline-matches"}, root,
                {"scores_sha256": pin, "behavior": {"conditioned": {"flags": {"unresolved": {"adverse": 4}}}}})
            metrics = _report_metrics(matches)
            self.assertEqual(metrics["reports/scores/arms/conditioned/counts/U"], 4)
            self.assertEqual(metrics["reports/behavior/conditioned/flags/unresolved/adverse"], 4)
            paired.write_text("{}")
            with self.assertRaisesRegex(ValueError, "expected hash"):
                _worker_reports(config, root, summary)

    def test_training_worker_checkpoints_and_epoch_metrics_register_with_the_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (stage, summary), ledger = self.stage(root)
            state = json.loads((root / "stage/run.json").read_text())
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["checkpoints"][0]["step"], 1)
            self.assertIsNotNone(state["checkpoints"][0]["wandb"])
            metrics = [json.loads(line) for line in (root / "stage/metrics.jsonl").read_text().splitlines()]
            self.assertEqual(metrics[0]["metrics"]["training/policy_loss"], 1.0)
            self.assertFalse(ledger["state"]["reservations"])
            self.assertEqual(len(ledger["state"]["entries"]), 2)

    def test_unfilled_matches_remain_partial_and_keep_the_scheduled_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (_, summary), ledger = self.stage(root, status="incomplete", training=False)
            self.assertEqual(summary["scheduled_games"], 12)
            self.assertEqual(json.loads((root / "stage/run.json").read_text())["status"], "partial")
            self.assertEqual(ledger["state"]["entries"][-1]["outcome"], "partial")

    def test_incomplete_training_stops_instead_of_becoming_a_matched_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "declared units"):
                self.stage(root, status="incomplete")
            self.assertEqual(json.loads((root / "stage/run.json").read_text())["status"], "failed")
            ledger = BudgetLedger.open(root / "ledger.json", budget_id="fixture", gpu_seconds=10).snapshot()
            self.assertFalse(ledger["state"]["reservations"])

    def test_late_tracking_failure_settles_parent_costs_as_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
                self.stage(root, fail_shutdown=True)
            ledger = BudgetLedger.open(root / "ledger.json", budget_id="fixture", gpu_seconds=10).snapshot()
            self.assertEqual(ledger["state"]["entries"][-1]["outcome"], "failed")
            self.assertFalse(ledger["state"]["reservations"])

    def wiring(self, root, *, mismatch=False):
        plan = json.loads((ROOT / "configs/e2_base_plan_v1.json").read_text())
        plan["parameters"]["ledger"] = str(root / "pilot-ledger.json")
        path = root / "plan.json"
        pin = save_json(path, plan)
        seen = []

        def stage(output, name, config, limits, ledger, **kwargs):
            seen.append((name, config, kwargs))
            directory = output / name
            worker = directory / "worker"
            worker.mkdir(parents=True)
            save_json(directory / "config.json", config)
            save_json(directory / "provenance.json", {"inputs": config["inputs"]})
            result = {"directory": str(directory), "config": _ref(directory / "config.json"),
                      "provenance": _ref(directory / "provenance.json")}
            if name == "preparation":
                items = []
                for spec in plan["parameters"]["common_sets"]:
                    common = worker / (spec["id"] + ".json")
                    save_json(common, spec)
                    items.append({"spec": spec, "manifest": _ref(common)})
                summary = {"common_sets": items}
            elif name in ("conditioned", "unconditioned"):
                checkpoint = worker / "checkpoint.json"
                save_json(checkpoint, {"fixture": name})
                summary = {"final_checkpoint": {"path": checkpoint.name, "sha256": _ref(checkpoint)["sha256"]},
                    "initial_tensor_sha256": name if mismatch else "same", "row_keys_sha256": "same",
                    "completed_updates": 3600, "presented_examples": 458320}
            elif name.startswith("evaluation-"):
                reports = []
                variants = ("ordinary", "unknown_ranks") if config["arm"] == "conditioned" else ("ordinary",)
                for spec in config["parameters"]["common_sets"]:
                    for variant in variants:
                        target = worker / spec["id"] / variant / "predictions.npz"
                        target.parent.mkdir(parents=True)
                        target.write_bytes(b"synthetic wiring fixture")
                        reports.append({"set": spec["id"], "variant": variant,
                            "artifacts": {"predictions.npz": {"path": str(target.relative_to(worker)),
                                                               "sha256": _ref(target)["sha256"]}}})
                summary = {"reports": reports}
            else:
                summary = {"status": "completed"}
            return result, summary

        with mock.patch("kenggo.base_pilot._stage", side_effect=stage):
            result = run(path, pin, root / "pilot")
        return result, seen

    def test_frozen_sets_and_final_checkpoints_flow_to_all_controls_before_paired_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, seen = self.wiring(Path(temporary))
            self.assertEqual([item[0] for item in seen], ["preparation", "conditioned", "unconditioned",
                "evaluation-conditioned", "evaluation-unconditioned", "evaluation-human-reference", "matches", "analysis"])
            self.assertTrue(result["matched_training"])
            self.assertEqual([name for name, _, kwargs in seen if kwargs.get("device")], ["conditioned", "unconditioned"])
            evaluations = [config for name, config, _ in seen if name.startswith("evaluation-")]
            self.assertTrue(all(config["parameters"]["common_sets"] == evaluations[0]["parameters"]["common_sets"]
                                for config in evaluations))
            analysis = seen[-1][1]
            for spec in analysis["parameters"]["common_sets"]:
                self.assertEqual(set(spec["prediction_inputs"]),
                                 {"conditioned", "unconditioned", "human-reference", "conditioned_unknown_ranks"})

    def test_unmatched_initialization_aborts_before_any_validation_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "matched initialization"):
                self.wiring(root, mismatch=True)
            failure = json.loads((root / "pilot/failure.json").read_text())
            self.assertEqual(set(failure["stages"]), {"preparation", "conditioned", "unconditioned"})


if __name__ == "__main__":
    unittest.main()
