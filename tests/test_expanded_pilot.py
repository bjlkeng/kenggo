import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kenggo.checkpoints import save_json
from kenggo.expanded_pilot import (ARMS, ROOT, Session, _ref, execute,
                                    freeze_training, matched_training, validate_plan)
from kenggo.resource_ledger import BudgetLedger


def probes():
    summaries, jobs = {}, {}
    for arm in ARMS:
        summaries[arm] = {"status": "completed", "completed_updates": 256,
            "presented_examples": 256 * 128, "initial_tensor_sha256": "a" * 64,
            "update_metrics": [{"phase": "warmup", "updates": 6, "seconds": 1.2,
                                "seconds_per_update": .2, "start_step": 0, "next_step": 6,
                                "presented_examples": 768}] + [
                {"phase": "measured", "updates": 50, "seconds": 10, "seconds_per_update": .2,
                 "start_step": 6 + 50 * index, "next_step": 56 + 50 * index,
                 "presented_examples": 6400} for index in range(5)]}
        jobs[arm] = {"status": "completed", "costs": {"wall_seconds": 151.2}}
    return summaries, jobs


class ExpandedPilotTests(unittest.TestCase):
    def plan(self):
        return json.loads((ROOT / "configs/e2_expanded_plan_v1.json").read_text())

    def test_frozen_training_uses_slowest_block_and_equal_full_cycles(self):
        summaries, jobs = probes()
        result = freeze_training(summaries, jobs, wall_seconds=9000)
        self.assertEqual(result["updates"], 31250)
        self.assertEqual(result["presentations_per_arm"], 4_000_000)
        self.assertFalse(result["probe_weights_reused"])
        summaries[ARMS[-1]]["update_metrics"][-1].update(seconds=50, seconds_per_update=1)
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            freeze_training(summaries, jobs, wall_seconds=9000)

    def test_incomplete_probe_and_changed_initialization_do_not_select_a_recipe(self):
        summaries, jobs = probes()
        summaries[ARMS[0]]["completed_updates"] = 255
        with self.assertRaises(ValueError):
            freeze_training(summaries, jobs, wall_seconds=9000)
        summaries, jobs = probes()
        summaries[ARMS[0]]["initial_tensor_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "initializations"):
            freeze_training(summaries, jobs, wall_seconds=9000)

    def test_matched_training_rejects_different_order_within_a_population(self):
        values = {arm: {"status": "completed", "initial_tensor_sha256": "a", "completed_updates": 5,
            "presented_examples": 640, "row_keys_sha256": arm.split('-')[0],
            "update_metrics": [{"order_sha256": arm.split('-')[0]}]} for arm in ARMS}
        self.assertTrue(matched_training(values))
        values["combined-unconditioned"]["update_metrics"][0]["order_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "training order"):
            matched_training(values)

    def test_inconsistent_probe_work_and_equal_but_unfinished_training_are_rejected(self):
        summaries, jobs = probes()
        summaries[ARMS[0]]["update_metrics"][-1]["seconds"] = 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            freeze_training(summaries, jobs, wall_seconds=9000)
        summaries, _ = probes()
        with self.assertRaisesRegex(ValueError, "frozen full-batch work"):
            matched_training(summaries, expected_updates=15625)

    def test_plan_keeps_old_charges_and_bounds_every_phase(self):
        plan = self.plan()
        validate_plan(plan)
        changed = copy.deepcopy(plan)
        changed["parameters"]["ledger"] = "runs/new-ledger.json"
        with self.assertRaises(ValueError):
            validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed["parameters"]["job_limits"]["training"]["wall_seconds"] += 1
        with self.assertRaises(ValueError):
            validate_plan(changed)

    def test_session_checks_original_deadline_and_preserves_failed_stage_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            ledger = BudgetLedger.create(output / "ledger.json", budget_id="fixture", gpu_seconds=86400)
            session = Session(output, self.plan(), ledger)
            session.deadline -= timedelta(days=2)
            with patch("kenggo.expanded_pilot._stage") as launch:
                with self.assertRaisesRegex(RuntimeError, "Remaining production"):
                    session.stage("late", {}, "probe", module="fake")
                launch.assert_not_called()
            session.close("failed", "original deadline")
            self.assertFalse(ledger.snapshot()["state"]["reservations"])
            self.assertEqual(ledger.snapshot()["state"]["entries"][0]["outcome"], "failed")

    def test_failure_before_stage_accounting_keeps_parent_cpu_and_failure_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            ledger = BudgetLedger.create(output / "ledger.json", budget_id="fixture", gpu_seconds=86400)
            with patch("kenggo.expanded_pilot.time.process_time", return_value=10) as cpu:
                session = Session(output, self.plan(), ledger)
                def fail(*args, **kwargs):
                    worker = output / "preparation/worker"
                    worker.mkdir(parents=True)
                    save_json(worker / "failure.json", {"status": "failed", "reason": "changed pin"})
                    cpu.return_value = 15
                    raise ValueError("changed pin")
                with patch("kenggo.expanded_pilot._stage", side_effect=fail):
                    with self.assertRaisesRegex(ValueError, "changed pin"):
                        session.stage("preparation", {}, "preparation", module="fake")
                session.close("failed", "changed pin")
                self.assertEqual(session.snapshot()["cpu_seconds"], 15)
                self.assertEqual(session.failed_stage["artifacts"]["worker/failure.json"],
                                 _ref(output / "preparation/worker/failure.json"))
                self.assertFalse(session.stages)

    def test_cpu_accounting_includes_setup_children_without_counting_workers_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            ledger = BudgetLedger.create(output / "ledger.json", budget_id="fixture", gpu_seconds=86400)
            with patch("kenggo.expanded_pilot.time.process_time", return_value=10) as cpu, \
                    patch("kenggo.expanded_pilot._usage", return_value=(100, 0, set())) as tree:
                session = Session(output, self.plan(), ledger)
                ledger.reserve("worker", estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
                ledger.record("worker", allocations=[], outcome="completed", arm_shares={"shared": 1},
                              cpu_seconds=20, wall_seconds=20)
                cpu.return_value = 12
                tree.return_value = (127, 0, set())
                self.assertEqual(session.snapshot()["cpu_seconds"], 37)
                session.close("completed", None)
                self.assertEqual(session.snapshot()["cpu_seconds"], 37)

    def test_completion_preserves_the_original_clock_and_counts_only_new_cpu_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            ledger = BudgetLedger.create(output / "ledger.json", budget_id="fixture", gpu_seconds=86400)
            initial = ledger.snapshot()
            ledger.reserve("old-worker", estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
            ledger.record("old-worker", allocations=[], outcome="completed", arm_shares={"shared": 1},
                          cpu_seconds=20, wall_seconds=20)
            old_entries = ledger.snapshot()["state"]["entries"]
            started = datetime.now(timezone.utc) - timedelta(hours=12)
            origin = {"launch_utc": started.isoformat(), "deadline_utc": (started + timedelta(days=1)).isoformat(),
                      "limits": self.plan()["parameters"]["limits"], "ledger_before": initial}
            save_json(output / "progress.json", {"immutable": True})
            with patch("kenggo.expanded_pilot.time.process_time", return_value=10), \
                    patch("kenggo.expanded_pilot._usage", return_value=(100, 0, set())) as tree:
                session = Session(output, self.plan(), ledger, origin=origin, phase="completion")
                tree.return_value = (107, 0, set())
                state = session.snapshot()
                self.assertGreaterEqual(state["elapsed_seconds"], 12 * 3600)
                self.assertEqual(state["deadline_utc"], origin["deadline_utc"])
                self.assertEqual(state["cpu_seconds"], 37)
                with patch("kenggo.expanded_pilot._stage", return_value=({}, {"status": "completed"})):
                    session.stage("analysis", {}, "analysis", module="fixture")
                session.close("completed", None)
            self.assertEqual(ledger.snapshot()["state"]["entries"][:1], old_entries)
            self.assertEqual(json.loads((output / "progress.json").read_text()), {"immutable": True})
            self.assertTrue((output / "completion-progress.json").exists())
            self.assertEqual(session.snapshot()["cpu_seconds"], 37)

    def test_completion_cannot_restore_spent_cpu_or_extend_the_original_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            ledger = BudgetLedger.create(output / "ledger.json", budget_id="fixture", gpu_seconds=86400)
            initial = ledger.snapshot()
            ledger.reserve("spent", estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
            ledger.record("spent", allocations=[], outcome="failed", arm_shares={"shared": 1},
                          cpu_seconds=115190, wall_seconds=20, reason="spent CPU")
            started = datetime.now(timezone.utc) - timedelta(hours=12)
            origin = {"launch_utc": started.isoformat(), "deadline_utc": (started + timedelta(days=1)).isoformat(),
                      "limits": self.plan()["parameters"]["limits"], "ledger_before": initial}
            changed = copy.deepcopy(origin)
            changed["deadline_utc"] = (started + timedelta(days=2)).isoformat()
            with self.assertRaisesRegex(ValueError, "deadline must remain unchanged"):
                Session(output, self.plan(), ledger, origin=changed, phase="completion")
            session = Session(output, self.plan(), ledger, origin=origin, phase="completion")
            with patch("kenggo.expanded_pilot._stage") as launch:
                with self.assertRaisesRegex(RuntimeError, "Remaining production"):
                    session.stage("analysis", {}, "analysis", module="fixture")
                launch.assert_not_called()
            session.close("failed", "spent CPU")

    def test_four_arm_wiring_preserves_training_fits_and_reference_scope(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls, stages = [], {}
            def fake_stage(name, config, kind, **options):
                directory = root / name
                worker = directory / "worker"
                worker.mkdir(parents=True)
                for filename in ("config.json", "provenance.json"):
                    save_json(directory / filename, config if filename == "config.json" else {})
                stage = {"directory": str(directory), "config": _ref(directory / "config.json"),
                         "provenance": _ref(directory / "provenance.json"),
                         "job": {"status": "completed", "costs": {"wall_seconds": 151.2}}}
                p = config["parameters"]
                if kind == "preparation":
                    sets = []
                    for role in ("population_development", "validation_development"):
                        for komi in (13, 1):
                            key = f"{role}_komi{komi}"
                            save_json(worker / (key + ".json"), {})
                            sets.append({"spec": {"id": key, "role": role, "source_ids": [key],
                                "aggregate_ranks": ["1d"], "komi_halfpoints": komi},
                                "manifest": _ref(worker / (key + ".json")), "rows": 3})
                    summary = {"status": "completed", "common_sets": sets}
                elif kind in {"probe", "training"}:
                    save_json(worker / "checkpoint.json", {})
                    summary = probes()[0][ARMS[0]]
                    summary.update(final_checkpoint={"path": "checkpoint.json", "sha256": file_hash_local(worker / "checkpoint.json")},
                        completed_updates=p["updates"], presented_examples=p["updates"] * 128,
                        row_keys_sha256=str(p["training_rows"]))
                    for block in summary["update_metrics"]:
                        block["order_sha256"] = str(p["training_rows"])
                elif kind in {"learned_evaluation", "human_evaluation"}:
                    reports = []
                    for spec in p["common_sets"]:
                        path = worker / (spec["id"] + ".npz")
                        path.write_bytes(b"synthetic prediction archive")
                        reports.append({"set": spec["id"], "variant": "ordinary",
                            "artifacts": {"predictions.npz": {"path": path.name,
                                                               "sha256": file_hash_local(path)}}})
                    summary = {"status": "completed", "reports": reports}
                else:
                    summary = {"status": "completed"}
                save_json(worker / "summary.json", summary)
                stage["summary"] = _ref(worker / "summary.json")
                stages[name] = stage
                calls.append((name, config, kind))
                return stage, summary
            from types import SimpleNamespace
            session = SimpleNamespace(stage=fake_stage, stages=stages)
            save_json(root / "plan.json", plan)
            result = execute(plan, _ref(root / "plan.json"), root, session)
            self.assertTrue(result["matched_training"])
            self.assertEqual(sum(kind == "training" for _, _, kind in calls), 4)
            self.assertEqual(sum(kind == "matches" for _, _, kind in calls), 2)
            for name, config, kind in calls:
                if kind == "learned_evaluation":
                    cohort = "baseline" if "baseline" in name else "combined"
                    self.assertEqual(config["inputs"]["training_normalizer"],
                                     plan["parameters"]["datasets"][cohort]["inputs"]["normalizer"])
                    self.assertEqual(len(config["parameters"]["common_sets"]), 4)
                if kind == "human_evaluation":
                    self.assertTrue(all(s["id"].endswith("komi13") for s in config["parameters"]["common_sets"]))
                    self.assertEqual(config["parameters"]["reference_budget"]["queries"], 6)
                    self.assertEqual(plan["parameters"]["reference_budget"]["queries"], 24000)
                if kind == "matches":
                    cohort = name.removeprefix("matches-")
                    self.assertEqual(config["inputs"]["normalizer"],
                                     plan["parameters"]["datasets"][cohort]["inputs"]["normalizer"])


def file_hash_local(path):
    from kenggo.checkpoints import file_hash
    return file_hash(path)


if __name__ == "__main__":
    unittest.main()
