"""Recovery admission and cumulative accounting using temporary synthetic evidence."""

from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kenggo import policy_recovery as recovery
from kenggo import policy_study
from kenggo.checkpoints import save_json
from kenggo.continuation_data import reference
from kenggo.pilot_jobs import _output_bytes
from kenggo.resource_ledger import BudgetLedger


class StoppedUnitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("proc", "cgroup"):
            (self.root / name).mkdir()
        for name, value in (("PROC", self.root / "proc"), ("CGROUP", self.root / "cgroup")):
            patcher = patch.object(recovery, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.state = {"Id": recovery.ORIGINAL_UNIT, "LoadState": "loaded", "ActiveState": "failed",
                      "MainPID": "0", "ControlGroup": ""}
        patcher = patch.object(recovery.subprocess, "run", side_effect=lambda *args, **kwargs:
            SimpleNamespace(stdout="\n".join(key + "=" + value for key, value in self.state.items())))
        self.query = patcher.start()
        self.addCleanup(patcher.stop)

    def test_only_stopped_loaded_unit_with_no_main_process_is_accepted(self):
        self.assertEqual(recovery.stopped_unit(recovery.ORIGINAL_UNIT), self.state)
        original = deepcopy(self.state)
        for change in ({"ActiveState": "active"}, {"MainPID": "999"}, {"LoadState": "not-found"},
                       {"Id": "different.service"}):
            self.state = original | change
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "not confirmed stopped"):
                recovery.stopped_unit(recovery.ORIGINAL_UNIT)
        command = self.query.call_args.args[0]
        self.assertEqual(command[:4], ["systemctl", "--user", "show", recovery.ORIGINAL_UNIT])

    def test_live_cgroup_or_orphaned_worker_blocks_admission(self):
        group = self.root / "cgroup/service/descendant"
        group.mkdir(parents=True)
        (group / "cgroup.procs").write_text("999\n")
        self.state["ControlGroup"] = "/service"
        with self.assertRaisesRegex(ValueError, "cgroup processes"):
            recovery.stopped_unit(recovery.ORIGINAL_UNIT)
        self.state["ControlGroup"] = ""
        process = self.root / "proc/999"
        process.mkdir()
        (process / "cgroup").write_text("0::/unrelated\n")
        (process / "cmdline").write_bytes(b"python\0-m\0kenggo.continuation_training\0"
            + (recovery.ORIGINAL_OUTPUT + "/trial/config.json").encode() + b"\0")
        with self.assertRaisesRegex(ValueError, "worker is still live"):
            recovery.stopped_unit(recovery.ORIGINAL_UNIT)
        (process / "cmdline").write_bytes(b"unrelated-command\0")
        (process / "cgroup").write_text("0::/user.slice/" + recovery.ORIGINAL_UNIT + "/child\n")
        with self.assertRaisesRegex(ValueError, "cgroup still has a process"):
            recovery.stopped_unit(recovery.ORIGINAL_UNIT)


class PolicyRecoveryTests(unittest.TestCase):
    def test_clock_rollback_before_admission_cannot_remove_downtime(self):
        first = recovery.admit(self.request_path, reference(self.request_path)["sha256"])
        self.now -= timedelta(seconds=600)
        second = recovery.admit(self.request_path, reference(self.request_path)["sha256"])
        self.assertGreaterEqual(second["elapsed"], first["elapsed"])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(recovery, "ROOT", self.root).start()
        self.original = self.root / recovery.ORIGINAL_OUTPUT
        self.output = self.root / recovery.RECOVERY_OUTPUT
        self.evidence = self.root / "evidence"
        self.original.mkdir(parents=True)
        self.evidence.mkdir()
        self.launch = datetime.fromisoformat(recovery.ORIGINAL_LAUNCH)
        self.now = self.launch + timedelta(hours=1)
        self.monotonic, self.cpu = 10000.0, 100.0
        proc = self.root / "proc"
        boot_file = proc / "sys/kernel/random/boot_id"
        boot_file.parent.mkdir(parents=True)
        self.boot_id = "88542a20-d89e-4dc2-b73e-1f6fd7765bc2"
        boot_file.write_text(self.boot_id)
        patch.object(recovery, "PROC", proc).start()
        self.anchor = {"schema": "policy_service_anchor_v1", "unit": recovery.ORIGINAL_UNIT,
            "boot_id": self.boot_id, "exec_main_pid": 1234, "started_monotonic_us": 6400000000,
            "invocation_id": "a" * 32}
        patch.object(recovery, "utc_now", side_effect=lambda: self.now).start()
        patch.object(recovery.time, "monotonic", side_effect=lambda: self.monotonic).start()
        patch.object(recovery.time, "process_time", side_effect=lambda: self.cpu).start()
        self.unit = patch.object(recovery, "stopped_unit", return_value={
            "Id": recovery.ORIGINAL_UNIT, "LoadState": "loaded", "ActiveState": "failed",
            "MainPID": "0", "ControlGroup": "", "ExecMainPID": "1234",
            "ExecMainStartTimestampMonotonic": "6400000000", "InvocationID": "a" * 32}).start()
        model = {"blocks": 8, "channels": 128, "rank_conditioned": True}
        data_path = self.evidence / "data.json"
        save_json(data_path, {"parameters": {"model": model}, "scope": "synthetic metadata only"})
        data = reference(data_path)
        inputs = {name: data for name in (*policy_study.PREPARATION_KEYS, "parent_checkpoint",
            "parent_config", "parent_provenance", "protocol", "prior_resources", "review_history")}
        base = {name: data for name in ("data", "split", "rules")}
        self.plan = {"version": "experiment_config_v1", "purpose": "policy-study", "arm": "shared",
            "entity": "fixture", "project": "fixture", "base_url": "https://api.wandb.ai",
            "comparison": "recovery-fixture", "seed": 20261126, "budget_id": "synthetic-policy-recovery",
            "inputs": inputs, "parameters": {"recipe": policy_study.RECIPE,
                "stage_id": policy_study.STAGE_ID, "round_updates": policy_study.ROUND_UPDATES,
                "trials": list(policy_study.TRIALS), "limits": deepcopy(policy_study.LIMITS),
                "training_rows": 3999997, "output": recovery.ORIGINAL_OUTPUT,
                "base_inputs": base, "model": model}}
        plan_path = self.root / "configs/e2_policy_tuning_v1.json"
        plan_path.parent.mkdir()
        save_json(plan_path, self.plan)
        self.session = {"schema": policy_study.VERSION, "plan": reference(plan_path),
            "launch_utc": recovery.ORIGINAL_LAUNCH, "deadline_utc": recovery.ORIGINAL_DEADLINE,
            "limits": deepcopy(policy_study.LIMITS), "finalization_reserve": policy_study.PolicyStudy.final_reserve,
            "prior_costs": inputs["prior_resources"]}
        ready = self.original / "readiness"
        (ready / "worker").mkdir(parents=True)
        save_json(ready / "config.json", {"purpose": "policy-readiness", "inputs": base})
        save_json(ready / "provenance.json", {"inputs": base})
        save_json(ready / "worker/summary.json", {"status": "completed", "check": "cuda_tensor_autograd_synchronized"})
        readiness = {"status": "completed", "directory": str(ready),
            "job": {"status": "completed", "exit_code": 0}, "config": reference(ready / "config.json"),
            "provenance": reference(ready / "provenance.json"), "summary": reference(ready / "worker/summary.json")}
        self.ledger = BudgetLedger.create(self.original / "resources.json", budget_id=self.plan["budget_id"],
                                          gpu_seconds=policy_study.LIMITS["gpu_seconds"])
        for index, (name, outcome) in enumerate((("readiness", "completed"), ("readiness-tracking", "completed"),
                ("preparation-tracking", "failed"), ("publication-tracking", "failed"), ("supervisor", "partial"))):
            gpu = name == "readiness"
            self.ledger.reserve(name, estimate_gpu_seconds=10 if gpu else 0, shutdown_gpu_seconds=0,
                                metadata={"synthetic": True})
            self.ledger.record(name, allocations=[{"device": "0", "start_utc": self.launch.isoformat(),
                "stop_utc": (self.launch + timedelta(seconds=2)).isoformat()}] if gpu else [],
                outcome=outcome, arm_shares={"shared": 1}, cpu_seconds=index + 1, wall_seconds=2 if gpu else 0,
                reason=None if outcome == "completed" else recovery.FAILURE)
        self.summary = {**deepcopy(self.session), "status": "publication_failed", "reason": recovery.FAILURE,
            "publication_failure": recovery.FAILURE, "active_stage": None, "promotion": False,
            "stages": {"readiness": readiness}, "assessments": {},
            "parent_checkpoint": {**inputs["parent_checkpoint"], "config": inputs["parent_config"],
                                  "provenance": inputs["parent_provenance"]},
            "failed_attempts": [{"name": name, "directory": str(self.original / name),
                "reason": recovery.FAILURE, "artifacts": {}} for name in ("preparation", "publication")],
            "trials": [{"id": item["id"], "parameters": {key: value for key, value in item.items() if key != "id"},
                        "training": [], "status": "pending"} for item in policy_study.TRIALS],
            "costs": {"elapsed_seconds": 7.0, "cpu_seconds": 15.0,
                      "output_bytes": 1000, "ledger": self.ledger.snapshot()}}
        self.request_path = self.evidence / "request.json"
        self.refresh()

    def refresh(self):
        for name, value in (("session", self.session), ("summary", self.summary), ("progress", self.summary)):
            save_json(self.original / (name + ".json"), value, replace=True)
        snapshot = self.evidence / "ledger-before.json"
        snapshot.write_bytes(self.ledger.path.read_bytes())
        anchor = self.evidence / "service-before.json"
        save_json(anchor, self.anchor, replace=True)
        self.request = {"schema": recovery.VERSION, "plan": self.session["plan"],
            **{name: reference(self.original / (name + ".json")) for name in ("session", "summary", "progress")},
            "ledger_before": reference(snapshot), "service_before": reference(anchor),
            "ledger_path": recovery.ORIGINAL_OUTPUT + "/resources.json",
            "original_output": recovery.ORIGINAL_OUTPUT, "recovery_output": recovery.RECOVERY_OUTPUT,
            "original_unit": recovery.ORIGINAL_UNIT}
        self.request_pin = save_json(self.request_path, self.request, replace=True)

    def original_bytes(self):
        return {path: path.read_bytes() for path in self.original.rglob("*") if path.is_file()}

    def assert_rejected_without_mutation(self, pattern):
        before = self.original_bytes()
        with patch.object(BudgetLedger, "reserve", side_effect=AssertionError("Unexpected reservation")):
            with self.assertRaisesRegex((ValueError, FileExistsError), pattern):
                recovery.Recovery(self.request_path, self.request_pin)
        self.assertEqual(self.original_bytes(), before)
        self.assertFalse(self.output.exists())

    def test_read_only_admission_keeps_original_state_and_all_charges(self):
        before = self.original_bytes()
        with patch.object(BudgetLedger, "create", side_effect=AssertionError("Must open the original ledger")):
            checked = recovery.admit(self.request_path, self.request_pin)
        self.assertEqual(checked["elapsed"], 3600)
        self.assertEqual(checked["launch"], self.launch)
        self.assertEqual(checked["deadline"], datetime.fromisoformat(recovery.ORIGINAL_DEADLINE))
        self.assertEqual(checked["ledger"].snapshot()["charged_gpu_seconds"], 2)
        self.assertEqual(self.original_bytes(), before)
        self.assertFalse(self.output.exists())
        self.unit.assert_called_once_with(recovery.ORIGINAL_UNIT)

    def test_recovery_uses_original_clock_ledger_reserves_and_preserves_failed_evidence(self):
        before = self.original_bytes()
        with patch.object(BudgetLedger, "create", side_effect=AssertionError("Cannot reset the allowance")):
            study = recovery.Recovery(self.request_path, self.request_pin)
        self.assertEqual(study.launch, self.launch)
        self.assertEqual(study.deadline - study.launch, timedelta(seconds=259200))
        self.assertEqual(study.deadline - study.production_end, timedelta(seconds=21600))
        self.assertEqual(study.limits, policy_study.LIMITS)
        self.assertEqual(study.final_reserve, self.session["finalization_reserve"])
        self.assertEqual(study.ledger.path, self.original / "resources.json")
        self.assertEqual(study.state["failed_attempts"], self.summary["failed_attempts"])
        self.assertEqual(study.state["trials"], self.summary["trials"])
        self.assertEqual(study.state["stages"], self.summary["stages"])
        self.assertEqual(study.state["status"], "running")
        self.assertNotIn("publication_failure", study.state)
        self.assertEqual(study.state["recovery"]["request"]["sha256"], self.request_pin)
        snapshot = study.ledger.snapshot()
        self.assertEqual(snapshot["state"]["entries"], self.summary["costs"]["ledger"]["state"]["entries"])
        self.assertEqual(set(snapshot["state"]["reservations"]), {"recovery1-supervisor"})
        costs = study.costs()
        self.assertEqual(costs["elapsed_seconds"], 3600)
        self.assertEqual(costs["cpu_seconds"], 75)
        self.assertEqual(costs["ledger"]["charged_gpu_seconds"], 2)
        self.assertEqual(costs["output_bytes"], _output_bytes(self.original) + _output_bytes(self.output))
        self.monotonic += 100
        self.now -= timedelta(minutes=10)
        self.assertEqual(study.costs()["elapsed_seconds"], 3700)
        self.now = self.launch + timedelta(hours=2)
        self.assertEqual(study.costs()["elapsed_seconds"], 7200)
        self.cpu += 5
        result = study.finish("completed", "Synthetic recovery completed without workers")
        self.assertEqual(result["costs"]["cpu_seconds"], 80)
        self.assertEqual(len(result["costs"]["ledger"]["state"]["entries"]), 6)
        self.assertEqual(result["costs"]["ledger"]["state"]["entries"][-1]["run_id"], "recovery1-supervisor")
        self.assertEqual(result["costs"]["ledger"]["state"]["reservations"], {})
        for filename, contents in before.items():
            if filename != self.ledger.path:
                self.assertEqual(filename.read_bytes(), contents)
        session = json.loads((self.output / "session.json").read_text())
        for key, value in self.session.items():
            self.assertEqual(session[key], value)

    def test_saved_elapsed_lower_bound_survives_a_clock_rollback(self):
        self.summary["costs"]["elapsed_seconds"] = 5000
        self.refresh()
        study = recovery.Recovery(self.request_path, self.request_pin)
        self.assertEqual(study.costs()["elapsed_seconds"], 5000)
        self.monotonic += 20
        self.assertEqual(study.costs()["elapsed_seconds"], 5020)

    def test_changed_boot_or_service_identity_blocks_admission(self):
        original = deepcopy(self.unit.return_value)
        for key, value in (("ExecMainPID", "9999"), ("ExecMainStartTimestampMonotonic", "6500000000"),
                           ("InvocationID", "b" * 32)):
            with self.subTest(key=key):
                self.unit.return_value = original | {key: value}
                self.assert_rejected_without_mutation("clock anchor or boot identity changed")
        self.unit.return_value = original
        (recovery.PROC / "sys/kernel/random/boot_id").write_text("11111111-1111-1111-1111-111111111111")
        self.assert_rejected_without_mutation("clock anchor or boot identity changed")

    def test_monotonic_expiry_blocks_admission_despite_a_utc_rollback(self):
        self.monotonic = self.anchor["started_monotonic_us"] / 1_000_000 + 259200
        self.assert_rejected_without_mutation("allowance has expired")

    def test_future_monotonic_anchor_blocks_admission(self):
        self.monotonic = self.anchor["started_monotonic_us"] / 1_000_000 - 1
        self.assert_rejected_without_mutation("monotonic start is in the future")

    def test_changed_pinned_evidence_fails_before_any_reservation(self):
        (self.original / "progress.json").write_text("{}")
        self.assert_rejected_without_mutation("expected hash")

    def test_a_changed_live_ledger_cannot_use_an_older_snapshot(self):
        self.ledger.reserve("unexpected", estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
        self.assert_rejected_without_mutation("ledger changed")

    def test_open_reservation_and_prior_recovery_ids_are_rejected_even_if_repinned(self):
        for settled in (False, True):
            with self.subTest(settled=settled):
                name = "recovery1-unexpected" if settled else "unexpected"
                self.ledger.reserve(name, estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
                if settled:
                    self.ledger.record(name, allocations=[], outcome="completed", arm_shares={"shared": 1},
                                       cpu_seconds=0, wall_seconds=0)
                self.summary["costs"]["ledger"] = self.ledger.snapshot()
                self.refresh()
                self.assert_rejected_without_mutation("ledger is active, changed, breached or already recovered")
                if not settled:
                    self.ledger.record(name, allocations=[], outcome="cancelled", arm_shares={"shared": 1},
                                       cpu_seconds=0, wall_seconds=0, reason="Synthetic cleanup")

    def test_expired_utc_or_elapsed_allowance_is_rejected(self):
        self.now = datetime.fromisoformat(recovery.ORIGINAL_DEADLINE)
        self.assert_rejected_without_mutation("allowance has expired")
        self.now = self.launch + timedelta(hours=1)
        self.summary["costs"]["elapsed_seconds"] = 259200
        self.refresh()
        self.assert_rejected_without_mutation("allowance has expired")

    def test_live_service_or_worker_blocks_recovery_without_mutation(self):
        self.unit.side_effect = ValueError("An original policy worker is still live")
        self.assert_rejected_without_mutation("still live")

    def test_recovery_directory_is_a_one_use_marker(self):
        self.output.mkdir()
        before = self.original_bytes()
        with self.assertRaisesRegex(ValueError, "already attempted"):
            recovery.Recovery(self.request_path, self.request_pin)
        self.assertEqual(self.original_bytes(), before)

    def test_other_failure_training_or_changed_allowance_cannot_be_adopted(self):
        original = deepcopy(self.summary)
        changes = (lambda state: state.update(reason="A different failure"),
            lambda state: state["trials"][0]["training"].append({"epoch": 0}),
            lambda state: state["stages"].update(preparation={"status": "completed"}),
            lambda state: state["limits"].update(wall_seconds=259201),
            lambda state: state.update(deadline_utc="2026-11-30T01:35:51.168611+00:00"))
        for change in changes:
            self.summary = deepcopy(original)
            change(self.summary)
            self.refresh()
            with self.subTest(change=change):
                self.assert_rejected_without_mutation("no longer agree|known provenance failure")

    def test_different_output_or_unit_is_rejected(self):
        for key, value in (("recovery_output", "runs/retry"), ("original_output", "runs/other"),
                           ("ledger_path", "runs/other/resources.json"), ("original_unit", "other.service")):
            request = deepcopy(self.request)
            request[key] = value
            self.request_pin = save_json(self.request_path, request, replace=True)
            with self.subTest(key=key):
                self.assert_rejected_without_mutation("fixed original|Unknown one-use")

    def test_cli_runs_only_the_admitted_recovery_instance(self):
        instance = object()
        with patch.object(recovery, "Recovery", return_value=instance) as create, \
                patch.object(policy_study, "run_study", return_value={"status": "completed"}) as run, \
                patch("sys.argv", ["policy_recovery", str(self.request_path), "--sha256", self.request_pin]):
            recovery.main()
        create.assert_called_once_with(self.request_path, self.request_pin)
        run.assert_called_once_with(instance)


if __name__ == "__main__":
    unittest.main()
