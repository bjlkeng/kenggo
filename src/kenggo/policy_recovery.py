"""One reviewed recovery of the policy study's pre-training provenance failure."""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import re
import subprocess
import time

from kenggo.checkpoints import file_hash, read_json, save_json
from kenggo.pilot_jobs import _output_bytes
from kenggo import policy_study
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import validate_config


ROOT = Path(__file__).resolve().parents[2]
PROC = Path("/proc")
CGROUP = Path("/sys/fs/cgroup")
VERSION = "e2_policy_recovery_v1"
ORIGINAL_OUTPUT = "runs/e2-policy-tuning-2026-11-26"
RECOVERY_OUTPUT = "runs/e2-policy-recovery-2026-11-26"
ORIGINAL_UNIT = "kenggo-e2-policy-tuning-20261126.service"
ORIGINAL_LAUNCH = "2026-11-26T01:35:51.168611+00:00"
ORIGINAL_DEADLINE = "2026-11-29T01:35:51.168611+00:00"
FAILURE = "ValueError: Run provenance requires data, split and rules pins"
SUPERVISOR_ID = "recovery1-supervisor"
RUN_PREFIX = "recovery1-"


def utc_now():
    return datetime.now(timezone.utc)


def _path(value):
    return (ROOT / value).resolve()


def _read(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Recovery evidence requires a path and independent SHA-256")
    return read_json(_path(ref["path"]), ref["sha256"])


def _fixed(value, expected):
    if not isinstance(value, str) or _path(value) != _path(expected):
        raise ValueError("Recovery must retain its fixed original and recovery paths")


def stopped_unit(unit):
    """Require the original service and any remaining study workers to be dead."""
    result = subprocess.run(["systemctl", "--user", "show", unit,
        "--property=Id", "--property=LoadState", "--property=ActiveState",
        "--property=MainPID", "--property=ControlGroup", "--property=ExecMainPID",
        "--property=ExecMainStartTimestampMonotonic", "--property=InvocationID"], check=True,
        capture_output=True, text=True, timeout=10)
    state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if (state.get("Id") != ORIGINAL_UNIT or state.get("LoadState") != "loaded"
            or state.get("ActiveState") not in {"inactive", "failed"}
            or state.get("MainPID") != "0" or "ControlGroup" not in state):
        raise ValueError("The original policy service is not confirmed stopped")
    group = state["ControlGroup"]
    if group:
        if not group.startswith("/") or ".." in Path(group).parts:
            raise ValueError("Invalid original service cgroup")
        directory = CGROUP / group.lstrip("/")
        if directory.exists() and any(item.read_text().strip() for item in directory.rglob("cgroup.procs")):
            raise ValueError("The original policy service still has cgroup processes")
    modules = {"kenggo.policy_study", "kenggo.policy_jobs", "kenggo.continuation_training"}
    for entry in PROC.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            memberships = (entry / "cgroup").read_text()
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        except PermissionError:
            if entry.stat().st_uid == os.getuid():
                raise ValueError("Cannot verify an owned process is outside the original study")
            continue
        if any(ORIGINAL_UNIT in Path(line.split(":", 2)[-1]).parts for line in memberships.splitlines()):
            raise ValueError("The original policy cgroup still has a process")
        arguments = [part.decode("utf-8", errors="replace") for part in command if part]
        if any(arguments[index:index + 2] == ["-m", module]
               for module in modules for index in range(len(arguments) - 1)) and any(
                str(_path(ORIGINAL_OUTPUT)) in argument or ORIGINAL_OUTPUT in argument
                for argument in arguments):
            raise ValueError("An original policy worker is still live")
    return state


def original_monotonic_start(anchor, unit):
    fields = {"schema", "unit", "boot_id", "exec_main_pid", "started_monotonic_us", "invocation_id"}
    if (not isinstance(anchor, dict) or set(anchor) != fields
            or anchor["schema"] != "policy_service_anchor_v1" or anchor["unit"] != ORIGINAL_UNIT
            or not isinstance(anchor["boot_id"], str)
            or re.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", anchor["boot_id"]) is None
            or (PROC / "sys/kernel/random/boot_id").read_text().strip() != anchor["boot_id"]
            or type(anchor["exec_main_pid"]) is not int or anchor["exec_main_pid"] <= 0
            or type(anchor["started_monotonic_us"]) is not int or anchor["started_monotonic_us"] <= 0
            or not isinstance(anchor["invocation_id"], str)
            or re.fullmatch(r"[0-9a-f]{32}", anchor["invocation_id"]) is None
            or unit.get("ExecMainPID") != str(anchor["exec_main_pid"])
            or unit.get("ExecMainStartTimestampMonotonic") != str(anchor["started_monotonic_us"])
            or unit.get("InvocationID") != anchor["invocation_id"]):
        raise ValueError("Original service clock anchor or boot identity changed")
    started = anchor["started_monotonic_us"] / 1_000_000
    if started > time.monotonic():
        raise ValueError("Original service monotonic start is in the future")
    return started


def _validate_state(plan, session, summary, progress, snapshot, request):
    common = ("schema", "plan", "launch_utc", "deadline_utc", "limits", "finalization_reserve", "prior_costs")
    if (summary != progress or any(summary.get(key) != session.get(key) for key in common)
            or session.get("schema") != policy_study.VERSION or session.get("plan") != request["plan"]
            or session.get("limits") != policy_study.LIMITS
            or session.get("finalization_reserve") != policy_study.PolicyStudy.final_reserve
            or session.get("prior_costs") != plan["inputs"]["prior_resources"]
            or session.get("launch_utc") != ORIGINAL_LAUNCH
            or session.get("deadline_utc") != ORIGINAL_DEADLINE):
        raise ValueError("Original policy session, summary and progress no longer agree")
    launch, deadline = (datetime.fromisoformat(session[key]) for key in ("launch_utc", "deadline_utc"))
    if deadline - launch != timedelta(seconds=259200):
        raise ValueError("Recovery cannot reset the original three-day allowance")
    trials = [{"id": trial["id"], "parameters": {key: value for key, value in trial.items() if key != "id"},
               "training": [], "status": "pending"} for trial in policy_study.TRIALS]
    failures = [{"name": name, "reason": FAILURE, "directory": str(_path(ORIGINAL_OUTPUT) / name),
                 "artifacts": {}} for name in ("preparation", "publication")]
    parent = {**plan["inputs"]["parent_checkpoint"],
              "config": plan["inputs"]["parent_config"], "provenance": plan["inputs"]["parent_provenance"]}
    if (summary.get("status") != "publication_failed" or summary.get("reason") != FAILURE
            or summary.get("publication_failure") != FAILURE or summary.get("active_stage") is not None
            or summary.get("failed_attempts") != failures or summary.get("trials") != trials
            or summary.get("parent_checkpoint") != parent or summary.get("assessments") != {}
            or summary.get("promotion") is not False
            or set(summary.get("stages", {})) != {"readiness"}
            or any(key in summary for key in ("preparation", "selected_trial_id", "selection_record",
                                             "comparison", "review", "recovery"))):
        raise ValueError("Recovery is limited to the known provenance failure before any training")
    readiness = summary["stages"]["readiness"]
    if (readiness.get("status") != "completed" or readiness.get("job", {}).get("status") != "completed"
            or readiness["job"].get("exit_code") != 0
            or readiness.get("directory") != str(_path(ORIGINAL_OUTPUT) / "readiness")):
        raise ValueError("The original readiness stage did not complete")
    for key, suffix in (("config", "config.json"), ("provenance", "provenance.json"),
                        ("summary", "worker/summary.json")):
        _fixed(readiness[key]["path"], ORIGINAL_OUTPUT + "/readiness/" + suffix)
    config, provenance, report = (_read(readiness[key]) for key in ("config", "provenance", "summary"))
    if (config.get("purpose") != "policy-readiness"
            or config.get("inputs") != plan["parameters"]["base_inputs"]
            or provenance.get("inputs") != config["inputs"]
            or report.get("status") != "completed"
            or report.get("check") != "cuda_tensor_autograd_synchronized"):
        raise ValueError("The retained readiness evidence differs from the frozen study")
    entries = snapshot["state"]["entries"]
    expected = {"readiness": "completed", "readiness-tracking": "completed",
                "preparation-tracking": "failed", "publication-tracking": "failed", "supervisor": "partial"}
    if (snapshot["state"]["reservations"] or snapshot["budget_breached"]
            or any(entry["run_id"].startswith(RUN_PREFIX) for entry in entries)
            or {entry["run_id"]: entry["outcome"] for entry in entries} != expected
            or len(entries) != len(expected)
            or any(entry["reason"] != (None if entry["outcome"] == "completed" else FAILURE) for entry in entries)
            or summary.get("costs", {}).get("ledger") != snapshot):
        raise ValueError("Original ledger is active, changed, breached or already recovered")
    elapsed, cpu = (summary["costs"].get(key) for key in ("elapsed_seconds", "cpu_seconds"))
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
            or type(cpu) not in (int, float) or not math.isfinite(cpu)
            or not math.isclose(cpu, sum(entry["cpu_seconds"] for entry in entries), rel_tol=1e-12)
            or cpu >= policy_study.LIMITS["cpu_seconds"]):
        raise ValueError("Original elapsed time or CPU charges are invalid")
    return launch, deadline, elapsed


def admit(request_path, request_pin):
    """Check retained evidence and process state without creating or reserving anything."""
    request_path = Path(request_path).resolve()
    request = read_json(request_path, request_pin)
    fields = {"schema", "plan", "session", "summary", "progress", "ledger_before", "service_before", "ledger_path",
              "original_output", "recovery_output", "original_unit"}
    if (not isinstance(request, dict) or set(request) != fields or request["schema"] != VERSION
            or request["original_unit"] != ORIGINAL_UNIT):
        raise ValueError("Unknown one-use policy recovery request")
    for key, expected in (("original_output", ORIGINAL_OUTPUT), ("recovery_output", RECOVERY_OUTPUT),
                          ("ledger_path", ORIGINAL_OUTPUT + "/resources.json")):
        _fixed(request[key], expected)
    original, output = _path(ORIGINAL_OUTPUT), _path(RECOVERY_OUTPUT)
    if (os.path.lexists(ROOT / RECOVERY_OUTPUT) or (ROOT / ORIGINAL_OUTPUT).is_symlink()
            or not original.is_dir()):
        raise ValueError("Original output is invalid or this recovery was already attempted")
    for key, expected in (("plan", "configs/e2_policy_tuning_v1.json"),
                          *((name, ORIGINAL_OUTPUT + "/" + name + ".json")
                            for name in ("session", "summary", "progress"))):
        _fixed(request[key]["path"], expected)
    before_path = _path(request["ledger_before"]["path"])
    service_path = _path(request["service_before"]["path"])
    if any(before_path.is_relative_to(directory) or service_path.is_relative_to(directory)
           or request_path.is_relative_to(directory)
           for directory in (original, output)):
        raise ValueError("Recovery request and ledger snapshot must remain outside both output directories")
    plan = _read(request["plan"])
    validate_config(plan)
    policy_study.validate_plan(plan)
    session, summary, progress, before = (_read(request[key]) for key in
                                         ("session", "summary", "progress", "ledger_before"))
    ledger_path = _path(request["ledger_path"])
    if file_hash(ledger_path) != request["ledger_before"]["sha256"]:
        raise ValueError("The live original ledger changed since its retained snapshot")
    ledger = BudgetLedger.open(ledger_path, budget_id=plan["budget_id"],
                               gpu_seconds=policy_study.LIMITS["gpu_seconds"])
    snapshot = ledger.snapshot()
    if snapshot["state"] != before:
        raise ValueError("The live original ledger differs from its retained snapshot")
    launch, deadline, saved_elapsed = _validate_state(plan, session, summary, progress, snapshot, request)
    unit = stopped_unit(request["original_unit"])
    monotonic_start = original_monotonic_start(_read(request["service_before"]), unit)
    now = utc_now()
    elapsed = max(saved_elapsed, (now - launch).total_seconds(), time.monotonic() - monotonic_start)
    if now >= deadline or elapsed >= policy_study.LIMITS["wall_seconds"]:
        raise ValueError("The original policy allowance has expired")
    if _output_bytes(original) >= policy_study.LIMITS["output_bytes"]:
        raise ValueError("The original policy output allowance is exhausted")
    return {"request": request, "request_ref": {"path": str(request_path), "sha256": request_pin},
            "plan": plan, "session": session, "summary": summary, "ledger": ledger,
            "original": original, "output": output, "launch": launch, "deadline": deadline,
            "saved_elapsed": saved_elapsed, "elapsed": elapsed, "unit": unit,
            "monotonic_start": monotonic_start}


class Recovery(policy_study.PolicyStudy):
    supervisor_id = SUPERVISOR_ID
    run_prefix = RUN_PREFIX

    def __init__(self, request_path, request_pin):
        self.cpu_started = time.process_time()
        checked = admit(request_path, request_pin)
        self.plan, self.output, self.original_output = checked["plan"], checked["output"], checked["original"]
        self.launch, self.deadline, self.ledger = checked["launch"], checked["deadline"], checked["ledger"]
        self.limits = deepcopy(checked["session"]["limits"])
        self.production_end = self.deadline - timedelta(seconds=self.limits["publication_seconds"])
        self.saved_elapsed = checked["saved_elapsed"]
        self.started = min(checked["monotonic_start"], time.monotonic() - max(
            checked["elapsed"], (utc_now() - self.launch).total_seconds()))
        self.accounted_cpu, self.closed = 0.0, False
        self.state = deepcopy(checked["summary"])
        self.state.update(status="running", reason=None, active_stage=None)
        self.state.pop("publication_failure")
        evidence = {"schema": VERSION, "request": checked["request_ref"],
            "original_evidence": deepcopy(checked["request"]), "original_unit_at_admission": checked["unit"],
            "recovery_started_utc": utc_now().isoformat(), "original_elapsed_lower_bound": self.saved_elapsed,
            "original_monotonic_start": checked["monotonic_start"],
            "supervisor_id": self.supervisor_id, "run_prefix": self.run_prefix,
            "accounting": "Original deadline, all prior charges and both output directories remain included."}
        self.state["recovery"] = evidence
        self.output.mkdir(parents=True, exist_ok=False)
        if file_hash(self.ledger.path) != checked["request"]["ledger_before"]["sha256"]:
            raise ValueError("The original ledger changed during recovery admission")
        self.ledger.reserve(self.supervisor_id, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                            metadata={"purpose": "policy_recovery_supervisor", "cpu_reserve": 60,
                                      "request_sha256": request_pin})
        save_json(self.output / "session.json", {**checked["session"], "recovery": evidence})
        self.persist()

    def costs(self):
        result = super().costs()
        result["elapsed_seconds"] = max(result["elapsed_seconds"], self.saved_elapsed,
                                        (utc_now() - self.launch).total_seconds())
        result["output_bytes"] += _output_bytes(self.original_output)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    result = policy_study.run_study(Recovery(args.request, args.sha256))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
