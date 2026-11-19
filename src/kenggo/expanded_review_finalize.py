"""Bound saved-evidence verification and review export by the original E2 deadline."""

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import math
import os
from pathlib import Path
import signal
import sys
import time

from kenggo.base_pilot import _ref
from kenggo.checkpoints import file_hash, read_json, save_json
from kenggo.expanded_review import ROOT, VERSION, validate_review
from kenggo.pilot_jobs import JobLimits, _output_bytes, run_job
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import capture_provenance


OUTPUT = "runs/e2-review-rerun-review-2026-11-19"
HELPER_LIMITS = JobLimits(wall_seconds=2400, cpu_seconds=1200,
                         output_bytes=384 * 1024**2, shutdown_seconds=10)
PUBLICATION_CPU = 600
PUBLICATION_BYTES = 512 * 1024**2


def _load(reference):
    return read_json(ROOT / reference["path"], reference["sha256"])


def _checked(producer, pin, ledger):
    result = read_json(producer, pin)
    plan = _load(result["plan"])
    validate_review(plan)
    p = plan["parameters"]
    run = (ROOT / p["output"]).resolve()
    if (producer.parent != run or producer.name not in {"summary.json", "failure.json"}
            or result.get("schema") != VERSION or result.get("status") not in {"completed", "incomplete"}
            or (producer.name == "summary.json") != (result["status"] == "completed")
            or any((ROOT / result[name]["path"]).resolve() != run / (name + ".json")
                   for name in ("session", "source"))):
        raise ValueError("Finalization requires the pinned, settled one-week producer")
    session, source = _load(result["session"]), _load(result["source"])
    launched = datetime.fromisoformat(session["launch_utc"])
    deadline = datetime.fromisoformat(session["deadline_utc"])
    now = datetime.now(timezone.utc)
    if (session.get("schema") != VERSION or session.get("plan") != result["plan"]
            or session["limits"] != p["limits"] or launched.tzinfo is None or deadline.tzinfo is None
            or launched > now or deadline - launched != timedelta(seconds=p["limits"]["wall_seconds"])
            or (deadline - now).total_seconds() < HELPER_LIMITS.wall_seconds + 120):
        raise ValueError("The original deadline cannot fund complete bounded verification")
    current = ledger.snapshot()
    before = session["ledger_before"]["state"]["entries"]
    entries = current["state"]["entries"]
    if (current["state"] != result["costs"]["ledger"]["state"] or current["state"]["reservations"]
            or current["budget_breached"] or entries[:len(before)] != before
            or any(entry["gpu_seconds"] != 0 for entry in entries[len(before):])):
        raise ValueError("Producer ledger changed, is unsettled, or allocated new GPU time")
    cpu = math.fsum(entry["cpu_seconds"] for entry in entries[len(before):])
    output_bytes = _output_bytes(run)
    if (not math.isclose(cpu, result["costs"]["cpu_seconds"], abs_tol=1e-6)
            or cpu + HELPER_LIMITS.cpu_seconds + PUBLICATION_CPU > p["limits"]["cpu_seconds"]
            or output_bytes + PUBLICATION_BYTES > p["limits"]["output_bytes"]):
        raise ValueError("Remaining CPU or output allowance cannot fund verification")
    current_source, _ = capture_provenance(ROOT, plan["inputs"])
    if current_source["source"] != source["source"]:
        raise ValueError("Finalization source changed from the producer")
    return result, plan, session, deadline, cpu, output_bytes


def _run(producer, producer_pin, helper, helper_pin, started, cpu_started):
    producer, helper = Path(producer).resolve(), Path(helper).resolve()
    if helper.stat().st_size > 1024**2 or file_hash(helper) != helper_pin:
        raise ValueError("Verification helper changed or exceeds one MiB")
    result = read_json(producer, producer_pin)
    plan = _load(result["plan"])
    output = ROOT / OUTPUT
    lock_path = ROOT / ".cache/e2-expanded-pilot.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() or output.is_symlink():
            raise ValueError("Final verification is one-use; retain any previous attempt")
        ledger = BudgetLedger.open(ROOT / plan["parameters"]["ledger"],
                                   budget_id=plan["budget_id"], gpu_seconds=86400)
        result, plan, session, deadline, production_cpu, production_bytes = _checked(producer, producer_pin, ledger)
        output.mkdir(parents=True)
        helper_copy = output / "verify.py"
        helper_copy.write_bytes(helper.read_bytes())
        if file_hash(helper_copy) != helper_pin:
            raise ValueError("Verification helper changed while copying")
        worker = output / "worker"
        run_id = producer.parent.name + "-verification"
        publication_id = run_id + "-publication"
        ledger.reserve(publication_id, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
            metadata={"purpose": "bounded_verification_setup_and_publication", "cpu_reserve_seconds": PUBLICATION_CPU,
                      "producer": _ref(producer), "helper": _ref(helper_copy)})
        status, reason, job = "failed", "verification_did_not_finish", None
        error = None
        try:
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", UV_OFFLINE="1", WANDB_MODE="offline")
            command = [sys.executable, str(helper_copy), str(producer), producer_pin, str(worker)]
            job = run_job(command, worker, cwd=ROOT, limits=HELPER_LIMITS, ledger=ledger,
                run_id=run_id, arm_shares={"shared": 1}, environment=environment,
                metadata={"purpose": "saved_evidence_only", "helper_sha256": helper_pin})
            if job["status"] != "completed":
                raise RuntimeError("Saved-evidence helper failed: " + str(job["reason"]))
            for name in ("review.html", "review.manifest.json", "verification.json", "browser-checks.json"):
                if not (worker / name).is_file():
                    raise ValueError("Verification did not produce " + name)
            for name in ("verification.json", "browser-checks.json"):
                path = worker / name
                if read_json(path, file_hash(path)).get("status") != "passed":
                    raise ValueError("Verification has not passed: " + name)
            status, reason = "completed", None
        except BaseException as caught:
            error, reason = caught, type(caught).__name__ + ": " + str(caught)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.setitimer(signal.ITIMER_PROF, 0)
            now = datetime.now(timezone.utc)
            measured_parent = time.process_time() - cpu_started
            output_charge = _output_bytes(output) + 1024**2
            if (measured_parent > PUBLICATION_CPU or output_charge > PUBLICATION_BYTES or now >= deadline):
                status, reason = "failed", "final_verification_resource_limit"
            ledger.record(publication_id, allocations=[], outcome=status, arm_shares={"shared": 1},
                cpu_seconds=max(PUBLICATION_CPU, measured_parent),
                wall_seconds=max(0, time.monotonic() - started - (job or {}).get("costs", {}).get("wall_seconds", 0)),
                reason=reason)
            state = ledger.snapshot()
            old_ids = {entry["run_id"] for entry in session["ledger_before"]["state"]["entries"]}
            cpu = math.fsum(entry["cpu_seconds"] for entry in state["state"]["entries"] if entry["run_id"] not in old_ids)
            if cpu > plan["parameters"]["limits"]["cpu_seconds"] or state["state"]["reservations"]:
                status, reason = "failed", "total_resource_limit_or_unsettled_ledger"
            record = {"schema": "expanded_review_final_resources_v1", "status": status, "reason": reason,
                "producer": _ref(producer), "helper": _ref(helper_copy), "session": result["session"],
                "launch_utc": session["launch_utc"], "deadline_utc": session["deadline_utc"],
                "finished_utc": now.isoformat(), "elapsed_seconds": (now - datetime.fromisoformat(session["launch_utc"])).total_seconds(),
                "cpu_seconds": cpu, "production_cpu_seconds": production_cpu,
                "verification_cpu_seconds": cpu - production_cpu, "publication_cpu_charge": PUBLICATION_CPU,
                "measured_parent_cpu_seconds": measured_parent, "output_bytes_charged": production_bytes + output_charge,
                "verification_output_bytes_charged": output_charge, "closing_output_reserve_bytes": 1024**2,
                "gpu_seconds": 0, "ledger": state, "job": _ref(worker / "job.json") if (worker / "job.json").is_file() else None,
                "artifacts": {path.name: _ref(path) for path in worker.glob("*") if path.is_file()},
                "scope": "Production plus bounded saved-evidence verification and page publication. The 600 CPU seconds are a conservative parent/setup/closing charge; job monitoring is also included in the job charge. Later source documentation and independent code review are development work."}
            worker.mkdir(exist_ok=True)
            save_json(worker / "final-resources.json", record)
        if error is not None:
            raise error
        if status != "completed":
            raise RuntimeError(reason)
        return record


def run(producer, producer_pin, helper, helper_pin):
    started, cpu_started = time.monotonic(), time.process_time()
    # Read only the pinned small envelope before arming the preflight deadline.
    result = read_json(producer, producer_pin)
    session = _load(result["session"])
    deadline = datetime.fromisoformat(session["deadline_utc"])
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining < HELPER_LIMITS.wall_seconds + 120:
        raise ValueError("The original deadline cannot fund complete bounded verification")
    old_alarm, old_prof = signal.getsignal(signal.SIGALRM), signal.getsignal(signal.SIGPROF)
    def expired(signum, frame):
        raise TimeoutError("Final verification reached its elapsed or parent CPU cutoff")
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGPROF, expired)
    signal.setitimer(signal.ITIMER_REAL, min(3480, remaining - 120))
    signal.setitimer(signal.ITIMER_PROF, max(.001, 550 - (time.process_time() - cpu_started)))
    try:
        return _run(producer, producer_pin, helper, helper_pin, started, cpu_started)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGPROF, old_prof)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("producer", type=Path)
    parser.add_argument("helper", type=Path)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--helper-sha256", required=True)
    args = parser.parse_args()
    run(args.producer, args.producer_sha256, args.helper, args.helper_sha256)
