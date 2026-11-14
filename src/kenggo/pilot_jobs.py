"""Killable local jobs with allocation intervals charged to an existing ledger."""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import time

from kenggo.checkpoints import save_json


@dataclass(frozen=True)
class JobLimits:
    wall_seconds: float
    cpu_seconds: float
    output_bytes: int
    shutdown_seconds: float = 10.0

    def __post_init__(self):
        for value in (self.wall_seconds, self.cpu_seconds, self.shutdown_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Job time limits must be finite and positive")
        if type(self.output_bytes) is not int or self.output_bytes <= 0:
            raise ValueError("Job output allowance must be positive bytes")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _usage(pid):
    queue, seen, cpu, rss = [pid], set(), 0.0, 0
    ticks, page = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE")
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            fields = Path(f"/proc/{current}/stat").read_text().rsplit(")", 1)[1].split()
            cpu += sum(int(fields[index]) for index in (11, 12, 13, 14)) / ticks
            rss += int(fields[21]) * page
            for children in Path(f"/proc/{current}/task").glob("*/children"):
                queue.extend(int(child) for child in children.read_text().split())
        except (OSError, ValueError):
            continue
    return cpu, rss, seen


def _alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _kill(pid, descendants):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for child in descendants | {pid}:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _output_bytes(output):
    total = 0
    for path in output.rglob("*"):
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def run_job(command, output, *, cwd, limits, ledger, run_id, arm_shares, device=None,
            environment=None, metadata=None):
    """Run one owned job; failed results stay recorded and are never retried.

    GPU allocation conservatively spans launch through process-tree cleanup,
    including imports, loading and idle work. The caller owns the single GPU
    and must not launch another allocating process outside this boundary.
    """
    if (not isinstance(limits, JobLimits) or not isinstance(command, (list, tuple))
            or not command or any(not isinstance(part, str) or not part for part in command)
            or device is not None and (not isinstance(device, str) or not device)):
        raise ValueError("A job requires an argument vector, limits and an optional device ID")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema": "bounded_pilot_job_v1", "command": list(command), "run_id": run_id,
              "device": device, "limits": limits.__dict__, "status": "preparing"}
    save_json(output / "job.json", record)
    ledger.reserve(run_id, estimate_gpu_seconds=limits.wall_seconds if device else 0,
                   shutdown_gpu_seconds=limits.shutdown_seconds if device else 0,
                   metadata={"command": list(command), **(metadata or {})})
    started, cpu_started = time.monotonic(), time.process_time()
    allocation_start = _utc()
    process, reaped, usage = None, False, None
    observed_cpu = retired_cpu = previous_cpu = peak_rss = 0
    known_pids = set()
    reason, error = None, None
    output_bytes = 0
    try:
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True)
            record.update(status="running", pid=process.pid, allocation_start_utc=allocation_start)
            save_json(output / "job.json", record, replace=True)
            while True:
                current, rss, pids = _usage(process.pid)
                known_pids.update(pids)
                if current < previous_cpu:
                    retired_cpu += previous_cpu - current
                previous_cpu = current
                observed_cpu = max(observed_cpu, current + retired_cpu)
                peak_rss = max(peak_rss, rss)
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    reaped = True
                    if process.returncode != 0:
                        reason = "process_exit"
                    break
                output_bytes = _output_bytes(output)
                if time.monotonic() - started >= limits.wall_seconds:
                    reason = "wall_budget"
                elif time.process_time() - cpu_started + observed_cpu >= limits.cpu_seconds:
                    reason = "cpu_budget"
                elif output_bytes > limits.output_bytes:
                    reason = "output_budget"
                if reason:
                    break
                time.sleep(0.02)
    except BaseException as caught:
        error, reason = caught, type(caught).__name__
    finally:
        shutdown_started = time.monotonic()
        if process is not None:
            current, rss, pids = _usage(process.pid)
            known_pids.update(pids)
            observed_cpu = max(observed_cpu, current + retired_cpu)
            peak_rss = max(peak_rss, rss)
            live = {pid for pid in known_pids if _alive(pid)}
            if reaped and live and reason is None:
                reason = "unclosed_descendants"
            _kill(process.pid, known_pids)
            while True:
                if not reaped:
                    pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                    if pid:
                        process.returncode = os.waitstatus_to_exitcode(status)
                        reaped = True
                if reaped and not any(_alive(pid) for pid in known_pids):
                    break
                if time.monotonic() - shutdown_started >= limits.shutdown_seconds:
                    record.update(status="allocation_end_unknown", reason="process_tree_did_not_stop",
                                  ledger_reservation_retained=True)
                    save_json(output / "job.json", record, replace=True)
                    raise RuntimeError("Job cleanup failed; allocation reservation remains open")
                time.sleep(0.01)
        allocation_stop = _utc()
        reaped_cpu = 0 if usage is None else usage.ru_utime + usage.ru_stime
        shutdown_charge = ((time.monotonic() - shutdown_started + 0.02) * len(os.sched_getaffinity(0))
                           if process is not None and len(known_pids) > 1 and reason else 0)
        parent_cpu = time.process_time() - cpu_started
        child_cpu = max(observed_cpu, reaped_cpu) + shutdown_charge
        costs = {"wall_seconds": time.monotonic() - started, "cpu_seconds": parent_cpu + child_cpu}
        output_bytes = _output_bytes(output)
        if reason is None:
            if costs["cpu_seconds"] > limits.cpu_seconds:
                reason = "cpu_budget"
            elif output_bytes > limits.output_bytes:
                reason = "output_budget"
            elif costs["wall_seconds"] > limits.wall_seconds + limits.shutdown_seconds:
                reason = "wall_budget"
        allocations = ([{"device": device, "start_utc": allocation_start, "stop_utc": allocation_stop}]
                       if device else [])
        outcome = "completed" if reason is None else "failed"
        ledger.record(run_id, allocations=allocations, outcome=outcome, arm_shares=arm_shares,
                      reason=reason, **costs)
        record.update(status=outcome, reason=reason, exit_code=process.returncode if process else None,
                      allocation_start_utc=allocation_start, allocation_stop_utc=allocation_stop,
                      costs=costs, peak_sampled_tree_rss_bytes=peak_rss,
                      parent_cpu_seconds=parent_cpu, child_cpu_seconds=child_cpu,
                      observed_output_bytes=output_bytes, conservative_shutdown_cpu_seconds=shutdown_charge,
                      cpu_scope="parent and reaped child; retained sampled descendants with conservative forced shutdown charge",
                      sampling_seconds=0.02, ledger_reservation_retained=False)
        save_json(output / "job.json", record, replace=True)
    if error is not None:
        raise error
    return record
