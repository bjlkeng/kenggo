"""Durable local accounting for caller-reported GPU allocation intervals."""

from contextlib import contextmanager
from datetime import datetime, timedelta
import fcntl
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import stat
import tempfile


SCHEMA_VERSION = 1
OUTCOMES = frozenset({"completed", "failed", "partial", "cancelled"})


def _identifier(value, name):
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(ord(character) < 32 for character in value)):
        raise ValueError(f"{name} must be a nonempty identifier without surrounding whitespace")
    return value


def _seconds(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _sum(values):
    try:
        # Decimal input quantities must not produce a breach from binary rounding.
        result = float(sum(Fraction(str(value)) for value in values))
    except OverflowError as error:
        raise ValueError("Combined resource quantities must remain finite") from error
    if not math.isfinite(result):
        raise ValueError("Combined resource quantities must remain finite")
    return result


def _json_value(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Metadata object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("Metadata must contain JSON values with finite numbers")


def _reservation(estimate_gpu_seconds, shutdown_gpu_seconds, metadata):
    estimate = _seconds(estimate_gpu_seconds, "estimate_gpu_seconds")
    shutdown = _seconds(shutdown_gpu_seconds, "shutdown_gpu_seconds")
    _sum((estimate, shutdown))
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    return {"estimate_gpu_seconds": estimate, "shutdown_gpu_seconds": shutdown,
            "metadata": _json_value(metadata)}


def _utc(value):
    if not isinstance(value, str):
        raise ValueError("Allocation timestamps must be timezone-aware ISO UTC strings")
    try:
        instant = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Allocation timestamps must be timezone-aware ISO UTC strings") from error
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise ValueError("Allocation timestamps must be timezone-aware ISO UTC strings")
    return instant


def _allocations(values):
    if not isinstance(values, list):
        raise ValueError("allocations must be a list")
    parsed = []
    for value in values:
        _fields(value, {"device", "start_utc", "stop_utc"}, "allocation")
        device = _identifier(value["device"], "device")
        start, stop = _utc(value["start_utc"]), _utc(value["stop_utc"])
        if stop <= start:
            raise ValueError("Allocation stop must follow its start; use [] for no GPU allocation")
        parsed.append((device, start, stop))
    parsed.sort()
    allocations = [{"device": device, "start_utc": start.isoformat().replace("+00:00", "Z"),
                    "stop_utc": stop.isoformat().replace("+00:00", "Z")}
                   for device, start, stop in parsed]
    return allocations, _sum((stop - start).total_seconds() for _, start, stop in parsed)


def _arm_shares(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("arm_shares must name at least one arm")
    shares = {_identifier(arm, "arm"): _seconds(share, "arm share")
              for arm, share in value.items()}
    if (any(share > 1 for share in shares.values())
            or not math.isclose(_sum(shares.values()), 1.0, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("arm_shares must sum to 1")
    return dict(sorted(shares.items()))


def _entry(run_id, reservation, allocations, outcome, arm_shares,
           cpu_seconds, wall_seconds, reason):
    _identifier(run_id, "run_id")
    if not isinstance(outcome, str) or outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("reason must be nonempty text")
    if outcome != "completed" and reason is None:
        raise ValueError("A failed, partial or cancelled run requires a reason")
    allocations, gpu_seconds = _allocations(allocations)
    return {"run_id": run_id, **reservation, "allocations": allocations,
            "gpu_seconds": gpu_seconds, "outcome": outcome,
            "arm_shares": _arm_shares(arm_shares),
            "cpu_seconds": _seconds(cpu_seconds, "cpu_seconds"),
            "wall_seconds": _seconds(wall_seconds, "wall_seconds"), "reason": reason,
            "reservation_overrun": gpu_seconds > _sum((reservation["estimate_gpu_seconds"],
                                                       reservation["shutdown_gpu_seconds"]))}


def _fields(value, fields, name):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Unexpected {name} fields")


def _check_overlaps(entries):
    intervals = sorted((allocation["device"], _utc(allocation["start_utc"]),
                        _utc(allocation["stop_utc"]))
                       for entry in entries for allocation in entry["allocations"])
    for previous, current in zip(intervals, intervals[1:]):
        if previous[0] == current[0] and current[1] < previous[2]:
            raise ValueError(f"Overlapping allocation intervals for device {current[0]}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate ledger JSON key: {key}")
        result[key] = value
    return result


class BudgetLedger:
    """A fixed-cap ledger for cooperating processes on a local Linux filesystem.

    Reserve the upper estimate plus shutdown allowance before launch. Record real
    allocated-device intervals, including idle time; the ledger does not measure
    devices or stop jobs. Each run settles once, including failures and partial
    work. Any reservation overrun or cap breach permanently blocks new reserves.
    Existing reservations can still settle after a breach.

    Each operation reloads under a parent-directory flock, then mutations fsync
    and atomically replace the JSON file. Keep the directory stable and use this
    API for all writers. No reservation expires automatically after a crash.
    """

    def __init__(self, path, *, budget_id, gpu_seconds):
        path = Path(path)
        if not path.name or path.name in {".", ".."}:
            raise ValueError("A ledger file path is required")
        self.path = path.parent.resolve(strict=True) / path.name
        self.budget_id = _identifier(budget_id, "budget_id")
        self.gpu_seconds = _seconds(gpu_seconds, "gpu_seconds")

    @classmethod
    def create(cls, path, *, budget_id, gpu_seconds):
        """Create a ledger without replacing any existing file or symlink."""
        ledger = cls(path, budget_id=budget_id, gpu_seconds=gpu_seconds)
        with ledger._locked() as directory_fd:
            state = {"schema_version": SCHEMA_VERSION, "budget_id": ledger.budget_id,
                     "gpu_seconds": ledger.gpu_seconds, "reservations": {}, "entries": [],
                     "budget_breached": False}
            ledger._write(state, directory_fd, create=True)
        return ledger

    @classmethod
    def open(cls, path, *, budget_id, gpu_seconds):
        """Open an existing ledger only if its budget identity and cap match."""
        ledger = cls(path, budget_id=budget_id, gpu_seconds=gpu_seconds)
        ledger.snapshot()
        return ledger

    @contextmanager
    def _locked(self):
        directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX)
            yield directory_fd
        finally:
            os.close(directory_fd)

    def _read(self):
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Ledger must be a regular file")
            state = json.load(stream, object_pairs_hook=_unique_object)
        _fields(state, {"schema_version", "budget_id", "gpu_seconds", "reservations",
                        "entries", "budget_breached"}, "ledger")
        if type(state["schema_version"]) is not int or state["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported ledger schema_version")
        if (state["budget_id"] != self.budget_id
                or _seconds(state["gpu_seconds"], "gpu_seconds") != self.gpu_seconds):
            raise ValueError("Ledger budget_id or fixed gpu_seconds cap differs")
        if not isinstance(state["reservations"], dict) or not isinstance(state["entries"], list):
            raise ValueError("Invalid ledger reservations or entries")
        seen = set()
        for run_id, reservation in state["reservations"].items():
            _identifier(run_id, "run_id")
            _fields(reservation, {"estimate_gpu_seconds", "shutdown_gpu_seconds", "metadata"},
                    "reservation")
            _reservation(**reservation)
            seen.add(run_id)
        for entry in state["entries"]:
            _fields(entry, {"run_id", "estimate_gpu_seconds", "shutdown_gpu_seconds", "metadata",
                            "allocations", "gpu_seconds", "outcome", "arm_shares", "cpu_seconds",
                            "wall_seconds", "reason", "reservation_overrun"}, "entry")
            reservation = _reservation(entry["estimate_gpu_seconds"], entry["shutdown_gpu_seconds"],
                                       entry["metadata"])
            expected = _entry(entry["run_id"], reservation, entry["allocations"], entry["outcome"],
                              entry["arm_shares"], entry["cpu_seconds"], entry["wall_seconds"],
                              entry["reason"])
            if (entry != expected or type(entry["reservation_overrun"]) is not bool
                    or type(entry["gpu_seconds"]) not in (int, float)):
                raise ValueError("Ledger entry differs from its allocation or reservation evidence")
            if entry["run_id"] in seen:
                raise ValueError("A run_id may appear only once in the ledger")
            seen.add(entry["run_id"])
        _check_overlaps(state["entries"])
        summary = self._snapshot(state)
        if (type(state["budget_breached"]) is not bool
                or state["budget_breached"] != summary["budget_breached"]):
            raise ValueError("Ledger breach flag differs from its recorded costs")
        return state

    def _write(self, state, directory_fd, *, create=False):
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(state, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            if create:
                os.link(temporary, self.path)
            else:
                os.replace(temporary, self.path)
            os.fsync(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _snapshot(state):
        charged = _sum(entry["gpu_seconds"] for entry in state["entries"])
        reserved = _sum(_sum((item["estimate_gpu_seconds"], item["shutdown_gpu_seconds"]))
                        for item in state["reservations"].values())
        reasons = [{"kind": "reservation_overrun", "run_id": entry["run_id"]}
                   for entry in state["entries"] if entry["reservation_overrun"]]
        if charged > state["gpu_seconds"]:
            reasons.append({"kind": "budget_cap_exceeded"})
        return {"state": state, "charged_gpu_seconds": charged, "reserved_gpu_seconds": reserved,
                "remaining_gpu_seconds": _sum((state["gpu_seconds"], -charged)),
                "available_gpu_seconds": _sum((state["gpu_seconds"], -charged, -reserved)),
                "budget_breached": bool(reasons), "breach_reasons": reasons}

    def snapshot(self):
        """Return detached state and balances; a breach can make balances negative."""
        with self._locked():
            return self._snapshot(self._read())

    def reserve(self, run_id, *, estimate_gpu_seconds, shutdown_gpu_seconds, metadata):
        """Durably reserve launch and shutdown costs, returning the new snapshot."""
        _identifier(run_id, "run_id")
        reservation = _reservation(estimate_gpu_seconds, shutdown_gpu_seconds, metadata)
        with self._locked() as directory_fd:
            state = self._read()
            summary = self._snapshot(state)
            if (run_id in state["reservations"]
                    or any(entry["run_id"] == run_id for entry in state["entries"])):
                raise ValueError("run_id is already reserved or recorded")
            if summary["budget_breached"]:
                raise ValueError("Budget breach blocks new reservations")
            requested = _sum((reservation["estimate_gpu_seconds"], reservation["shutdown_gpu_seconds"]))
            if requested > summary["available_gpu_seconds"]:
                raise ValueError("Insufficient available GPU budget for estimate plus shutdown allowance")
            state["reservations"][run_id] = reservation
            result = self._snapshot(state)
            self._write(state, directory_fd)
            return result

    def record(self, run_id, *, allocations, outcome, arm_shares, cpu_seconds, wall_seconds,
               reason=None):
        """Settle one reservation and retain actual costs, even when they breach its budget.

        outcomes are completed, failed, partial and cancelled; the last three
        require a reason. Arm shares remain fixed as supplied and sum to one
        within 1e-12 for floating-point rounding. Return the new snapshot.
        """
        _identifier(run_id, "run_id")
        with self._locked() as directory_fd:
            state = self._read()
            if run_id not in state["reservations"]:
                raise ValueError("run_id has no outstanding reservation")
            entry = _entry(run_id, state["reservations"][run_id], allocations, outcome,
                           arm_shares, cpu_seconds, wall_seconds, reason)
            state["entries"].append(entry)
            _check_overlaps(state["entries"])
            del state["reservations"][run_id]
            result = self._snapshot(state)
            state["budget_breached"] = result["budget_breached"]
            self._write(state, directory_fd)
            return result
