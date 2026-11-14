from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import math
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kenggo.resource_ledger import BudgetLedger


def allocation(start, stop, device="gpu-0"):
    epoch = datetime(2026, 11, 13, tzinfo=timezone.utc)
    return {"device": device, "start_utc": (epoch + timedelta(seconds=start)).isoformat(),
            "stop_utc": (epoch + timedelta(seconds=stop)).isoformat()}


def try_reserve(path, run_id):
    ledger = BudgetLedger.open(path, budget_id="concurrent", gpu_seconds=10)
    try:
        ledger.reserve(run_id, estimate_gpu_seconds=6, shutdown_gpu_seconds=0, metadata={})
    except ValueError:
        return False
    return True


class BudgetLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ledger.json"
        self.ledger = BudgetLedger.create(self.path, budget_id="pilot", gpu_seconds=100)

    def reserve(self, run_id="run", estimate=80, shutdown=10, metadata=None):
        return self.ledger.reserve(run_id, estimate_gpu_seconds=estimate,
                                   shutdown_gpu_seconds=shutdown,
                                   metadata=metadata if metadata is not None else {"purpose": "fixture"})

    def record(self, run_id="run", allocations=None, **kwargs):
        return self.ledger.record(run_id, allocations=allocations if allocations is not None else [],
                                  outcome=kwargs.pop("outcome", "completed"),
                                  arm_shares=kwargs.pop("arm_shares", {"conditioned": 1}),
                                  cpu_seconds=kwargs.pop("cpu_seconds", 2),
                                  wall_seconds=kwargs.pop("wall_seconds", 5), **kwargs)

    def test_estimate_and_shutdown_reserve_available_balance_then_release_unused_cost(self):
        snapshot = self.reserve()
        self.assertEqual(snapshot["charged_gpu_seconds"], 0)
        self.assertEqual(snapshot["remaining_gpu_seconds"], 100)
        self.assertEqual(snapshot["reserved_gpu_seconds"], 90)
        self.assertEqual(snapshot["available_gpu_seconds"], 10)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "shutdown"):
            self.reserve("too-large", estimate=10, shutdown=1)
        self.assertEqual(self.path.read_bytes(), before)
        self.reserve("other", estimate=5, shutdown=5)
        snapshot = self.record(allocations=[allocation(0, 20)])
        self.assertEqual(snapshot["charged_gpu_seconds"], 20)
        self.assertEqual(snapshot["remaining_gpu_seconds"], 80)
        self.assertEqual(snapshot["reserved_gpu_seconds"], 10)
        self.assertEqual(snapshot["available_gpu_seconds"], 70)
        self.assertFalse(snapshot["budget_breached"])

    def test_independent_devices_sum_all_allocated_time_and_fixed_arm_shares(self):
        self.reserve()
        snapshot = self.record(allocations=[allocation(0, 20, "gpu-1"), allocation(0, 20)],
                               arm_shares={"unconditioned": 0.25, "conditioned": 0.75})
        self.assertEqual(snapshot["charged_gpu_seconds"], 40)
        entry = snapshot["state"]["entries"][0]
        self.assertEqual(entry["arm_shares"], {"conditioned": 0.75, "unconditioned": 0.25})
        self.assertEqual({arm: share * entry["gpu_seconds"]
                          for arm, share in entry["arm_shares"].items()},
                         {"conditioned": 30, "unconditioned": 10})
        self.assertEqual([item["device"] for item in entry["allocations"]], ["gpu-0", "gpu-1"])

    def test_overlap_within_one_entry_or_history_rejects_without_losing_reservation(self):
        self.reserve()
        before = self.path.read_bytes()
        for intervals in ([allocation(0, 20), allocation(10, 30)],
                          [allocation(0, 20), allocation(0, 20)],
                          [allocation(0, 30), allocation(10, 20)]):
            with self.subTest(intervals=intervals), self.assertRaisesRegex(ValueError, "Overlapping"):
                self.record(allocations=intervals)
            self.assertEqual(self.path.read_bytes(), before)
        self.record(allocations=[allocation(0, 20)])
        self.reserve("second", estimate=50, shutdown=0)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            self.record("second", allocations=[allocation(19, 30)])
        self.assertEqual(self.path.read_bytes(), before)
        snapshot = self.record("second", allocations=[allocation(20, 30)])
        self.assertEqual(snapshot["charged_gpu_seconds"], 30)

    def test_failed_partial_and_cancelled_attempts_keep_all_costs_and_reasons(self):
        for index, outcome in enumerate(("failed", "partial", "cancelled")):
            self.reserve(outcome, estimate=20, shutdown=2)
            snapshot = self.record(outcome, allocations=[allocation(index * 10, index * 10 + 10)],
                                   outcome=outcome, reason="worker stopped", cpu_seconds=7)
        self.assertEqual(snapshot["charged_gpu_seconds"], 30)
        self.assertEqual([entry["outcome"] for entry in snapshot["state"]["entries"]],
                         ["failed", "partial", "cancelled"])
        self.assertTrue(all(entry["cpu_seconds"] == 7 and entry["reason"] == "worker stopped"
                            for entry in snapshot["state"]["entries"]))

    def test_reservation_overrun_under_cap_is_saved_and_blocks_further_launches(self):
        self.reserve(estimate=10, shutdown=2)
        self.reserve("already-running", estimate=8, shutdown=2)
        snapshot = self.record(allocations=[allocation(0, 13)])
        self.assertTrue(snapshot["budget_breached"])
        self.assertTrue(snapshot["state"]["budget_breached"])
        self.assertEqual(snapshot["charged_gpu_seconds"], 13)
        self.assertEqual(snapshot["breach_reasons"], [{"kind": "reservation_overrun", "run_id": "run"}])
        self.assertGreater(snapshot["available_gpu_seconds"], 0)
        reopened = BudgetLedger.open(self.path, budget_id="pilot", gpu_seconds=100)
        with self.assertRaisesRegex(ValueError, "breach"):
            reopened.reserve("new", estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
        snapshot = self.record("already-running", allocations=[allocation(13, 18)],
                               outcome="partial", reason="budget stop")
        self.assertEqual(snapshot["charged_gpu_seconds"], 18)
        self.assertTrue(snapshot["budget_breached"])
        self.assertEqual(snapshot["state"]["reservations"], {})

    def test_actual_cap_overrun_is_charged_without_resizing_or_resetting_budget(self):
        self.reserve(estimate=90, shutdown=10)
        snapshot = self.record(allocations=[allocation(0, 110)], outcome="failed", reason="stop failed")
        self.assertEqual(snapshot["charged_gpu_seconds"], 110)
        self.assertEqual(snapshot["remaining_gpu_seconds"], -10)
        self.assertEqual(snapshot["available_gpu_seconds"], -10)
        self.assertEqual(snapshot["state"]["gpu_seconds"], 100)
        self.assertIn({"kind": "budget_cap_exceeded"}, snapshot["breach_reasons"])
        with self.assertRaisesRegex(ValueError, "breach"):
            self.reserve("next", estimate=0, shutdown=0)
        with self.assertRaisesRegex(ValueError, "cap differs"):
            BudgetLedger.open(self.path, budget_id="pilot", gpu_seconds=120)

    def test_run_ids_cannot_be_reserved_or_recorded_twice(self):
        with self.assertRaisesRegex(ValueError, "no outstanding"):
            self.record("unknown")
        self.reserve()
        with self.assertRaisesRegex(ValueError, "already"):
            self.reserve()
        self.record(allocations=[allocation(0, 1)])
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "already"):
            self.reserve()
        with self.assertRaisesRegex(ValueError, "no outstanding"):
            self.record()
        self.assertEqual(self.path.read_bytes(), before)

    def test_reopen_retains_outstanding_reservations_and_detaches_caller_state(self):
        metadata = {"purpose": "training", "versions": {"data": "fixture-v1"}, "seed": 0}
        snapshot = self.reserve(metadata=metadata)
        metadata["versions"]["data"] = "changed"
        snapshot["state"]["reservations"].clear()
        reopened = BudgetLedger.open(self.path, budget_id="pilot", gpu_seconds=100)
        snapshot = reopened.snapshot()
        self.assertEqual(snapshot["reserved_gpu_seconds"], 90)
        self.assertEqual(snapshot["state"]["reservations"]["run"]["metadata"]["versions"]["data"],
                         "fixture-v1")
        self.record(allocations=[allocation(0, 1)])
        self.assertEqual(reopened.snapshot()["charged_gpu_seconds"], 1)

    def test_cpu_only_record_is_valid_under_zero_gpu_cap(self):
        path = self.path.parent / "cpu.json"
        ledger = BudgetLedger.create(path, budget_id="cpu-smoke", gpu_seconds=0)
        ledger.reserve("cpu", estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
        snapshot = ledger.record("cpu", allocations=[], outcome="completed", arm_shares={"smoke": 1},
                                 cpu_seconds=11, wall_seconds=8)
        self.assertEqual(snapshot["charged_gpu_seconds"], 0)
        self.assertEqual(snapshot["state"]["entries"][0]["cpu_seconds"], 11)
        self.assertFalse(snapshot["budget_breached"])

    def test_subsecond_accounting_does_not_invent_an_overrun(self):
        ledger = BudgetLedger.create(self.path.parent / "fractional.json", budget_id="tiny", gpu_seconds=0.3)
        for run_id, seconds in (("first", 0.1), ("second", 0.2)):
            ledger.reserve(run_id, estimate_gpu_seconds=seconds, shutdown_gpu_seconds=0, metadata={})
        self.assertEqual(ledger.snapshot()["available_gpu_seconds"], 0)
        for run_id, start, stop in (("first", 0, 0.1), ("second", 0.1, 0.3)):
            snapshot = ledger.record(run_id, allocations=[allocation(start, stop)], outcome="completed",
                                     arm_shares={"arm": 1}, cpu_seconds=0, wall_seconds=0.2)
        self.assertEqual(snapshot["charged_gpu_seconds"], 0.3)
        self.assertEqual(snapshot["remaining_gpu_seconds"], 0)
        self.assertFalse(snapshot["budget_breached"])
        ledger = BudgetLedger.create(self.path.parent / "fractional-one.json", budget_id="tiny", gpu_seconds=0.3)
        ledger.reserve("one", estimate_gpu_seconds=0.1, shutdown_gpu_seconds=0.2, metadata={})
        snapshot = ledger.record("one", allocations=[allocation(0, 0.1), allocation(0.1, 0.3)],
                                 outcome="completed", arm_shares={"arm": 1}, cpu_seconds=0, wall_seconds=0.3)
        self.assertEqual(snapshot["charged_gpu_seconds"], 0.3)
        self.assertFalse(snapshot["budget_breached"])

    def test_create_refuses_existing_outputs_including_dangling_symlinks(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            BudgetLedger.create(self.path, budget_id="other", gpu_seconds=5)
        self.assertEqual(self.path.read_bytes(), before)
        unrelated = self.path.parent / "unrelated.txt"
        unrelated.write_text("keep this", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            BudgetLedger.create(unrelated, budget_id="other", gpu_seconds=5)
        self.assertEqual(unrelated.read_text(), "keep this")
        link = self.path.parent / "dangling"
        link.symlink_to(self.path.parent / "absent")
        with self.assertRaises(FileExistsError):
            BudgetLedger.create(link, budget_id="other", gpu_seconds=5)
        self.assertTrue(link.is_symlink())
        self.assertEqual(sorted(item.name for item in self.path.parent.iterdir()),
                         ["dangling", "ledger.json", "unrelated.txt"])

    def test_open_requires_existing_matching_fixed_header(self):
        for budget_id, cap in (("other", 100), ("pilot", 101)):
            with self.subTest(budget_id=budget_id, cap=cap), self.assertRaisesRegex(ValueError, "differs"):
                BudgetLedger.open(self.path, budget_id=budget_id, gpu_seconds=cap)
        with self.assertRaises(FileNotFoundError):
            BudgetLedger.open(self.path.parent / "missing", budget_id="pilot", gpu_seconds=100)

    def test_invalid_resource_scalars_leave_file_unchanged(self):
        invalid = (-1, math.inf, -math.inf, math.nan, True, "1", None, 10 ** 1000)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                BudgetLedger.create(self.path.parent / "invalid", budget_id="bad", gpu_seconds=value)
            for field in ("estimate_gpu_seconds", "shutdown_gpu_seconds"):
                before = self.path.read_bytes()
                arguments = {"estimate_gpu_seconds": 1, "shutdown_gpu_seconds": 0}
                arguments[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.ledger.reserve("bad", metadata={}, **arguments)
                self.assertEqual(self.path.read_bytes(), before)
        self.reserve()
        before = self.path.read_bytes()
        for value in invalid:
            for field in ("cpu_seconds", "wall_seconds"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.record(**{field: value})
            with self.subTest(share=value), self.assertRaises(ValueError):
                self.record(arm_shares={"arm": value})
        self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_timestamps_intervals_shares_metadata_and_statuses_are_rejected(self):
        for metadata in ({"bad": math.nan}, {"nested": [math.inf]}, {1: "not a string key"},
                         {"bad": object()}, []):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.reserve(metadata=metadata)
        for run_id in ("", " ", " run", "run\n", None, 1):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                self.reserve(run_id)
        self.reserve()
        before = self.path.read_bytes()
        bad_timestamp = ("2026-11-13", "2026-11-13T00:00:00", "2026-11-13T00:00:00+02:00", "bad", None)
        for timestamp in bad_timestamp:
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.record(allocations=[{**allocation(0, 1), "start_utc": timestamp}])
        for intervals in ([allocation(1, 0)], [allocation(0, 0)],
                          [{**allocation(0, 1), "device": ""}], [{"device": "gpu"}], None):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                self.ledger.record("run", allocations=intervals, outcome="completed",
                                   arm_shares={"arm": 1}, cpu_seconds=0, wall_seconds=0)
        for shares in ({}, {"arm": 0.5}, {"arm": 1 + 1e-13}, {"a": 0.8, "b": 0.3}, {"": 1}, []):
            with self.subTest(shares=shares), self.assertRaises(ValueError):
                self.record(arm_shares=shares)
        for outcome, reason in (("unknown", None), ([], None), (None, None), ("failed", None),
                                ("partial", ""), ("cancelled", " ")):
            with self.subTest(outcome=outcome, reason=reason), self.assertRaises(ValueError):
                self.record(outcome=outcome, reason=reason)
        self.assertEqual(self.path.read_bytes(), before)

    def test_failed_atomic_replace_preserves_previous_durable_reservation(self):
        self.reserve()
        before = self.path.read_bytes()
        with patch("kenggo.resource_ledger.os.replace", side_effect=OSError("write interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.record(allocations=[allocation(0, 20)])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.ledger.snapshot()["reserved_gpu_seconds"], 90)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(self.record(allocations=[allocation(0, 20)])["charged_gpu_seconds"], 20)

    def test_concurrent_reservations_cannot_spend_the_same_balance(self):
        path = self.path.parent / "concurrent.json"
        ledger = BudgetLedger.create(path, budget_id="concurrent", gpu_seconds=10)
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(try_reserve, [str(path)] * 8, [f"run-{index}" for index in range(8)]))
        self.assertEqual(sum(results), 1)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["reserved_gpu_seconds"], 6)
        self.assertEqual(snapshot["available_gpu_seconds"], 4)
        self.assertEqual(len(snapshot["state"]["reservations"]), 1)

    def test_reopening_detects_inconsistent_costs_and_breach_state(self):
        self.reserve()
        self.record(allocations=[allocation(0, 20)])
        contents = json.loads(self.path.read_text())
        contents["entries"][0]["gpu_seconds"] = 0
        self.path.write_text(json.dumps(contents), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evidence"):
            self.ledger.snapshot()
        contents["entries"][0]["gpu_seconds"] = 20
        contents["budget_breached"] = True
        self.path.write_text(json.dumps(contents), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "breach flag"):
            self.ledger.snapshot()


if __name__ == "__main__":
    unittest.main()
