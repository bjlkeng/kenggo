import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from kenggo.pilot_jobs import JobLimits, run_job
from kenggo.resource_ledger import BudgetLedger


class PilotJobTests(unittest.TestCase):
    def test_atomic_progress_replacement_during_output_scan_does_not_kill_the_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, replaced = Path.stat, []

            def replace_after_stat(path, *args, **kwargs):
                info = original(path, *args, **kwargs)
                if path.name == ".writing-progress" and not replaced:
                    os.replace(path, path.with_name("progress.json"))
                    replaced.append(True)
                return info

            program = ("from pathlib import Path; import time; "
                       "Path('job/.writing-progress').write_text('{}'); time.sleep(.2)")
            with patch.object(Path, "stat", replace_after_stat):
                result, ledger = self.run_fixture(root, program)
            self.assertTrue(replaced)
            self.assertEqual(result["status"], "completed")
            self.assertEqual((root / "job/progress.json").read_text(), "{}")
            self.assertEqual(ledger["state"]["entries"][0]["outcome"], "completed")

    def run_fixture(self, root, program, *, device=None, limits=None, name="job"):
        path = root / "ledger.json"
        ledger = (BudgetLedger.open if path.exists() else BudgetLedger.create)(
            path, budget_id="synthetic", gpu_seconds=20)
        result = run_job([sys.executable, "-c", program], root / name, cwd=root,
                         limits=limits or JobLimits(3, 3, 100000, 1), ledger=ledger,
                         run_id=name, arm_shares={"fixture": 1}, device=device)
        return result, ledger.snapshot()

    def test_success_and_failure_both_settle_with_retained_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, ledger = self.run_fixture(root, "print('retained')")
            self.assertEqual(result["status"], "completed")
            self.assertEqual((root / "job/stdout.log").read_text(), "retained\n")
            self.assertEqual(ledger["charged_gpu_seconds"], 0)
            failed, ledger = self.run_fixture(root, "raise RuntimeError('expected')", name="failed")
            self.assertEqual((failed["status"], failed["reason"]), ("failed", "process_exit"))
            self.assertIn("RuntimeError", (root / "failed/stderr.log").read_text())
            self.assertEqual(len(ledger["state"]["entries"]), 2)
            self.assertFalse(ledger["state"]["reservations"])

    def test_synthetic_allocation_includes_sleep_and_process_startup(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, ledger = self.run_fixture(Path(temporary), "import time; time.sleep(.1)", device="fake-gpu")
            charge = ledger["charged_gpu_seconds"]
            self.assertGreaterEqual(charge, .1)
            self.assertLess(abs(charge - result["costs"]["wall_seconds"]), .02)
            self.assertEqual(len(ledger["state"]["entries"][0]["allocations"]), 1)

    def test_wall_and_cpu_limits_terminate_work_and_keep_failures(self):
        for program, expected in (("import time; time.sleep(20)", "wall_budget"),
                                  ("while True: pass", "cpu_budget")):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                result, ledger = self.run_fixture(Path(temporary), program,
                    limits=JobLimits(.5 if expected == "wall_budget" else 3, .1, 100000, 1))
                self.assertEqual(result["reason"], expected)
                self.assertLess(result["costs"]["wall_seconds"], 2)
                self.assertFalse(ledger["state"]["reservations"])

    def test_output_limit_and_existing_destination_do_not_start_unaccounted_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, _ = self.run_fixture(root, "print('x' * 50000)", limits=JobLimits(3, 3, 10000, 1))
            self.assertEqual(result["reason"], "output_budget")
            with self.assertRaises(FileExistsError):
                self.run_fixture(root, "raise AssertionError('must not launch')")

    def test_forced_stop_kills_a_descendant_in_a_separate_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program = ("import subprocess,sys,time,pathlib; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
                "pathlib.Path('child.pid').write_text(str(p.pid)); time.sleep(30)")
            result, _ = self.run_fixture(root, program, limits=JobLimits(.5, 3, 100000, 1))
            self.assertEqual(result["reason"], "wall_budget")
            pid = int((root / "child.pid").read_text())
            status = Path(f"/proc/{pid}/stat")
            self.assertTrue(not status.exists() or status.read_text().rsplit(")", 1)[1].split()[0] == "Z")

    def test_launch_error_is_charged_and_durable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = BudgetLedger.create(root / "ledger.json", budget_id="fixture", gpu_seconds=10)
            with self.assertRaises(FileNotFoundError):
                run_job([str(root / "absent")], root / "job", cwd=root,
                        limits=JobLimits(3, 3, 100000, 1), ledger=ledger, run_id="missing",
                        arm_shares={"fixture": 1}, device="fake-gpu")
            self.assertEqual(json.loads((root / "job/job.json").read_text())["status"], "failed")
            self.assertGreater(ledger.snapshot()["charged_gpu_seconds"], 0)
            self.assertFalse(ledger.snapshot()["state"]["reservations"])


if __name__ == "__main__":
    unittest.main()
