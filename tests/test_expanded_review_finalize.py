"""Finalizer boundary checks with tiny child processes and no model queries."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

from kenggo import expanded_review as producer
from kenggo import expanded_review_finalize as finalize
from kenggo.base_pilot import _ref
from kenggo.checkpoints import file_hash, save_json
from kenggo.pilot_jobs import JobLimits

import test_expanded_review_runner as fixtures


class ExpandedReviewFinalizeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExpandedReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        for target, name, value in ((finalize, "ROOT", self.root),
                (producer, "capture_provenance", lambda *args: ({"source": {}}, b"")),
                (finalize, "capture_provenance", lambda *args: ({"source": {}}, b"")),
                (producer, "execute", lambda *args: None)):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        producer.run(self.fixture.path, file_hash(self.fixture.path), self.fixture.output)
        self.path = self.fixture.output / "summary.json"
        self.before = self.fixture.fixture.ledger.snapshot()
        self.helper = self.root / "helper.py"
        self.helper.write_text("""import json, os, sys
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert os.environ['WANDB_MODE'] == 'offline'
output = Path(sys.argv[3])
(output / 'review.html').write_text('<a href="final-resources.json">Final costs</a>')
for name in ('review.manifest.json', 'verification.json', 'browser-checks.json'):
    (output / name).write_text(json.dumps({'status': 'passed'}))
""")

    def run_finalizer(self):
        return finalize.run(self.path, file_hash(self.path), self.helper, file_hash(self.helper))

    def test_one_use_verification_preserves_old_charges_and_records_totals_beside_page(self):
        handlers = [signal.getsignal(signum) for signum in (signal.SIGALRM, signal.SIGPROF)]
        production_bytes = self.path.read_bytes()
        result = self.run_finalizer()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["gpu_seconds"], 0)
        self.assertEqual(result["publication_cpu_charge"], 600)
        self.assertGreaterEqual(result["verification_cpu_seconds"], 600)
        self.assertGreater(result["output_bytes_charged"], result["verification_output_bytes_charged"])
        self.assertEqual(result["ledger"]["state"]["entries"][:-2], self.before["state"]["entries"])
        self.assertFalse(result["ledger"]["state"]["reservations"])
        self.assertEqual(self.path.read_bytes(), production_bytes)
        worker = self.root / finalize.OUTPUT / "worker"
        self.assertEqual(json.loads((worker / "final-resources.json").read_text()), result)
        self.assertIn("review.html", result["artifacts"])
        self.assertEqual([signal.getsignal(signum) for signum in (signal.SIGALRM, signal.SIGPROF)], handlers)
        with self.assertRaisesRegex(ValueError, "one-use"):
            self.run_finalizer()

    def test_expired_deadline_rejects_before_creating_output_or_charging(self):
        result = json.loads(self.path.read_text())
        session_path = self.fixture.output / "session.json"
        session = json.loads(session_path.read_text())
        now = datetime.now(timezone.utc)
        session.update(launch_utc=(now - timedelta(days=7)).isoformat(), deadline_utc=now.isoformat())
        save_json(session_path, session, replace=True)
        result["session"] = _ref(session_path)
        save_json(self.path, result, replace=True)
        with self.assertRaisesRegex(ValueError, "deadline"):
            self.run_finalizer()
        self.assertFalse((self.root / finalize.OUTPUT).exists())
        self.assertEqual(self.fixture.fixture.ledger.snapshot(), self.before)

    def test_changed_helper_ledger_and_source_reject_before_starting_child(self):
        with self.assertRaisesRegex(ValueError, "helper changed"):
            finalize.run(self.path, file_hash(self.path), self.helper, "0" * 64)
        with patch.object(finalize, "capture_provenance", return_value=({"source": {"changed": True}}, b"")):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.run_finalizer()
        self.fixture.fixture.ledger.reserve("outside", estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
        with self.assertRaisesRegex(ValueError, "ledger changed"):
            self.run_finalizer()
        self.assertFalse((self.root / finalize.OUTPUT).exists())

    def test_helper_cpu_limit_kills_child_and_retains_failed_charge(self):
        self.helper.write_text("while True: pass\n")
        with patch.object(finalize, "HELPER_LIMITS", JobLimits(5, .1, 1024**2, 1)):
            with self.assertRaisesRegex(RuntimeError, "helper failed: cpu_budget"):
                self.run_finalizer()
        worker = self.root / finalize.OUTPUT / "worker"
        result = json.loads((worker / "final-resources.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["ledger"]["state"]["reservations"])
        job = json.loads((worker / "job.json").read_text())
        self.assertEqual(job["reason"], "cpu_budget")
        self.assertGreaterEqual(result["verification_cpu_seconds"], 600.1)

    def test_failed_browser_evidence_cannot_finish_even_if_helper_exits_zero(self):
        self.helper.write_text(self.helper.read_text() + "(output / 'browser-checks.json').write_text('{\"status\":\"failed\"}')\n")
        with self.assertRaisesRegex(ValueError, "has not passed"):
            self.run_finalizer()
        result = json.loads((self.root / finalize.OUTPUT / "worker/final-resources.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["ledger"]["state"]["entries"][-1]["outcome"], "failed")


if __name__ == "__main__":
    unittest.main()
