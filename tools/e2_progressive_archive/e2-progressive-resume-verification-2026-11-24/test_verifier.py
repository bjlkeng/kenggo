import copy
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('verifier', Path(__file__).with_name('verify_resume.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class VerifierTests(unittest.TestCase):
    def test_final_summary_uses_original_deadline_and_exact_settled_ledger(self):
        ledger = {'entries': [{'cpu_seconds': 4}], 'reservations': {}}
        summary = {'status': 'completed', 'deadline_utc': v.DEADLINE.isoformat(),
                   'costs': {'cpu_seconds': 6, 'ledger': {'state': copy.deepcopy(ledger)}}}
        self.assertEqual(v.settled(summary, ledger), 6)
        wrong = copy.deepcopy(summary)
        wrong['deadline_utc'] = (v.DEADLINE + timedelta(days=7)).isoformat()
        with self.assertRaisesRegex(ValueError, 'deadline'):
            v.settled(wrong, ledger)
        wrong = copy.deepcopy(summary)
        wrong['costs']['ledger']['state']['entries'][0]['cpu_seconds'] = 0
        with self.assertRaisesRegex(ValueError, 'ledger differ'):
            v.settled(wrong, ledger)
        with self.assertRaisesRegex(ValueError, 'unsettled'):
            v.settled(summary, {**ledger, 'reservations': {'running': {}}})

    def test_missing_and_running_summaries_do_not_trigger_verification(self):
        for summary in (None, {'status': 'running'}):
            with self.assertRaisesRegex(ValueError, 'final resumed'):
                v.settled(summary, {'reservations': {}})

    def test_active_production_is_never_stopped(self):
        with patch.object(v, 'unit_state', return_value={'ActiveState': 'active', 'MainPID': '123'}):
            self.assertFalse(v.stopped())

    def test_verifier_bounds_include_original_allowance_and_deadline(self):
        with patch.object(v, 'now', return_value=v.DEADLINE - timedelta(hours=2)):
            self.assertIsNone(v.budget_reason(20, 20, 100000, 1000, 1000))
            self.assertIsNotNone(v.budget_reason(1770, 20))
            self.assertIsNotNone(v.budget_reason(20, 3570))
            self.assertIsNotNone(v.budget_reason(20, 20, 604200))
            self.assertIsNotNone(v.budget_reason(20, 20, output_bytes=1024**3))
        with patch.object(v, 'now', return_value=v.DEADLINE - timedelta(seconds=100)):
            self.assertIsNotNone(v.budget_reason(0, 0))

    def test_file_or_manifest_change_blocks_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'helper.py').write_text('original')
            (root / 'manifest.json').write_text(json.dumps({'files': {'helper.py': v.digest(root / 'helper.py')}}))
            expected = v.digest(root / 'manifest.json')
            with patch.object(v, 'ROOT', root), patch.object(v, 'OUT', root):
                self.assertEqual(v.validate_pins(expected)['files'], 1)
                (root / 'helper.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'pin changed'):
                    v.validate_pins(expected)
                with self.assertRaisesRegex(ValueError, 'manifest changed'):
                    v.validate_pins('0' * 64)

    def test_helper_never_starts_after_exhausted_budget(self):
        with patch.object(v, 'cpu', return_value=1800), patch.object(v, 'size', return_value=0), patch.object(v.subprocess, 'Popen') as popen:
            with self.assertRaises(ValueError):
                v.run_helper('test', ['false'], '/test', time.monotonic(), 0, 0)
            popen.assert_not_called()

    def test_account_appends_only_new_cpu_and_never_recharges_preparation(self):
        from kenggo.resource_ledger import BudgetLedger
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prod, out, resume = [root / name for name in ('prod', 'out', 'resume')]
            for p in (prod, out, resume):
                p.mkdir()
            ledger = BudgetLedger.create(prod / 'resources.json', budget_id='synthetic-resume', gpu_seconds=604800)
            ledger.reserve('old-work', estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
            ledger.record('old-work', allocations=[], outcome='completed', arm_shares={'shared': 1}, cpu_seconds=100, wall_seconds=20)
            initial = ledger.snapshot()['state']
            summary = {'status': 'completed', 'deadline_utc': v.DEADLINE.isoformat(),
                       'costs': {'cpu_seconds': 100, 'output_bytes': 10, 'ledger': {'state': initial}}}
            (resume / 'summary.json').write_text(json.dumps(summary))
            (out / 'manifest.json').write_text('{}')
            (out / 'verification.json').write_text('{"status":"saved_evidence_checks_finished"}')
            (out / 'browser-evidence.json').write_text('{"status":"passed"}')
            with patch.object(v, 'PROD', prod), patch.object(v, 'OUT', out), patch.object(v, 'RESUME', resume), \
                 patch.object(v, 'stopped', return_value=True), patch.object(v, 'cpu', return_value=12), \
                 patch.object(v, 'now', return_value=v.DEADLINE - timedelta(hours=2)):
                v.account('/synthetic', time.monotonic(), initial, summary)
                final = ledger.snapshot()['state']
                self.assertEqual(final['entries'][:-1], initial['entries'])
                self.assertEqual(final['entries'][-1]['cpu_seconds'], 612)
                self.assertEqual(final['entries'][-1]['gpu_seconds'], 0)
                self.assertFalse(v.load(out / 'final-resources.json')['preparation_charged_again'])
                with self.assertRaisesRegex(ValueError, 'ledger changed'):
                    v.account('/synthetic', time.monotonic(), initial, summary)


if __name__ == '__main__':
    unittest.main(verbosity=2)
