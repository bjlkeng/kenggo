from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

spec = importlib.util.spec_from_file_location('phase_runner', Path(__file__).with_name('phase-runner.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class PhaseTests(unittest.TestCase):
    def test_preparation_uses_original_start_and_all_host_cpus(self):
        start = r.DEADLINE - timedelta(days=3)
        incident = {'observed_utc': start.isoformat(), 'logical_cpus': 12}
        with patch.object(r.os, 'cpu_count', return_value=12):
            self.assertEqual(r.preparation(incident, start + timedelta(seconds=100))['cpu_seconds_upper'], 1800)
            for end in (start - timedelta(seconds=1), start + timedelta(seconds=7201)):
                with self.assertRaises(ValueError):
                    r.preparation(incident, end)

    def test_watchdog_has_phase_and_original_deadline_bounds(self):
        early = r.DEADLINE - timedelta(days=2)
        self.assertFalse(r.over_limit(10, 10, 10, early))
        self.assertTrue(r.over_limit(1740, 10, 10, early))
        self.assertTrue(r.over_limit(10, 3540, 10, early))
        self.assertTrue(r.over_limit(10, 10, 1024**3, early))
        self.assertTrue(r.over_limit(10, 10, 10, r.DEADLINE - timedelta(seconds=100)))

    def test_changed_source_pin_blocks_before_any_request(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'helper.py').write_text('original')
            (root / 'phase-manifest.json').write_text(json.dumps({'files': {'helper.py': r.sha(root / 'helper.py')}}))
            expected = r.sha(root / 'phase-manifest.json')
            with patch.object(r, 'ROOT', root), patch.object(r, 'OUT', root), patch.object(r, 'source'):
                r.verify(expected)
                (root / 'helper.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Phase file changed'):
                    r.verify(expected)

    def test_live_prior_service_blocks(self):
        with patch.object(r, 'state', return_value={'ActiveState': 'active', 'MainPID': '100'}):
            with self.assertRaisesRegex(ValueError, 'still active'):
                r.old_stopped()

    def test_one_shot_preflight_rejects_changed_ledger_or_missing_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out, prod = root / 'out', root / 'prod'
            out.mkdir(); prod.mkdir()
            before = {'entries': [], 'reservations': {}, 'budget_breached': False}
            for p in (out / 'before-resources.json', prod / 'resources.json'):
                p.write_text(json.dumps(before))
            (out / 'incident.json').write_text('{"prior_cpu_upper_seconds":100}')
            (out / 'review-approved.json').write_text('{"manifest_sha256":"bad","verdict":"no significant findings"}')
            with patch.object(r, 'OUT', out), patch.object(r, 'PROD', prod), patch.object(r, 'verify', return_value={'sealed_output_bytes': 0}), \
                 patch.object(r, 'old_stopped'), patch.object(r, 'preparation', return_value={'cpu_seconds_upper': 600}), \
                 patch.object(r, 'now', return_value=r.DEADLINE - timedelta(days=2)):
                with self.assertRaisesRegex(ValueError, 'Fresh independent review'):
                    r.preflight('expected')
                (out / 'service-start.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'already started'):
                    r.preflight('expected', approved=False)
                (out / 'service-start.json').unlink()
                (prod / 'resources.json').write_text(json.dumps({**before, 'reservations': {'other': {}}}))
                with self.assertRaisesRegex(ValueError, 'ledger changed'):
                    r.preflight('expected', approved=False)

    def test_source_failure_preserves_old_entries_and_charges_cpu_once(self):
        from kenggo.resource_ledger import BudgetLedger
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out, prod = root / 'out', root / 'prod'
            out.mkdir(); prod.mkdir()
            ledger = BudgetLedger.create(prod / 'resources.json', budget_id='fixture', gpu_seconds=604800)
            ledger.reserve('old', estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
            ledger.record('old', allocations=[], outcome='completed', arm_shares={'shared': 1}, cpu_seconds=100, wall_seconds=1)
            before = ledger.snapshot()['state']
            (out / 'incident.json').write_text('{"prior_cpu_upper_seconds":100}')
            (out / 'phase-manifest.json').write_text('{}')
            (out / 'phase-config.json').write_text('{"inputs":{}}')
            fake = MagicMock()
            fake.__enter__.return_value = fake

            def failed_job(*args, **kwargs):
                current = kwargs['ledger']
                current.reserve(r.JOB_ID, estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
                current.record(r.JOB_ID, allocations=[], outcome='partial', arm_shares={'shared': 1}, cpu_seconds=20, wall_seconds=1, reason='fixture')
                return {'status': 'failed', 'reason': 'fixture', 'costs': {'cpu_seconds': 20, 'wall_seconds': 1}}

            with patch.object(r, 'OUT', out), patch.object(r, 'PROD', prod), \
                 patch.object(r, 'preflight', return_value=({'request': {'path': 'fixture', 'sha256': 'fixture'}}, before, {})), \
                 patch.object(r, 'preparation', return_value={'cpu_seconds_upper': 600}), patch.object(r, 'cpu', return_value=100), \
                 patch.object(r, 'group', return_value='/' + r.UNIT), patch.object(r, 'capture_provenance', return_value=({}, b'')), \
                 patch.object(r, 'Experiment', return_value=fake), patch.object(r, 'run_job', side_effect=failed_job), \
                 patch.object(r.threading, 'Thread'), patch.object(r.signal, 'signal'):
                self.assertEqual(r.run('fixture', '/not-read'), 1)
                final = ledger.snapshot()['state']
                self.assertEqual(final['entries'][0], before['entries'][0])
                self.assertFalse(final['reservations'])
                self.assertEqual(sum(e['cpu_seconds'] for e in final['entries']), 100 + 600 + 100 + 600)
                self.assertTrue(all(e['gpu_seconds'] == 0 for e in final['entries']))
                self.assertEqual(r.load(out / 'phase-summary.json')['status'], 'partial')


if __name__ == '__main__':
    unittest.main(verbosity=2)
