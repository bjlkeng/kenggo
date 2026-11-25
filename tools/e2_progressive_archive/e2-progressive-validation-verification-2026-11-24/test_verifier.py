"""Small fixture checks for service identity, budgets and append-only accounting."""
import copy
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('verifier', HERE / 'verify_resume.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE, prefix='fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prod, self.out, self.resume = [self.root / name for name in ('prod', 'out', 'resume')]
        for path in (self.prod, self.out, self.resume):
            path.mkdir()
        for name, value in (('PROD', self.prod), ('OUT', self.out), ('RESUME', self.resume)):
            mocked = patch.object(v, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)
        mocked = patch.object(v, 'now', return_value=v.DEADLINE - timedelta(hours=2))
        mocked.start()
        self.addCleanup(mocked.stop)

    def ledger(self):
        from kenggo.resource_ledger import BudgetLedger
        ledger = BudgetLedger.create(self.prod / 'resources.json', budget_id='synthetic-verification', gpu_seconds=604800)
        ledger.reserve('old-work', estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
        ledger.record('old-work', allocations=[], outcome='completed', arm_shares={'shared': 1}, cpu_seconds=100, wall_seconds=20)
        initial = ledger.snapshot()['state']
        summary = {'status': 'completed', 'deadline_utc': v.DEADLINE.isoformat(),
                   'costs': {'cpu_seconds': 100, 'output_bytes': 10, 'ledger': {'state': initial}}}
        (self.resume / 'summary.json').write_text(json.dumps(summary))
        (self.out / 'manifest.json').write_text('{}')
        (self.out / 'verification.json').write_text('{"status":"saved_evidence_checks_finished"}')
        (self.out / 'browser-evidence.json').write_text('{"status":"passed"}')
        return ledger, initial, summary

    def test_final_summary_requires_original_deadline_and_exact_settled_ledger(self):
        _, ledger, summary = self.ledger()
        self.assertEqual(v.settled(summary, ledger), 100)
        wrong = copy.deepcopy(summary)
        wrong['deadline_utc'] = (v.DEADLINE + timedelta(days=7)).isoformat()
        with self.assertRaisesRegex(ValueError, 'deadline'):
            v.settled(wrong, ledger)
        wrong = copy.deepcopy(summary)
        wrong['costs']['ledger']['state']['entries'][0]['cpu_seconds'] = 0
        with self.assertRaisesRegex(ValueError, 'ledger differ'):
            v.settled(wrong, ledger)
        with self.assertRaisesRegex(ValueError, 'unsettled'):
            v.settled(summary, {**ledger, 'reservations': {'unknown': {}}})

    def test_missing_and_running_summaries_cannot_trigger_verification(self):
        for summary in (None, {'status': 'running'}):
            with self.assertRaisesRegex(ValueError, 'final resumed'):
                v.settled(summary, {'reservations': {}})

    def identity(self):
        manifest = {'path': str(self.resume / 'manifest.json'), 'sha256': 'x'}
        identity = {'pid': 12345, 'start_ticks': 777, 'cgroup': '/test/' + v.PRODUCTION_UNIT,
                    'command': v.expected_command(manifest)}
        record = {**identity, 'manifest': manifest, 'deadline_utc': v.DEADLINE.isoformat()}
        (self.resume / 'service-start.json').write_text(json.dumps(record))
        launched = {'record': v.ref(self.resume / 'service-start.json'), 'identity': identity, 'manifest': manifest}
        (self.out / 'launch-record.json').write_text(json.dumps({'production': launched}))
        return manifest, identity

    def test_exact_new_unit_pid_start_ticks_cgroup_and_command_are_required(self):
        manifest, identity = self.identity()
        state = {'ActiveState': 'active', 'MainPID': str(identity['pid']), 'ControlGroup': identity['cgroup']}
        with patch.object(v, 'unit_state', return_value=state) as state_read, patch.object(v, 'process', return_value=identity):
            self.assertEqual(v.production_identity(manifest)['identity'], identity)
            state_read.assert_called_once_with(v.PRODUCTION_UNIT)
        for key, value in (('pid', 2), ('start_ticks', 1), ('cgroup', '/other'), ('command', ['old_resume.py'])):
            with self.subTest(key=key), patch.object(v, 'unit_state', return_value=state), \
                 patch.object(v, 'process', return_value={**identity, key: value}):
                with self.assertRaisesRegex(ValueError, 'identity'):
                    v.production_identity(manifest)
        self.assertIn('validation-resume', v.PRODUCTION_UNIT)
        self.assertIn('validation-verification', v.UNIT)
        self.assertNotEqual(v.ENTRY, 'e2-resume-verification-20261124-v1')

    def test_active_production_and_descendants_are_never_stopped(self):
        _, identity = self.identity()
        with patch.object(v, 'unit_state', return_value={'ActiveState': 'active', 'MainPID': '12345'}), \
             patch.object(v, 'process', return_value=identity), patch.object(v.os, 'killpg') as signal_group:
            self.assertFalse(v.stopped())
            signal_group.assert_not_called()
        with patch.object(v, 'unit_state', return_value={'ActiveState': 'inactive', 'MainPID': '0'}), \
             patch.object(v, 'process', return_value=None), patch.object(v, 'group_processes', return_value=[54321]):
            self.assertFalse(v.stopped())

    def test_changed_service_identity_and_launch_record_fail_closed(self):
        _, identity = self.identity()
        with patch.object(v, 'unit_state', return_value={'ActiveState': 'active', 'MainPID': '12345'}), \
             patch.object(v, 'process', return_value={**identity, 'start_ticks': 778}):
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                v.stopped()
        (self.resume / 'service-start.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'record changed'):
            v.stopped()

    def test_stop_requires_no_old_workers_and_no_matching_orphan(self):
        self.identity()
        states = {'ActiveState': 'inactive', 'MainPID': '0'}
        with patch.object(v, 'unit_state', return_value=states), patch.object(v, 'process', return_value=None), \
             patch.object(v, 'group_processes', return_value=[]), patch.object(v.Path, 'glob', return_value=[]):
            self.assertTrue(v.stopped())
        active_old = [states, {'ActiveState': 'active', 'MainPID': '9'}]
        with patch.object(v, 'unit_state', side_effect=active_old), patch.object(v, 'process', return_value=None), \
             patch.object(v, 'group_processes', return_value=[]):
            self.assertFalse(v.stopped())
        orphan = self.root / '9999'
        orphan.mkdir()
        (orphan / 'cmdline').write_bytes(str(self.prod).encode() + b'\0kenggo.continuation_training\0')
        with patch.object(v, 'unit_state', return_value=states), patch.object(v, 'process', return_value=None), \
             patch.object(v, 'group_processes', return_value=[]), patch.object(v.Path, 'glob', return_value=[orphan]):
            self.assertFalse(v.stopped())

    def test_limits_include_passive_cpu_original_allowance_deadline_and_disk(self):
        self.assertIsNone(v.budget_reason(20, 20, 100000, 1000, 1000))
        for args in ((1770, 0), (20, 3570), (20, 20, 604200), (float('nan'), 0), (-1, 0)):
            self.assertIsNotNone(v.budget_reason(*args))
        self.assertIsNotNone(v.budget_reason(0, 0, output_bytes=1024**3))
        self.assertIsNotNone(v.budget_reason(0, 0, production_bytes=160 * 1024**3))
        with patch.object(v, 'now', return_value=v.DEADLINE - timedelta(seconds=100)):
            self.assertIsNotNone(v.budget_reason(0, 0))

    def test_manifest_and_production_pin_changes_block(self):
        helper = self.root / 'helper.py'
        helper.write_text('original')
        production = self.resume / 'manifest.json'
        production.write_text('{}')
        manifest = {'schema': 'e2_validation_resume_verification_manifest_v1', 'source_commit': v.HEAD,
                    'production_manifest': v.ref(production), 'files': {'helper.py': v.digest(helper)}}
        (self.out / 'manifest.json').write_text(json.dumps(manifest))
        expected = v.digest(self.out / 'manifest.json')
        with patch.object(v, 'ROOT', self.root):
            self.assertEqual(v.validate_pins(expected)['files'], 1)
            helper.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'pin changed'):
                v.validate_pins(expected)
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                v.validate_pins('0' * 64)
            production.write_text('{"changed":1}')
            with self.assertRaisesRegex(ValueError, 'Production manifest changed'):
                v.validate_pins(expected)

    def test_helper_never_starts_after_exhausted_budget(self):
        with patch.object(v, 'cpu', return_value=1800), patch.object(v, 'size', return_value=0), patch.object(v.subprocess, 'Popen') as popen:
            with self.assertRaises(ValueError):
                v.run_helper('test', ['false'], '/test', time.monotonic(), 0, 0)
            popen.assert_not_called()

    def test_account_appends_own_wait_and_failure_costs_without_second_preparation(self):
        ledger, initial, summary = self.ledger()
        with patch.object(v, 'stopped', return_value=True), patch.object(v, 'cpu', return_value=12):
            reserved = v.reserve(initial)
            v.account('/synthetic', time.monotonic(), time.monotonic() - 200, initial, reserved, summary)
            final = ledger.snapshot()['state']
            self.assertEqual(final['entries'][:-1], initial['entries'])
            last = final['entries'][-1]
            self.assertEqual(last['run_id'], v.ENTRY)
            self.assertEqual(last['cpu_seconds'], 612)
            self.assertEqual(last['gpu_seconds'], 0)
            self.assertGreaterEqual(last['wall_seconds'], 200)
            self.assertFalse(v.load(self.out / 'final-resources.json')['preparation_charged_again'])
            with self.assertRaisesRegex(ValueError, 'ledger changed'):
                v.account('/synthetic', time.monotonic(), time.monotonic(), initial, reserved, summary)

    def test_budget_stop_is_still_charged_as_partial(self):
        ledger, initial, summary = self.ledger()
        with patch.object(v, 'stopped', return_value=True), patch.object(v, 'cpu', return_value=1790):
            reserved = v.reserve(initial)
            v.account('/synthetic', time.monotonic(), time.monotonic() - 9, initial, reserved, summary, 'Fixture helper failed')
        last = ledger.snapshot()['state']['entries'][-1]
        self.assertEqual(last['outcome'], 'partial')
        self.assertEqual(last['cpu_seconds'], 2390)
        self.assertEqual(last['reason'], 'Fixture helper failed')

    def test_run_settles_failed_helper_after_reservation(self):
        ledger, initial, summary = self.ledger()
        manifest = {'path': str(self.resume / 'manifest.json'), 'sha256': 'fixture'}
        summary['continuation'] = {'manifest': manifest}
        (self.resume / 'summary.json').write_text(json.dumps(summary))
        (self.out / 'launch-record.json').write_text(json.dumps({'production': {'manifest': manifest}}))
        group = '/synthetic/' + v.UNIT
        with patch.object(v.signal, 'signal'), patch.object(v, 'source'), patch.object(v, 'validate_pins'), \
             patch.object(v, 'own_group', return_value=group), patch.object(v, 'cpu', return_value=12), \
             patch.object(v, 'stopped', return_value=True), patch.object(v.time, 'sleep'), \
             patch.object(v, 'stop_descendants'), patch.object(v, 'run_helper', side_effect=RuntimeError('fixture helper failed')):
            with self.assertRaisesRegex(RuntimeError, 'fixture helper failed'):
                v.run()
        final = ledger.snapshot()['state']
        self.assertEqual(final['entries'][:-1], initial['entries'])
        self.assertFalse(final['reservations'])
        self.assertEqual(final['entries'][-1]['outcome'], 'partial')
        self.assertEqual(final['entries'][-1]['cpu_seconds'], 612)
        self.assertIn('fixture helper failed', final['entries'][-1]['reason'])

    def test_unknown_reservations_are_retained_and_never_settled(self):
        ledger, initial, summary = self.ledger()
        with patch.object(v, 'stopped', return_value=True), patch.object(v, 'cpu', return_value=12):
            v.reserve(initial)
            ledger.reserve('unknown-job', estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
            unknown = ledger.snapshot()['state']
            with self.assertRaisesRegex(ValueError, 'Unknown reservations'):
                v.account('/synthetic', time.monotonic(), time.monotonic(), initial, unknown, summary)
            self.assertEqual(ledger.snapshot()['state'], unknown)

    def test_live_writer_prevents_reservation_and_settlement(self):
        ledger, initial, summary = self.ledger()
        with patch.object(v, 'stopped', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Production'):
                v.reserve(initial)
        self.assertEqual(ledger.snapshot()['state'], initial)
        with patch.object(v, 'stopped', return_value=True):
            reserved = v.reserve(initial)
        with patch.object(v, 'stopped', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Production'):
                v.account('/synthetic', time.monotonic(), time.monotonic(), initial, reserved, summary)
        self.assertEqual(ledger.snapshot()['state'], reserved)

    def test_duplicate_launch_marker_is_exclusive(self):
        v.write('launch-record.json', {'attempt': 1}, exclusive=True)
        with self.assertRaises(FileExistsError):
            v.write('launch-record.json', {'attempt': 2}, exclusive=True)
        self.assertEqual(v.load(self.out / 'launch-record.json'), {'attempt': 1})

    def test_descendant_cleanup_cannot_signal_production(self):
        group = '/synthetic/' + v.UNIT
        with patch.object(v, 'own_group', return_value=group), patch.object(v.os, 'kill') as kill:
            with self.assertRaisesRegex(ValueError, 'outside the verifier'):
                v.stop_descendants('/synthetic/' + v.PRODUCTION_UNIT)
            kill.assert_not_called()
        snapshots = [[os_pid := 9001, 9002], [], [], [], []]
        with patch.object(v, 'own_group', return_value=group), patch.object(v.os, 'getpid', return_value=os_pid), \
             patch.object(v, 'group_processes', side_effect=snapshots), patch.object(v, 'process', return_value={'cgroup': group}), \
             patch.object(v.os, 'kill') as kill:
            v.stop_descendants(group)
            kill.assert_called_once_with(9002, v.signal.SIGTERM)

    def test_actual_launch_paths_no_restart_and_both_runtime_units(self):
        (self.out / 'review-approved.json').write_text(json.dumps({'manifest_sha256': 'fixture', 'verdict': 'no significant findings'}))
        production = {'path': str(self.resume / 'manifest.json'), 'sha256': 'production'}
        fake = type('Result', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
        with patch.object(v, 'source'), patch.object(v, 'validate_pins', return_value={'production_manifest': production}), \
             patch.object(v, 'production_identity', return_value={'manifest': production}), \
             patch.object(v, 'unit_state', return_value={'ActiveState': 'inactive'}), patch.object(v.subprocess, 'run', return_value=fake) as launch:
            v.launch('fixture')
            args = launch.call_args.args[0]
            self.assertIn('--unit=' + v.UNIT, args)
            self.assertIn('--property=Restart=no', args)
            self.assertIn(str(self.out / 'verify_resume.py'), args)
            self.assertIn('--setenv=CUDA_VISIBLE_DEVICES=', args)
            self.assertEqual(v.load(self.out / 'launch-record.json')['production']['manifest'], production)
            with self.assertRaisesRegex(ValueError, 'already launched'):
                v.launch('fixture')
            self.assertEqual(launch.call_count, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
