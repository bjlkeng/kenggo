"""Small CPU-only tests; all writes and ledger mutations use temporary fixtures."""
import importlib.util
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import signal
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('resume', HERE / 'resume.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=HERE, prefix='fixture-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for key in ('OUT', 'PROD', 'OLD', 'OLD_VERIFY', 'VERIFY'):
            directory = self.root / key.lower()
            directory.mkdir()
            fixture = patch.object(r, key, directory)
            fixture.start()
            self.addCleanup(fixture.stop)

    def ledger(self):
        ledger = r.BudgetLedger.create(r.PROD / 'resources.json', budget_id='fixture', gpu_seconds=604800)
        ledger.reserve('prior', estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
        ledger.record('prior', allocations=[], outcome='completed', arm_shares={'shared': 1},
                      cpu_seconds=20, wall_seconds=5)
        before = ledger.snapshot()['state']
        r.save_json(r.OUT / 'before-resources.json', before)
        return ledger, before

    def study(self):
        ledger, before = self.ledger()
        study = object.__new__(r.ResumedStudy)
        study.before, study.ledger, study.group = before, ledger, '/fixture'
        study.output = r.PROD
        study.base_cpu, study.prep_cpu = 20, 40
        study.closed, study.accounted_cpu = False, 0
        study.started, study.cpu_started = time.monotonic() - 10, time.process_time()
        study.launch = r.now() - timedelta(seconds=10)
        study.deadline = study.launch + timedelta(seconds=604800)
        study.production_end = study.deadline - timedelta(seconds=21600)
        study.limits = dict(cpu_seconds=604800, gpu_seconds=604800, wall_seconds=604800,
                            output_bytes=160 * 1024**3, publication_seconds=21600)
        study.manifest = {'sealed_production_names': [], 'sealed_output_bytes': 0}
        study.manifest_ref = {'path': str(r.OUT / 'manifest.json'), 'sha256': 'fixture'}
        study.preparation = {'cpu_seconds_upper': 40}
        study.guard_error, study.stop_event = None, threading.Event()
        study.phase, study.finalizing = 'testing', False
        study.state = {'status': 'running', 'stages': {}, 'tiers': []}
        return study

    def test_exit_snapshot_is_refreshed_against_current_liveness(self):
        with patch.object(r.old, 'cgroup_pids', return_value={17}), patch.object(r.old, 'proc', return_value=None):
            self.assertEqual(r.live_scope_pids('/fixture'), set())
        with patch.object(r.old, 'cgroup_pids', return_value={17}), patch.object(r.old, 'proc', return_value={'state': 'Z'}):
            self.assertEqual(r.live_scope_pids('/fixture'), set())
        with patch.object(r.old, 'cgroup_pids', return_value={17}), patch.object(r.old, 'proc', return_value={'state': 'R'}):
            self.assertEqual(r.live_scope_pids('/fixture'), {17})

    def test_prior_running_service_blocks(self):
        with patch.object(r, 'unit_status', return_value={'ActiveState': 'active', 'MainPID': '17'}):
            with self.assertRaisesRegex(ValueError, 'not stopped'):
                r.require_old_stopped()

    def test_no_running_pid_in_failed_service_can_be_ignored(self):
        status = {'ActiveState': 'failed', 'MainPID': '0', 'ControlGroup': '/fixture'}
        with patch.object(r, 'unit_status', return_value=status), patch.object(r, 'live_scope_pids', return_value={17}):
            with self.assertRaisesRegex(ValueError, 'live processes'):
                r.require_old_stopped()

    def test_preparation_upper_covers_all_host_cpus_and_has_no_reset(self):
        start = r.now() - timedelta(seconds=100)
        incident = {'observed_utc': start.isoformat(), 'logical_cpus': 12}
        with patch.object(r.os, 'cpu_count', return_value=12):
            record = r.preparation_charge(incident, start + timedelta(seconds=100))
            self.assertEqual(record['cpu_seconds_upper'], 1800)
            with self.assertRaisesRegex(ValueError, 'two-hour'):
                r.preparation_charge(incident, start + timedelta(seconds=7201))
            with self.assertRaises(ValueError):
                r.preparation_charge(incident, start - timedelta(seconds=1))
        with patch.object(r.os, 'cpu_count', return_value=6), self.assertRaises(ValueError):
            r.preparation_charge(incident, start + timedelta(seconds=100))

    def test_initial_ledger_rejects_new_entries_and_unresolved_reservations(self):
        ledger, before = self.ledger()
        manifest = {'ledger_before': r.reference(r.OUT / 'before-resources.json')}
        self.assertEqual(r.initial_ledger(manifest), before)
        ledger.reserve('unexpected', estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            r.initial_ledger(manifest)
        r.save_json(r.OUT / 'before2.json', ledger.snapshot()['state'])
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            r.initial_ledger({'ledger_before': r.reference(r.OUT / 'before2.json')})

    def test_prefix_check_retains_all_prior_entries(self):
        _, before = self.ledger()
        current = deepcopy(before)
        current['entries'][0]['cpu_seconds'] = 0
        with self.assertRaisesRegex(ValueError, 'Prior ledger'):
            r.check_prefix(current, before)
        current = deepcopy(before)
        current['gpu_seconds'] += 1
        with self.assertRaisesRegex(ValueError, 'identity'):
            r.check_prefix(current, before)

    def test_one_shot_lock_rejects_concurrent_and_later_start(self):
        first = r.lock_once()
        try:
            with self.assertRaises(BlockingIOError):
                r.lock_once()
        finally:
            first.close()
        r.save_json(r.OUT / 'service-start.json', {})
        with self.assertRaisesRegex(ValueError, 'already started'):
            r.lock_once()

    def test_original_request_is_immutable(self):
        filename = r.PROD / 'tier1-growth-request.json'
        r.save_json(filename, {'rows': 4})
        saved = filename.read_bytes()
        r.immutable_request(filename, {'rows': 4})
        self.assertEqual(filename.read_bytes(), saved)
        with self.assertRaises(ValueError):
            r.immutable_request(filename, {'rows': 5})
        with self.assertRaises(ValueError):
            r.immutable_request(filename, {'rows': 4}, replace=True)

    def test_manifest_and_source_drift_block(self):
        source = r.OUT / 'pinned.py'
        source.write_text('original')
        manifest = r.OUT / 'manifest.json'
        r.save_json(manifest, {'schema': 'e2_validation_resume_manifest_v1',
            'source_commit': r.COMMIT, 'files': {str(source): r.digest(source)}})
        pin = r.digest(manifest)
        with patch.object(r.old, 'verify_manifest'), patch.object(r.old, 'source_frozen') as frozen:
            r.verify_manifest(manifest, pin)
            frozen.assert_called_once()
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Pinned'):
                r.verify_manifest(manifest, pin)
        source.write_text('original')
        with patch.object(r.old, 'verify_manifest'), patch.object(r.old, 'source_frozen', side_effect=ValueError('source drift')):
            with self.assertRaisesRegex(ValueError, 'source drift'):
                r.verify_manifest(manifest, pin)
        manifest.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            r.verify_manifest(manifest, pin)

    def test_live_costs_include_service_descendants_without_double_charging_jobs(self):
        study = self.study()
        with patch.object(r, 'group_cpu', return_value=100):
            study.account_preparation()
            study.ledger.reserve('new-job', estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
            study.ledger.record('new-job', allocations=[], outcome='completed', arm_shares={'shared': 1},
                                cpu_seconds=80, wall_seconds=3)
            costs = study.costs()
        self.assertEqual(costs['cpu_seconds'], 20 + 40 + 100 + 600)
        self.assertEqual(study.ledger.snapshot()['state']['entries'][0], study.before['entries'][0])

    def test_output_size_caches_sealed_paths_and_counts_new_paths(self):
        study = self.study()
        (r.PROD / 'sealed').mkdir()
        (r.PROD / 'sealed/large').write_bytes(b'abc')
        (r.PROD / 'new').mkdir()
        (r.PROD / 'new/new.bin').write_bytes(b'12345')
        study.manifest = {'sealed_production_names': ['sealed', 'resources.json'], 'sealed_output_bytes': 100}
        with patch.object(r, 'size', wraps=r.size) as measure:
            measured = study.output_bytes()
        self.assertEqual(measured, 105 + r.size(r.OUT) + r.size(r.VERIFY))
        self.assertNotIn(r.PROD / 'sealed', [call.args[0] for call in measure.call_args_list])

    def test_absolute_deadline_and_cumulative_cpu_stop_new_work(self):
        study = self.study()
        with patch.object(r, 'group_cpu', return_value=0):
            study.started = time.monotonic() - 604700
            with self.assertRaises(r.original.StopStudy):
                study.check()
            study.started = time.monotonic()
            study.base_cpu = 604300
            with self.assertRaises(r.original.StopStudy):
                study.check()

    def test_guard_error_prevents_new_stage(self):
        study = self.study()
        study.guard_error = 'budget boundary'
        with self.assertRaisesRegex(r.original.StopStudy, 'budget boundary'):
            study.stage('new', {}, wall=1)

    def test_existing_stage_reuse_requires_exact_config(self):
        study = self.study()
        wanted = {'parameters': {'tier': 1}, 'inputs': {'fixed': 1}}
        config = deepcopy(wanted)
        config['parameters'].update(deadline_utc=r.now().isoformat(), max_artifact_bytes=1024**3)
        r.save_json(r.OUT / 'config.json', config)
        r.save_json(r.OUT / 'provenance.json', {})
        r.save_json(r.OUT / 'worker-summary.json', {'status': 'completed'})
        stage = {'status': 'completed', 'config': r.reference(r.OUT / 'config.json'),
                 'provenance': r.reference(r.OUT / 'provenance.json'),
                 'summary': r.reference(r.OUT / 'worker-summary.json')}
        study.state['stages']['reused'] = stage
        with patch.object(study, 'check'), patch.object(r, 'verify_manifest'):
            actual, _ = study.stage('reused', wanted, wall=1)
            self.assertEqual(actual, stage)
            wanted['inputs']['fixed'] = 2
            with self.assertRaisesRegex(ValueError, 'differs'):
                study.stage('reused', wanted, wall=1)

    def test_unregistered_directory_and_unresolved_allocation_block_stage(self):
        study = self.study()
        (r.PROD / 'already').mkdir()
        with patch.object(study, 'check'), patch.object(r, 'verify_manifest'):
            with self.assertRaisesRegex(ValueError, 'unregistered'):
                study.stage('already', {}, wall=1)
            study.ledger.reserve('lost', estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
            with self.assertRaisesRegex(ValueError, 'Unresolved'):
                study.stage('new', {}, wall=1)

    def test_new_preparation_preserves_failed_directory_and_original_parameters(self):
        study = self.study()
        study.manifest.update(composite_request={'path': 'request', 'sha256': 'request'},
                              source_summary={'path': 'source', 'sha256': 'source'})
        old_directory = r.PROD / 'tier1-preparation'
        old_directory.mkdir()
        (old_directory / 'failure.json').write_text('sealed failure')
        config = {'purpose': 'progressive-preparation', 'inputs': {'normalizer': 'original'},
                  'parameters': {'tier': 1}}
        with patch.object(study, 'check'), patch.object(r, 'verify_manifest'), patch.object(r.old, 'gpu_idle'), \
                patch.object(r.original.Study, 'stage', return_value=('stage', 'summary')) as stage:
            result = study.stage('tier1-preparation', config, wall=3600, cpu=3600)
        self.assertEqual(result, ('stage', 'summary'))
        self.assertEqual(stage.call_args.args[0], 'tier1-preparation-validation-recovery')
        self.assertEqual(stage.call_args.args[1]['parameters'], config['parameters'])
        self.assertEqual(stage.call_args.args[1]['inputs']['normalizer'], 'original')
        self.assertEqual((old_directory / 'failure.json').read_text(), 'sealed failure')

    def test_missing_later_stopping_coverage_blocks_before_collection_or_growth(self):
        study = self.study()
        with patch.object(study, 'check'), patch.object(r, 'verify_manifest'), \
                patch.object(study, 'coverage', side_effect=r.original.StopStudy('missing fresh cell')) as coverage, \
                patch.object(r.original.Study, 'stage') as stage:
            with self.assertRaisesRegex(r.original.StopStudy, 'missing fresh cell'):
                study.stage('tier2-sources', {'purpose': 'progressive-sources', 'parameters': {'tier': 2}}, wall=1800)
        coverage.assert_called_once_with(2)
        stage.assert_not_called()

    def test_dispatch_changes_only_cpu_job_entrypoint(self):
        with patch('kenggo.pilot_jobs.run_job') as job:
            cpu_command = ['python', '-m', 'kenggo.progressive_jobs', 'config', 'provenance', 'worker']
            r.dispatch_job(cpu_command, 'output', cwd='repo')
            self.assertEqual(job.call_args.args[0], ['python', str(r.OUT / 'recovered_jobs.py'), 'config', 'provenance', 'worker'])
            training = ['cuda-python', '-m', 'kenggo.continuation_training', 'config', 'provenance', 'worker']
            r.dispatch_job(training, 'output', cwd='repo')
            self.assertEqual(job.call_args.args[0], training)

    def test_summary_and_page_paths_never_replace_originals(self):
        study = self.study()
        originals = {}
        for name in ('progress.json', 'summary.json', 'review.html'):
            filename = r.PROD / name
            filename.write_bytes(('sealed ' + name).encode())
            originals[name] = filename.read_bytes()
        with patch.object(r, 'group_cpu', return_value=100), patch.object(r.old, 'gpu_idle'), \
                patch('kenggo.progressive_review.export', side_effect=lambda ref, dest: dest.write_text('new page')):
            study.account_preparation()
            summary = study.finish('completed', 'finished')
        for name, saved in originals.items():
            self.assertEqual((r.PROD / name).read_bytes(), saved)
        self.assertEqual(r.load(r.OUT / 'summary.json'), summary)
        self.assertEqual((r.OUT / 'review.html').read_text(), 'new page')
        self.assertEqual(summary['costs']['cpu_seconds'], 760)
        entries = study.ledger.snapshot()['state']['entries']
        self.assertEqual(entries[:len(study.before['entries'])], study.before['entries'])
        self.assertEqual(sum(item['cpu_seconds'] for item in entries), 760)

    def test_export_failure_is_retained_and_accounted(self):
        study = self.study()
        with patch.object(r, 'group_cpu', return_value=100), patch.object(r.old, 'gpu_idle'), \
                patch('kenggo.progressive_review.export', side_effect=RuntimeError('fixture export failure')):
            study.account_preparation()
            result = study.finish('failed', 'fixture training failure')
        self.assertIn('export_failure', result)
        self.assertEqual(r.load(r.OUT / 'export-failure.json')['reason'], 'fixture export failure')
        self.assertEqual(study.ledger.snapshot()['state']['reservations'], {})
        self.assertEqual(study.ledger.snapshot()['state']['entries'][-1]['outcome'], 'partial')

    def test_unknown_allocation_is_never_settled_as_stopped(self):
        study = self.study()
        with patch.object(r, 'group_cpu', return_value=100):
            study.account_preparation()
            study.ledger.reserve('unknown-end', estimate_gpu_seconds=20, shutdown_gpu_seconds=10, metadata={})
            with self.assertRaisesRegex(RuntimeError, 'Unsettled'):
                study.finish('failed', 'job cleanup failed')
        self.assertIn('unknown-end', study.ledger.snapshot()['state']['reservations'])
        self.assertFalse((r.OUT / 'summary.json').exists())

    def test_saved_bundle_missing_or_failed_is_rejected(self):
        worker = self.root / 'worker'
        worker.mkdir()
        with self.assertRaises(FileNotFoundError):
            r.old.validate_growth(worker, {}, {})
        r.save_json(worker / 'summary.json', {'schema': 'progressive_growth_job_v1', 'status': 'partial'})
        with self.assertRaisesRegex(ValueError, 'publication'):
            r.old.validate_growth(worker, {}, {})
        r.save_json(worker / 'failure.json', {'reason': 'broken'})
        with self.assertRaisesRegex(ValueError, 'failure'):
            r.old.validate_growth(worker, {}, {})

    def test_launch_does_not_retry_and_uses_service_cgroup_and_absolute_runtime(self):
        session = r.OUT / 'session.json'
        r.save_json(session, {'deadline_utc': (r.now() + timedelta(seconds=5000)).isoformat()})
        manifest = {'session': r.reference(session)}
        with patch.object(r, 'preflight', return_value=(manifest, {})), patch.object(r.subprocess, 'run') as launch:
            r.launch((r.OUT / 'manifest.json').relative_to(Path.cwd()), 'fixture')
            command = launch.call_args.args[0]
            self.assertIn('--property=KillMode=control-group', command)
            self.assertIn('--property=Restart=no', command)
            self.assertEqual(command[-2], str((r.OUT / 'manifest.json').resolve()))
            runtime = next(arg for arg in command if 'RuntimeMaxSec=' in arg)
            seconds = int(runtime.split('=')[-1][:-1])
            self.assertTrue(4870 <= seconds <= 4880)
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                r.launch(r.OUT / 'manifest.json', 'fixture')
            self.assertEqual(launch.call_count, 1)

    def mocked_run(self, execute):
        manifest = r.OUT / 'manifest.json'
        r.save_json(manifest, {})
        r.save_json(r.OUT / 'incident.json', {})
        study = Mock()
        study.deadline = r.now() + timedelta(days=1)
        study.group, study.preparation, study.guard_error = '/fixture', {}, None
        study.stop_event = threading.Event()
        handlers = {}
        old_save = r.original.save_json
        with patch.object(r, 'preflight', return_value=({}, {})), \
                patch.object(r, 'ResumedStudy', return_value=study), \
                patch.object(r, 'preparation_charge', return_value={}), \
                patch.object(r.old, 'proc', return_value={'start': 1}), \
                patch.object(r.threading, 'Thread') as thread, \
                patch.object(r.signal, 'signal', side_effect=lambda number, handler: handlers.update({number: handler})), \
                patch.object(r.original, 'execute', side_effect=lambda _: execute(study, handlers)):
            result = r.run(manifest, 'fixture')
        self.assertIs(r.original.save_json, old_save)
        thread.return_value.join.assert_called_once_with(timeout=6)
        self.assertTrue(study.stop_event.is_set())
        return result, study, handlers

    def test_stage_exception_retains_failure_and_finalizes_once(self):
        def fail(study, handlers):
            raise RuntimeError('fixture stage failed')
        result, study, _ = self.mocked_run(fail)
        self.assertEqual(result, 1)
        self.assertEqual(r.load(r.OUT / 'resume-failure.json')['reason'], 'fixture stage failed')
        study.finish.assert_called_once_with('failed', 'RuntimeError: fixture stage failed')

    def test_termination_ignores_repeated_signals_during_cleanup(self):
        def terminate(study, handlers):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        result, study, handlers = self.mocked_run(terminate)
        self.assertEqual(result, 0)
        self.assertEqual(handlers[signal.SIGTERM], signal.SIG_IGN)
        self.assertEqual(handlers[signal.SIGINT], signal.SIG_IGN)
        study.finish.assert_called_once_with('budget_or_evidence_stop', 'Resume service termination requested')

    def test_inherited_job_runner_reaps_and_settles_after_supervisor_interrupt(self):
        import kenggo.pilot_jobs as jobs
        ledger, _ = self.ledger()
        process = SimpleNamespace(pid=555, returncode=None)
        usage = SimpleNamespace(ru_utime=0, ru_stime=0)
        with patch.object(jobs.subprocess, 'Popen', return_value=process), \
                patch.object(jobs, '_usage', side_effect=[r.original.StopStudy('fixture signal'), (0, 0, {555})]), \
                patch.object(jobs, '_alive', return_value=False), \
                patch.object(jobs, '_kill') as cleanup, \
                patch.object(jobs.os, 'wait4', return_value=(555, 0, usage)) as wait:
            with self.assertRaisesRegex(r.original.StopStudy, 'fixture signal'):
                jobs.run_job(['fixture'], self.root / 'job', cwd=self.root,
                    limits=jobs.JobLimits(60, 60, 1024**2), ledger=ledger, run_id='interrupt',
                    arm_shares={'shared': 1})
        cleanup.assert_called_once_with(555, {555})
        wait.assert_called_once()
        self.assertEqual(ledger.snapshot()['state']['reservations'], {})
        entry = ledger.snapshot()['state']['entries'][-1]
        self.assertEqual(entry['outcome'], 'failed')
        self.assertEqual(entry['reason'], 'StopStudy')


if __name__ == '__main__':
    unittest.main()
