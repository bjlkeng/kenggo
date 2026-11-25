"""CPU synthetic recovery regressions. Only the lifetime test starts a short test service."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from copy import deepcopy
from datetime import timedelta
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('recovery', HERE / 'recovery.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
sys.path.insert(0, str(r.ROOT / 'tests'))
from test_progressive_training import Fixture, SyntheticStudy
from test_data_growth import parent_fixture, sources_for, request_for
from kenggo import data_growth


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=HERE, prefix='fixture-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def worker(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        self.addCleanup(lambda: child.poll())
        return child

    def test_wrong_start_identity_is_rejected(self):
        with self.assertRaises(ValueError):
            r.Worker(os.getpid(), 0)

    def test_pidfd_stops_only_owned_process(self):
        child, other = self.worker(), self.worker()
        owned = r.Worker(child.pid, r.proc(child.pid)['start'])
        self.assertTrue(owned.alive())
        owned.kill()
        child.wait(timeout=5)
        self.assertFalse(owned.alive())
        self.assertIsNone(other.poll())
        owned.close()
        other.kill()
        other.wait()

    def test_identity_mismatch_stop_hook_does_not_signal(self):
        child = self.worker()
        with patch.object(r, 'cgroup_pids', return_value={child.pid}):
            result = r.stop_old_worker({'worker_pid': child.pid, 'worker_start_ticks': 0, 'worker_cgroup': r.cgroup(child.pid)})
        self.assertEqual(result['status'], 'identity_mismatch_no_signal')
        self.assertIsNone(child.poll())
        child.kill()
        child.wait()

    def test_double_adoption_and_prior_adoption_are_rejected(self):
        first = r.lock_once(self.root)
        with self.assertRaises(BlockingIOError):
            r.lock_once(self.root)
        first.close()
        (self.root / 'adopted.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'already'):
            r.lock_once(self.root)

    def test_counter_retains_missing_scope_cost_and_race_bound(self):
        with patch.object(r.time, 'monotonic', return_value=100):
            counter = r.ScopeCounter('/fixture', 500, r.now().isoformat(), 12)
            with patch.object(r, 'cgroup_cpu', return_value=700):
                self.assertEqual(counter.sample(), 712)
        with patch.object(r.time, 'monotonic', return_value=105), patch.object(r, 'cgroup_cpu', side_effect=FileNotFoundError):
            self.assertEqual(counter.sample(final=True), 772)
        with patch.object(r.time, 'monotonic', return_value=99999):
            self.assertEqual(counter.sample(), 772)

    def test_counter_decrease_is_not_hidden(self):
        counter = r.ScopeCounter('/fixture', 500, r.now().isoformat(), 12)
        with patch.object(r, 'cgroup_cpu', return_value=499), self.assertRaises(ValueError):
            counter.sample()

    def ledger(self):
        ledger = r.BudgetLedger.create(self.root / 'resources.json', budget_id='fixture', gpu_seconds=604800)
        ledger.reserve('settled', estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
        ledger.record('settled', allocations=[], outcome='completed', arm_shares={'shared': 1}, cpu_seconds=20, wall_seconds=5)
        for name in ('tier1-growth', 'tier1-growth-tracking', 'supervisor'):
            ledger.reserve(name, estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
        return ledger

    def test_exact_once_settlement_preserves_costs_and_original_entries(self):
        ledger = self.ledger()
        before = ledger.snapshot()['state']
        r.settle_old(ledger, before, 1000, 900)
        state = ledger.snapshot()['state']
        self.assertEqual(state['entries'][0], before['entries'][0])
        self.assertEqual(sum(e['cpu_seconds'] for e in state['entries']), 1000)
        self.assertEqual(sum(e['wall_seconds'] for e in state['entries']), 905)
        self.assertEqual(state['reservations'], {})
        self.assertEqual(state['entries'][1]['outcome'], 'partial')
        with self.assertRaises(ValueError):
            r.settle_old(ledger, before, 1000, 900)
        self.assertEqual(ledger.snapshot()['state'], state)

    def test_existing_request_must_be_byte_identical(self):
        target = self.root / 'tier1-growth-request.json'
        r.save_json(target, {'frozen': 1})
        with patch.object(r, 'OUTPUT', self.root):
            before = target.read_bytes()
            r.immutable_request(target, {'frozen': 1})
            self.assertEqual(target.read_bytes(), before)
            with self.assertRaises(ValueError):
                r.immutable_request(target, {'frozen': 2})
            target.write_text('{"frozen": 1}\n')
            with self.assertRaises(ValueError):
                r.immutable_request(target, {'frozen': 1})

    def test_existing_stage_only_ignores_generated_deadline(self):
        wanted = {'parameters': {'tier': 1}, 'inputs': {'pin': 'stable'}}
        saved = deepcopy(wanted)
        saved['parameters'].update(deadline_utc=r.now().isoformat(), max_artifact_bytes=1024**3)
        r.matching_stage(saved, wanted, 24 * 1024**3)
        saved['inputs']['pin'] = 'changed'
        with self.assertRaises(ValueError):
            r.matching_stage(saved, wanted, 24 * 1024**3)

    def test_growth_no_summary_partial_or_failed_output_is_rejected(self):
        worker = self.root / 'worker'
        worker.mkdir()
        with self.assertRaises(FileNotFoundError):
            r.validate_growth(worker, {}, {})
        r.save_json(worker / 'summary.json', {'schema': 'progressive_growth_job_v1', 'status': 'partial'})
        with self.assertRaisesRegex(ValueError, 'publication'):
            r.validate_growth(worker, {}, {})
        r.save_json(worker / 'failure.json', {'status': 'failed'})
        with self.assertRaisesRegex(ValueError, 'failure'):
            r.validate_growth(worker, {}, {})

    def test_published_growth_validates_then_rejects_corrupt_shard(self):
        parent = parent_fixture(self.root)
        _, sources = sources_for(self.root, parent)
        request = request_for(parent, sources, 4)
        request_path = self.root / 'request.json'
        r.save_json(request_path, request)
        request_ref = r.reference(request_path)
        worker = self.root / 'worker'
        dataset = data_growth.build_growth_round(request, worker / 'dataset')
        publication = {'schema': 'progressive_growth_job_v1', 'status': 'completed', 'growth': dataset,
                       'dataset_summary': r.reference(worker / 'dataset/summary.json')}
        r.save_json(worker / 'summary.json', publication)
        artifacts = r.read(parent)['artifacts']
        base = {'rules': parent, 'normalizer': artifacts['normalizer'],
                'examples': artifacts['examples_manifest'], 'split': artifacts['split_manifest']}
        self.assertEqual(r.validate_growth(worker, request_ref, base), publication)
        manifest_ref = dataset['artifacts']['examples_manifest']
        manifest = r.read(manifest_ref)
        shard = r.path(manifest_ref).parent / manifest['artifacts']['train'][-1]['arrays']
        shard.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'shard hash'):
            r.validate_growth(worker, request_ref, base)

    def study(self):
        fixture = Fixture(self.root)
        initial = r.original.Study(fixture.plan, fixture.plan_ref, self.root / 'study')
        initial.launch = r.now() - timedelta(hours=7)
        initial.deadline = initial.launch + timedelta(seconds=604800)
        initial.production_end = initial.deadline - timedelta(seconds=21600)
        initial.state.update(launch_utc=initial.launch.isoformat(), deadline_utc=initial.deadline.isoformat())
        recovery = self.root / 'recovery'
        recovery.mkdir()
        r.save_json(recovery / 'before-progress.json', initial.state)
        r.save_json(recovery / 'before-session.json', {k: initial.state[k] for k in
            ('limits', 'launch_utc', 'deadline_utc')})
        r.save_json(recovery / 'before-resources.json', initial.ledger.snapshot()['state'])
        config = {'plan': fixture.plan_ref['path']}
        counter = unittest.mock.Mock()
        counter.sample.return_value = 26000
        with patch.object(r, 'OUTPUT', initial.output), patch.object(r, 'RECOVERY', recovery), \
             patch.object(r, 'cgroup', return_value='/app.slice/' + r.SERVICE):
            recovered = r.RecoveredStudy(config, fixture.input, counter)
        return recovered, fixture, recovery

    def test_rehydration_preserves_deadline_reserves_and_charges_recovery(self):
        study, _, recovery = self.study()
        with patch.object(r, 'RECOVERY', recovery), patch.object(r, 'cgroup_cpu', return_value=50):
            costs = study.costs()
            self.assertGreater(costs['elapsed_seconds'], 7 * 3600 - 1)
            self.assertEqual(costs['cpu_seconds'], 26000 + 3600 + 50 + 600)
            self.assertEqual((study.deadline - study.launch).total_seconds(), 604800)
            self.assertEqual((study.deadline - study.production_end).total_seconds(), 21600)
            study.require(1, 1, 1)
            with self.assertRaises(r.original.StopStudy):
                study.require(604800, 1, 1)
            with self.assertRaises(r.original.StopStudy):
                study.require(1, 604800, 1)
            with self.assertRaises(r.original.StopStudy):
                study.require(1, 1, 1, gpu=604800)

    def test_replay_reuses_three_stages_without_duplicate_tier_or_jobs(self):
        fixture = Fixture(self.root)
        synthetic = SyntheticStudy(fixture)
        synthetic.execute()
        saved = {key: deepcopy(synthetic.state['stages'][key]) for key in
                 ('tier0-readiness', 'tier1-sources', 'tier1-growth')}
        # Freeze the same generated fields that real Study.stage records.
        for name, stage in saved.items():
            config = r.read(stage['config'])
            config['parameters'].update(deadline_utc=r.now().isoformat(), max_artifact_bytes=
                 256 * 1024**2 if name == 'tier0-readiness' else 1024**3)
            r.save_json(stage['config']['path'], config, replace=True)
            stage['config'] = r.reference(stage['config']['path'])
        recovered = object.__new__(r.RecoveredStudy)
        recovered.plan, recovered.output = fixture.plan, synthetic.output
        recovered.state = {'tiers': [], 'stages': saved}
        recovered.recovery_ref = fixture.input
        recovered.ledger = unittest.mock.Mock()
        recovered.ledger.snapshot.return_value = {'state': {'reservations': {'recovery-supervisor': {}}}}
        recovered.check = lambda: None
        recovered.require = lambda *a, **kw: None
        recovered.persist = lambda: None
        calls = []
        def new_stage(self, name, config, **options):
            calls.append(name)
            return synthetic.stage(name, config, **options)
        with patch.object(r.original.Study, 'stage', new_stage), patch.object(r, 'gpu_idle'), \
             patch.object(r, 'source_frozen'), patch.object(r, 'verify_manifest'), patch.object(r.original, 'save_json', r.immutable_request), \
             patch.object(r, 'OUTPUT', synthetic.output), patch.object(r.original, 'available_memory', return_value=1024**4):
            # Remove later synthetic outputs only; cached first three are immutable.
            import shutil
            for child in synthetic.output.iterdir():
                if child.is_dir() and child.name not in saved:
                    shutil.rmtree(child)
            r.original.execute(recovered)
        self.assertEqual(len(recovered.state['tiers']), 1)
        self.assertFalse(set(calls) & set(saved))
        self.assertEqual(calls[0], 'tier1-preparation')
        self.assertEqual(recovered.state['tiers'][0]['training_stop'], 'validation_plateau')

    def test_source_and_manifest_mutation_are_rejected(self):
        pinned = self.root / 'helper'
        pinned.write_text('frozen')
        manifest = self.root / 'manifest.json'
        r.save_json(manifest, {'files': {str(pinned): r.digest(pinned)}})
        expected = r.digest(manifest)
        r.verify_manifest(manifest, expected)
        pinned.write_text('changed')
        with self.assertRaises(ValueError):
            r.verify_manifest(manifest, expected)
        with patch.object(r.subprocess, 'check_output', return_value=b'bad source'):
            with self.assertRaises(ValueError):
                r.source_frozen()

    def test_gpu_allocation_uncertainty_refuses_continuation(self):
        with patch.object(r.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '1234\n', '')):
            with self.assertRaises(ValueError):
                r.gpu_idle()

    def test_stop_dispatch_survives_broken_production_imports(self):
        import builtins
        child = self.worker()
        identity = {'worker_pid': child.pid, 'worker_start_ticks': r.proc(child.pid)['start'],
                    'worker_cgroup': r.cgroup(child.pid)}
        source = (HERE / 'recovery.py').read_text()
        marker = "if __name__ == '__main__' and sys.argv[1:2] == ['stop']:"
        source = source.replace(marker, marker + "\n    STOP_IDENTITY = fixture_identity\n    RECOVERY = fixture_directory")
        real_import = builtins.__import__
        def broken(name, *args, **kwargs):
            if name == 'kenggo' or name.startswith('kenggo.'):
                raise ImportError('Synthetic production source failure')
            return real_import(name, *args, **kwargs)
        namespace = {'__name__': '__main__', 'fixture_identity': identity, 'fixture_directory': self.root}
        with patch.object(sys, 'argv', ['recovery.py', 'stop']), patch.object(builtins, '__import__', broken):
            with self.assertRaises(SystemExit) as stopped:
                exec(compile(source, str(HERE / 'recovery.py'), 'exec'), namespace)
        self.assertEqual(stopped.exception.code, 0)
        child.wait(timeout=5)
        self.assertEqual(child.returncode, -signal.SIGKILL)
        self.assertEqual(r.load(self.root / 'stop-hook.json')['pids'], [child.pid])

    def test_unresolved_allocation_blocks_new_jobs_even_when_gpu_is_idle(self):
        study, fixture, recovery = self.study()
        study.check = lambda: None
        study.ledger.reserve('unknown-old-gpu-job', estimate_gpu_seconds=100, shutdown_gpu_seconds=10, metadata={})
        with patch.object(r, 'source_frozen'), patch.object(r, 'verify_manifest'), \
             patch.object(r, 'gpu_idle') as queried, patch.object(r.original.Study, 'stage') as launched:
            with self.assertRaisesRegex(ValueError, 'Unresolved job reservations'):
                study.stage('new-assessment', {'inputs': {}, 'parameters': {}}, wall=100, gpu=True)
            launched.assert_not_called()
            queried.assert_not_called()
        self.assertIn('unknown-old-gpu-job', study.ledger.snapshot()['state']['reservations'])

    def test_termination_latches_before_original_loop_can_launch_assessments(self):
        import ast
        import threading
        fixture = Fixture(self.root)
        study = SyntheticStudy(fixture)
        study.guard_error = None
        study.stop_event = threading.Event()
        study.limits = fixture.plan['parameters']['limits']
        study.costs = lambda: {'elapsed_seconds': 1, 'cpu_seconds': 1, 'output_bytes': 1}
        tree = ast.parse((HERE / 'recovery.py').read_text())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        handler = next(node for node in run.body if isinstance(node, ast.FunctionDef) and node.name == 'terminated')
        namespace = {'study': study}
        exec(compile(ast.Module(body=[handler], type_ignores=[]), 'exact-termination-handler', 'exec'), namespace)
        real_stage = study.stage
        def stage(name, config, **options):
            r.RecoveredStudy.check(study)
            if name == 'tier1-candidate-epoch01':
                namespace['terminated'](signal.SIGTERM, None)
            return real_stage(name, config, **options)
        study.stage = stage
        with self.assertRaisesRegex(RuntimeError, 'Durable service termination requested'):
            study.execute()
        self.assertTrue(study.stop_event.is_set())
        self.assertFalse(any('-assess-' in call['name'] or '-comparison' in call['name'] for call in study.calls))

    def test_heartbeat_does_not_signal_again_during_shutdown_cleanup(self):
        import threading
        study = object.__new__(r.RecoveredStudy)
        study.stop_event = threading.Event()
        study.guard_error = None
        def external_shutdown():
            study.guard_error = 'Durable service termination requested'
            study.stop_event.set()
            raise RuntimeError(study.guard_error)
        study.check = external_shutdown
        # The event changes after wait returns, like TERM racing the current check.
        with patch.object(study.stop_event, 'wait', return_value=False), patch.object(r.os, 'kill') as kill, patch.object(r, 'RECOVERY', self.root):
            study.heartbeat()
            kill.assert_not_called()

    def test_launch_requires_review_and_derives_runtime_from_original_deadline(self):
        manifest = self.root / 'manifest.json'
        r.save_json(manifest, {'files': {}})
        pin = r.digest(manifest)
        deadline = r.now() + timedelta(seconds=5000)
        config = {'deadline_utc': deadline.isoformat()}
        r.save_json(self.root / 'recovery-config.json', config)
        with patch.object(r, 'RECOVERY', self.root), patch.object(r, 'source_frozen'), \
             patch.object(r, 'preparation_cost'), patch.object(r.subprocess, 'run') as launched:
            with self.assertRaises(FileNotFoundError):
                r.launch(manifest, pin)
            launched.assert_not_called()
            r.save_json(self.root / 'review-approved.json', {'manifest_sha256': pin, 'verdict': 'no significant findings'})
            r.launch(manifest, pin)
            command = launched.call_args.args[0]
            record = r.load(self.root / 'launch-command.json')
            self.assertLessEqual(record['remaining_runtime_seconds'], 4880)
            self.assertGreater(record['remaining_runtime_seconds'], 4870)
            self.assertIn('--property=Restart=no', command)
            self.assertIn('--property=KillMode=control-group', command)
            self.assertTrue(any(arg.startswith('--property=ExecStopPost=') for arg in command))

    def test_durable_service_survives_launcher_and_stop_hook_stops_external_worker(self):
        child = self.worker()
        start = r.proc(child.pid)['start']
        service = 'kenggo-recovery-fixture-' + str(os.getpid()) + '.service'
        marker = self.root / 'service.json'
        python, script = str(Path(sys.executable).absolute()), str(Path(__file__).absolute())
        command = ['systemd-run', '--user', '--unit=' + service, '--property=Type=exec',
            '--property=RuntimeMaxSec=30s', '--property=CPUAccounting=yes', '--property=KillMode=control-group',
            '--property=TimeoutStopSec=5s', '--setenv=PYTHONDONTWRITEBYTECODE=1',
            '--property=ExecStopPost=' + ' '.join([python, script, '--fixture-stop', str(child.pid), str(start)]),
            '--working-directory=' + str(r.ROOT), python, script, '--fixture-service', str(marker)]
        try:
            launcher = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, err = launcher.communicate(timeout=10)
            self.assertEqual(launcher.returncode, 0, err)
            until = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < until:
                time.sleep(.1)
            info = r.load(marker)
            self.assertIsNone(child.poll())
            self.assertTrue(info['cgroup'].endswith('/' + service))
            self.assertNotEqual(info['cgroup'], r.cgroup(child.pid))
            self.assertNotEqual(info['ppid'], os.getpid())
            self.assertIsNotNone(r.proc(info['pid']))
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=KILL', service], check=True)
            child.wait(timeout=10)
            self.assertEqual(child.returncode, -signal.SIGKILL)
        finally:
            subprocess.run(['systemctl', '--user', 'stop', service], capture_output=True)
            measured = subprocess.run(['systemctl', '--user', 'show', service, '--property=CPUUsageNSec', '--value'],
                                      capture_output=True, text=True, check=True)
            with (HERE / 'preparation-services.jsonl').open('a') as log:
                log.write(json.dumps({'service': service, 'cpu_seconds': int(measured.stdout.strip()) / 1e9}) + '\n')
            subprocess.run(['systemctl', '--user', 'reset-failed', service], capture_output=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--fixture-service']:
        r.save_json(Path(sys.argv[2]), {'pid': os.getpid(), 'ppid': os.getppid(), 'cgroup': r.cgroup(os.getpid())})
        time.sleep(20)
    elif sys.argv[1:2] == ['--fixture-stop']:
        pid = int(sys.argv[2])
        r.stop_old_worker({'worker_pid': pid, 'worker_start_ticks': int(sys.argv[3]), 'worker_cgroup': r.cgroup(pid)})
    else:
        unittest.main()
