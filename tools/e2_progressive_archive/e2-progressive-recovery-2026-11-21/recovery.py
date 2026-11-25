"""One-shot adoption of the 21 November first growth job; never a new allowance."""
import argparse
import ctypes
import platform
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path('/home/bojji/code/kenggo')
RECOVERY = ROOT / 'runs/e2-progressive-recovery-2026-11-21'
OUTPUT = ROOT / 'runs/e2-progressive-2026-11-21'
COMMIT = '59d29abf57d8c03eec783a30222fa288403c8e7b'
SERVICE = 'kenggo-e2-progressive-recovery-20261121.service'
PREPARATION_CPU = 3600
MARGIN_CPU = 600


def now():
    return datetime.now(timezone.utc)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def proc(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'state': fields[0], 'ppid': int(fields[1]), 'group': int(fields[2]),
                'session': int(fields[3]), 'start': int(fields[19])}
    except FileNotFoundError:
        return None


def cgroup(pid):
    return Path(f'/proc/{pid}/cgroup').read_text().strip().removeprefix('0::')


def cgroup_cpu(group):
    rows = dict(line.split() for line in (Path('/sys/fs/cgroup') / group.lstrip('/') / 'cpu.stat').read_text().splitlines())
    return int(rows['usage_usec']) / 1e6


def cgroup_pids(group):
    try:
        return {int(x) for x in (Path('/sys/fs/cgroup') / group.lstrip('/') / 'cgroup.procs').read_text().split()}
    except FileNotFoundError:
        return set()


def pidfd_call(number, *args):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This recovery requires the reviewed Linux x86_64 pidfd ABI')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(x) for x in args))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


class Worker:
    def __init__(self, pid, start, group=None):
        self.pid, self.start, self.group = pid, start, group
        self.fd = pidfd_call(434, pid, 0)
        actual = proc(pid)
        if actual is None or actual['start'] != start or (group and cgroup(pid) != group):
            os.close(self.fd)
            raise ValueError('Worker PID/start/cgroup identity differs')
        self.poll = select.poll()
        self.poll.register(self.fd, select.POLLIN)

    def alive(self):
        return not self.poll.poll(0)

    def kill(self):
        if self.alive():
            pidfd_call(424, self.fd, signal.SIGKILL, 0, 0)

    def close(self):
        os.close(self.fd)


def stop_old_worker(config):
    """Used by ExecStopPost even after a fatal supervisor failure."""
    group, pid, start = config['worker_cgroup'], config['worker_pid'], config['worker_start_ticks']
    original = proc(pid)
    # A reused leader PID must never authorize signalling its new process tree.
    if original and original['start'] != start:
        return {'status': 'identity_mismatch_no_signal'}
    stopped = []
    for candidate in cgroup_pids(group):
        info = proc(candidate)
        if not info or info['group'] != pid or info['session'] != pid:
            continue
        if candidate == pid and info['start'] != start:
            continue
        try:
            owned = Worker(candidate, info['start'], group)
            owned.kill()
            owned.close()
            stopped.append(candidate)
        except ProcessLookupError:
            pass
    return {'status': 'stop_requested', 'pids': sorted(stopped)}


class ScopeCounter:
    def __init__(self, group, initial, observed, cpus):
        self.group, self.last, self.cpus = group, initial, cpus
        self.when = time.monotonic() - max(0, (now() - datetime.fromisoformat(observed)).total_seconds())
        self.lock = threading.Lock()
        self.final = None

    def sample(self, *, final=False):
        with self.lock:
            if self.final is not None:
                return self.final
            current = time.monotonic()
            try:
                value = cgroup_cpu(self.group)
                if value < self.last:
                    raise ValueError('Original cgroup CPU counter decreased')
                self.last, self.when = value, current
            except FileNotFoundError:
                pass
            # Covers the gap if the scope vanishes between two samples.
            bound = self.last + (current - self.when + 1) * self.cpus
            if final:
                self.final = bound
            return bound


def source_frozen():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], timeout=15).strip()
    if (git('rev-parse', 'HEAD').decode() != COMMIT or git('diff', '--binary', 'HEAD')
            or git('ls-files', '--others', '--exclude-standard')):
        raise ValueError('Production source freeze violated')


def verify_manifest(filename, expected):
    if digest(filename) != expected:
        raise ValueError('Recovery manifest changed')
    manifest = load(filename)
    for name, pin in manifest['files'].items():
        if digest(ROOT / name) != pin:
            raise ValueError('Pinned recovery/input file changed: ' + name)
    return manifest


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=15)
    if result.stdout.strip():
        raise ValueError('GPU has an existing allocation; continuation refused')


def preparation_cost(config):
    # Earlier root probing and two pre-instrumentation lifetime fixtures get a full 600s bound.
    total = 600 + cgroup_cpu(config['preparation_cgroup'])
    record = RECOVERY / 'preparation-services.jsonl'
    if record.exists():
        total += sum(json.loads(line)['cpu_seconds'] for line in record.read_text().splitlines())
    if total > PREPARATION_CPU:
        raise ValueError('Recovery preparation CPU cap exhausted')
    return total


def lock_once(directory):
    handle = (directory / 'recovery.lock').open('ab')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / 'adopted.json').exists():
            raise ValueError('Recovery was already adopted; automatic restart is forbidden')
        return handle
    except BaseException:
        handle.close()
        raise


def matching_stage(saved, wanted, output_bytes):
    saved = deepcopy(saved)
    params = saved['parameters']
    datetime.fromisoformat(params.pop('deadline_utc'))
    if params.pop('max_artifact_bytes') != min(output_bytes, 1024**3) or saved != wanted:
        raise ValueError('Existing stage configuration differs from requested stage')


STOP_IDENTITY = {'worker_pid': 1083877, 'worker_start_ticks': 65573504,
    'worker_cgroup': '/user.slice/user-1001.slice/user@1001.service/app.slice/app-t3code.slice/app-t3code-agents.slice/app-t3code-agents-dd31e7595309.slice/t3code-codex-6d327d95.scope'}


def stop_dispatch(identity, directory):
    record = {**stop_old_worker(identity), 'utc': now().isoformat()}
    fd, temporary = tempfile.mkstemp(dir=directory, prefix='.stop-hook-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(record, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / 'stop-hook.json')
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


# ExecStopPost must still work when production imports or source pins are broken.
if __name__ == '__main__' and sys.argv[1:2] == ['stop']:
    stop_dispatch(STOP_IDENTITY, RECOVERY)
    raise SystemExit(0)



from kenggo import progressive_training as original
from kenggo.checkpoints import json_bytes, save_json
from kenggo.continuation_data import path, read, reference, validate_lineage
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import WandbSink


def immutable_request(filename, value, *, replace=False):
    filename = Path(filename)
    if filename == OUTPUT / 'tier1-growth-request.json' and filename.exists() and not replace:
        if filename.read_bytes() != json_bytes(value):
            raise ValueError('Existing growth request differs byte-for-byte')
        return digest(filename)
    return save_json(filename, value, replace=replace)


def validate_growth(worker, request_ref, base, check=lambda: None):
    """Accept the worker's published bundle, never its temporary encoding shards."""
    if (worker / 'failure.json').exists():
        raise ValueError('Growth worker recorded a failure')
    summary = load(worker / 'summary.json')
    if summary.get('schema') != 'progressive_growth_job_v1' or summary.get('status') != 'completed':
        raise ValueError('No completed growth worker publication')
    dataset = read(summary['dataset_summary'])
    if (path(summary['dataset_summary']) != (worker / 'dataset/summary.json').resolve()
            or summary['growth'] != dataset or dataset['status'] not in ('completed', 'budget_stopped')
            or dataset['request_sha256'] != request_ref['sha256']
            or read(dataset['request']) != read(request_ref)
            or dataset['training_performed'] or dataset['held_out_targets_decoded']
            or dataset['final_targets_decoded'] or not dataset['old_training_shards_preserved']
            or not dataset['protected_memberships_unchanged']):
        raise ValueError('Growth publication does not match the frozen request/contract')
    for name, ref in dataset['artifacts'].items():
        check()
        if digest(path(ref)) != ref['sha256']:
            raise ValueError('Growth artifact hash differs: ' + name)
    inputs = original.dataset_inputs(summary['dataset_summary'], base)
    manifest, _ = validate_lineage(inputs)
    if manifest['counts']['train']['examples'] != dataset['actual_train_rows']:
        raise ValueError('Published row count disagrees with manifest')
    for name, pin in dataset['code_sha256'].items():
        if digest(ROOT / name) != pin:
            raise ValueError('Published growth code changed')
    # Hash every shard including protected files without decoding their targets.
    for shards in manifest['artifacts'].values():
        for shard in shards:
            for kind in ('arrays', 'metadata'):
                check()
                if digest(path(inputs['examples']).parent / shard[kind]) != shard[kind + '_sha256']:
                    raise ValueError('Published shard hash differs')
    return summary


def settle_old(ledger, before, total_cpu, wall):
    state = ledger.snapshot()['state']
    if state != before:
        raise ValueError('Original ledger changed before settlement')
    settled = sum(entry['cpu_seconds'] for entry in before['entries'])
    if total_cpu < settled:
        raise ValueError('Recovered full-scope CPU cannot be below known costs')
    ledger.record('tier1-growth', allocations=[], outcome='partial', arm_shares={'shared': 1},
        cpu_seconds=total_cpu - settled, wall_seconds=wall,
        reason='Original exit unknown. CPU conservatively includes all old-scope work, lost supervisor and tracking; artifact validity is recorded separately.')
    for name in ('tier1-growth-tracking', 'supervisor'):
        ledger.record(name, allocations=[], outcome='partial', arm_shares={'shared': 1},
            cpu_seconds=0, wall_seconds=0,
            reason='Interrupted without final timing. Its CPU is included in tier1-growth full-scope charge, not assumed zero.')


class RecoveredStudy(original.Study):
    def __init__(self, config, manifest_ref, counter):
        self.recovery_config, self.recovery_ref, self.counter = config, manifest_ref, counter
        self.plan = load(ROOT / config['plan'])
        self.output = OUTPUT
        self.state = load(RECOVERY / 'before-progress.json')
        session = load(RECOVERY / 'before-session.json')
        self.limits = session['limits']
        self.launch = datetime.fromisoformat(session['launch_utc'])
        self.deadline = datetime.fromisoformat(session['deadline_utc'])
        self.production_end = self.deadline - timedelta(seconds=self.limits['publication_seconds'])
        self.started = time.monotonic() - (now() - self.launch).total_seconds()
        self.cpu_started, self.accounted_cpu, self.closed = time.process_time(), 0, False
        self.ledger = BudgetLedger.open(self.output / 'resources.json', budget_id=self.plan['budget_id'],
                                        gpu_seconds=self.limits['gpu_seconds'])
        self.old_ledger = load(RECOVERY / 'before-resources.json')
        if self.ledger.snapshot()['state'] != self.old_ledger:
            raise ValueError('Ledger changed since the incident')
        self.service_group = cgroup(os.getpid())
        if not self.service_group.endswith('/' + SERVICE):
            raise ValueError('Recovery must run in its durable systemd service')
        self.state['recovery'] = {'manifest': manifest_ref, 'original_exit_code': None,
            'accounting': 'Original scope cumulative CPU plus recovery service CPU and 3600 CPU seconds preparation upper charge. Wall time since original launch only.'}
        self.settled = False
        self.guard_error = None
        self.stop_event = threading.Event()
        self.phase = 'adopting'

    def costs(self):
        snapshot = self.ledger.snapshot()
        old = self.counter.sample()
        service = cgroup_cpu(self.service_group)
        cpu = max(old + PREPARATION_CPU + service + MARGIN_CPU,
                  sum(e['cpu_seconds'] for e in snapshot['state']['entries']) + (0 if self.closed else MARGIN_CPU))
        return {'elapsed_seconds': max(time.monotonic() - self.started, (now() - self.launch).total_seconds()),
            'cpu_seconds': cpu, 'old_scope_cpu_upper_seconds': old, 'service_cpu_seconds': service,
            'preparation_cpu_upper_seconds': PREPARATION_CPU,
            'output_bytes': original._output_bytes(self.output) + original._output_bytes(RECOVERY),
            'ledger': snapshot}

    def check(self):
        if self.guard_error:
            raise RuntimeError(self.guard_error)
        costs = self.costs()
        if (costs['elapsed_seconds'] >= self.limits['wall_seconds'] - 120
                or costs['cpu_seconds'] >= self.limits['cpu_seconds'] - MARGIN_CPU
                or costs['output_bytes'] >= self.limits['output_bytes'] - 256 * 1024**2):
            raise original.StopStudy('Original overall resource boundary reached')
        return costs

    def heartbeat(self):
        while not self.stop_event.wait(5):
            try:
                costs = self.check()
                save_json(RECOVERY / 'heartbeat.json', {'utc': now().isoformat(), 'pid': os.getpid(),
                    'cgroup': self.service_group, 'phase': self.phase, 'deadline_utc': self.deadline.isoformat(),
                    'worker_pid': self.recovery_config['worker_pid'],
                    'costs': {k: v for k, v in costs.items() if k != 'ledger'}}, replace=True)
            except BaseException as error:
                if self.stop_event.is_set():
                    return
                self.guard_error = type(error).__name__ + ': ' + str(error)
                save_json(RECOVERY / 'guard-stop.json', {'reason': self.guard_error, 'utc': now().isoformat()}, replace=True)
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def stage(self, name, config, **options):
        self.check()
        source_frozen()
        verify_manifest(path(self.recovery_ref), self.recovery_ref['sha256'])
        if name in self.state['stages']:
            stage = self.state['stages'][name]
            matching_stage(read(stage['config']), config, options.get('output_bytes', 2 * 1024**3))
            read(stage['provenance'])
            original.require_completed(stage)
            return stage, read(stage['summary'])
        if (self.output / name).exists():
            raise ValueError('An unexpected existing stage prevents continuation')
        unresolved = set(self.ledger.snapshot()['state']['reservations']) - {'recovery-supervisor'}
        if unresolved:
            raise ValueError('Unresolved job reservations block new stages: ' + ', '.join(sorted(unresolved)))
        gpu_idle()
        self.phase = name
        config = deepcopy(config)
        config['inputs']['recovery_manifest'] = self.recovery_ref
        return super().stage(name, config, **options)

    def settle(self):
        if self.settled:
            raise ValueError('Old recovery settlement already performed')
        total = self.counter.sample(final=True)
        start = datetime.fromisoformat(load(RECOVERY / 'before-tier1-growth-worker-job.json')['allocation_start_utc'])
        settle_old(self.ledger, self.old_ledger, total, (now() - start).total_seconds())
        self.ledger.reserve('recovery-preparation', estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                            metadata={'purpose': 'Conservative upper charge for all recovery setup/probes/tests/review', 'manifest': self.recovery_ref})
        self.ledger.record('recovery-preparation', allocations=[], outcome='completed', arm_shares={'shared': 1},
                           cpu_seconds=PREPARATION_CPU, wall_seconds=0)
        self.ledger.reserve('recovery-supervisor', estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                            metadata={'purpose': 'Service and final validation/tracking/export CPU not charged by new jobs', 'manifest': self.recovery_ref})
        self.settled = True

    def finish(self, status, reason):
        if not self.settled:
            self.settle()
        self.phase = 'finalizing'
        self.state.update(status=status, reason=reason, active_stage=None)
        self.persist()
        # Export before final CPU settlement; retain a separate pinned provisional input.
        provisional = RECOVERY / 'pre-export-summary.json'
        save_json(provisional, self.state)
        from kenggo.progressive_review import export
        try:
            export(reference(provisional), self.output / 'review.html')
        except Exception as error:
            save_json(RECOVERY / 'export-failure.json', {'type': type(error).__name__, 'reason': str(error)})
            self.state['export_failure'] = reference(RECOVERY / 'export-failure.json')
        entries = self.ledger.snapshot()['state']['entries']
        new_cpu = sum(e['cpu_seconds'] for e in entries if e['run_id'] not in
                      {e['run_id'] for e in self.old_ledger['entries']} | {'tier1-growth', 'tier1-growth-tracking', 'supervisor', 'recovery-preparation'})
        overhead = max(0, cgroup_cpu(self.service_group) - new_cpu) + MARGIN_CPU
        self.ledger.record('recovery-supervisor', allocations=[], outcome='completed' if status == 'completed' else 'partial',
            arm_shares={'shared': 1}, cpu_seconds=overhead, wall_seconds=0,
            reason=None if status == 'completed' else reason)
        self.closed = True
        self.persist()
        save_json(self.output / 'summary.json', self.state)
        return self.state


def adopt(study, worker, config):
    job = load(RECOVERY / 'before-tier1-growth-worker-job.json')
    started = datetime.fromisoformat(job['allocation_start_utc'])
    start_elapsed = (now() - started).total_seconds()
    began = time.monotonic()
    limit = job['limits']
    while True:
        costs = study.check()
        pids = cgroup_pids(config['worker_cgroup'])
        for pid in pids:
            info = proc(pid)
            if info and (info['group'] != worker.pid or info['session'] != worker.pid):
                raise ValueError('Another live writer/process entered the original scope')
        elapsed = max((now() - started).total_seconds(), start_elapsed + time.monotonic() - began)
        if (elapsed >= limit['wall_seconds'] - 30
                or costs['old_scope_cpu_upper_seconds'] >= limit['cpu_seconds'] - MARGIN_CPU
                or original._output_bytes(OUTPUT / 'tier1-growth/worker') >= limit['output_bytes']):
            raise original.StopStudy('Original growth job resource boundary reached')
        if not worker.alive():
            if pids:
                raise ValueError('Growth leader exited with remaining processes; outputs are not settled')
            break
        time.sleep(2)
    study.phase = 'validating_saved_growth'
    study.settle()
    source_frozen()
    stage_dir = OUTPUT / 'tier1-growth'
    growth = validate_growth(stage_dir / 'worker', load(stage_dir / 'config.json')['inputs']['request'],
                             study.plan['parameters']['base_inputs'], study.check)
    source_frozen()
    evidence = {'schema': 'e2_recovered_growth_v1', 'status': 'validated_saved_artifacts',
        'exit_code': None, 'original_job': reference(RECOVERY / 'before-tier1-growth-worker-job.json'),
        'original_run': reference(RECOVERY / 'before-tier1-growth-run.json'),
        'worker_summary': reference(stage_dir / 'worker/summary.json'), 'manifest': study.recovery_ref,
        'cpu_seconds_upper': study.counter.sample(), 'observed_exit_by_utc': now().isoformat()}
    save_json(RECOVERY / 'growth-recovered.json', evidence)
    # Mark the interrupted remote run partial. Never rewrite its original local run/job records.
    study.phase = 'tracking_recovered_growth'
    sink = WandbSink(stage_dir, load(stage_dir / 'config.json'), load(stage_dir / 'provenance.json'),
                    load(stage_dir / 'run.json')['id'], mode='online', resume=True)
    sink.log(1, {'recovery/saved_artifacts_valid': True, 'recovery/original_exit_known': False}, 2)
    sink.finish('partial', {'outcome': 'partial', 'reason': 'Original supervisor disappeared; original exit unknown; recovered saved artifacts validated.',
                           'recovery_manifest_sha256': study.recovery_ref['sha256'],
                           'recovery_evidence_sha256': digest(RECOVERY / 'growth-recovered.json')})
    save_json(RECOVERY / 'tracking-recovered.json', {'status': 'partial', 'tracking_finished': True,
               'original_run': reference(RECOVERY / 'before-tier1-growth-run.json'), 'evidence': reference(RECOVERY / 'growth-recovered.json')})
    stage = {'directory': str(stage_dir), 'config': reference(stage_dir / 'config.json'),
        'provenance': reference(stage_dir / 'provenance.json'), 'summary': reference(stage_dir / 'worker/summary.json'),
        'status': 'completed', 'completion_basis': 'validated_saved_artifacts_original_exit_unknown', 'job': evidence}
    save_json(RECOVERY / 'growth-stage.json', stage)
    study.state['stages']['tier1-growth'] = stage
    study.state['tiers'] = []
    study.persist()
    return growth


def run(manifest_path, manifest_hash):
    verify_manifest(manifest_path, manifest_hash)
    config = load(RECOVERY / 'recovery-config.json')
    source_frozen()
    gpu_idle()
    if (now() - datetime.fromisoformat(config['preparation_start_utc'])).total_seconds() > 10800:
        raise ValueError('Recovery preparation wall cap expired')
    preparation_cost(config)
    lock = lock_once(RECOVERY)
    stage_lock = (OUTPUT / 'tier1-growth/.writer.lock').open('ab')
    fcntl.flock(stage_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    worker = Worker(config['worker_pid'], config['worker_start_ticks'], config['worker_cgroup'])
    if Path(f'/proc/{worker.pid}/cmdline').read_bytes().split(b'\0')[:-1] != [s.encode() for s in load(RECOVERY / 'before-tier1-growth-worker-job.json')['command']]:
        raise ValueError('Worker command identity differs')
    incident = load(RECOVERY / 'incident.json')
    counter = ScopeCounter(config['worker_cgroup'], int(incident['cgroup_cpu_stat'].splitlines()[0].split()[1]) / 1e6,
                           incident['observed_utc'], incident['affinity_cpus'])
    study = RecoveredStudy(config, reference(manifest_path), counter)
    study.check()
    if (OUTPUT / 'summary.json').exists() or set(study.state['stages']) != {'tier0-readiness', 'tier1-sources'}:
        raise ValueError('Recovery only supports the first growth interruption without a summary')
    save_json(RECOVERY / 'adopted.json', {'utc': now().isoformat(), 'supervisor_pid': os.getpid(),
        'supervisor_cgroup': study.service_group, 'worker_pid': worker.pid, 'worker_start_ticks': worker.start,
        'deadline_utc': study.deadline.isoformat(), 'manifest': reference(manifest_path)})
    def terminated(signum, frame):
        study.guard_error = study.guard_error or 'Durable service termination requested'
        study.stop_event.set()
        raise InterruptedError(study.guard_error)
    signal.signal(signal.SIGTERM, terminated)
    monitor = threading.Thread(target=study.heartbeat, daemon=True)
    monitor.start()
    status, reason = 'failed', 'Recovery did not finish'
    old_save = original.save_json
    try:
        adopt(study, worker, config)
        verify_manifest(manifest_path, manifest_hash)
        original.save_json = immutable_request
        study.phase = 'continuing_original_study'
        reason = original.execute(study)
        status = 'completed'
    except original.StopStudy as error:
        status, reason = 'budget_or_evidence_stop', str(error)
    except BaseException as error:
        reason = type(error).__name__ + ': ' + str(error)
        save_json(RECOVERY / 'recovery-failure.json', {'type': type(error).__name__, 'reason': str(error)})
    finally:
        original.save_json = old_save
        stop_old_worker(config)
        end = time.monotonic() + 10
        while cgroup_pids(config['worker_cgroup']) and time.monotonic() < end:
            time.sleep(.1)
        if cgroup_pids(config['worker_cgroup']):
            raise RuntimeError('Old scope still has live processes; reservations retained')
        study.finish(status, reason)
        study.stop_event.set()
        monitor.join(timeout=6)
        worker.close()
        lock.close()
        stage_lock.close()
    return 0 if status != 'failed' else 1


def launch(manifest_path, manifest_hash):
    verify_manifest(manifest_path, manifest_hash)
    source_frozen()
    config = load(RECOVERY / 'recovery-config.json')
    preparation_cost(config)
    review = load(RECOVERY / 'review-approved.json')
    if review.get('manifest_sha256') != manifest_hash or review.get('verdict') != 'no significant findings':
        raise ValueError('Clean independent review of these exact files is required')
    if (RECOVERY / 'adopted.json').exists():
        raise ValueError('Recovery already launched')
    remaining = (datetime.fromisoformat(config['deadline_utc']) - now()).total_seconds() - 120
    if remaining <= 120 or remaining > 604800:
        raise ValueError('Original remaining deadline cannot support supervision')
    command = ['systemd-run', '--user', '--unit=' + SERVICE, '--property=Type=exec',
        '--property=Restart=no', '--property=CPUAccounting=yes', '--property=KillMode=control-group',
        '--property=TimeoutStopSec=20s', '--property=RuntimeMaxSec=' + str(int(remaining)) + 's',
        '--property=StandardOutput=append:' + str(RECOVERY / 'service.stdout.log'),
        '--property=StandardError=append:' + str(RECOVERY / 'service.stderr.log'),
        '--property=ExecStopPost=' + ' '.join([str(ROOT / '.venv/bin/python'), str(RECOVERY / 'recovery.py'),
                                            'stop', str(manifest_path), manifest_hash]),
        '--working-directory=' + str(ROOT), '--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=CUDA_VISIBLE_DEVICES=', str(ROOT / '.venv/bin/python'), str(RECOVERY / 'recovery.py'),
        'run', str(manifest_path), manifest_hash]
    save_json(RECOVERY / 'launch-command.json', {'utc': now().isoformat(), 'command': command,
                                              'remaining_runtime_seconds': int(remaining)})
    subprocess.run(command, check=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'launch', 'check'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('sha256')
    args = parser.parse_args()
    if args.mode == 'check':
        verify_manifest(args.manifest, args.sha256)
        source_frozen()
        print('Frozen source and recovery pins verified')
    elif args.mode == 'launch':
        launch(args.manifest, args.sha256)
    else:
        raise SystemExit(run(args.manifest, args.sha256))


if __name__ == '__main__':
    main()
