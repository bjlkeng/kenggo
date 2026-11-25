"""Continue the original study with a separate supplemented stopping sample."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

STARTED_UTC = datetime.now(timezone.utc)
ROOT = Path('/home/bojji/code/kenggo')
OUT = ROOT / 'runs/e2-progressive-validation-resume-2026-11-24'
PREVIOUS = ROOT / 'runs/e2-progressive-resume-2026-11-24'
PREVIOUS_VERIFY = ROOT / 'runs/e2-progressive-resume-verification-2026-11-24'
SOURCE_PHASE = ROOT / 'runs/e2-validation-gap-2026-11-24'
PROD = ROOT / 'runs/e2-progressive-2026-11-21'
OLD = ROOT / 'runs/e2-progressive-recovery-2026-11-21'
OLD_VERIFY = ROOT / 'runs/e2-progressive-final-verification-2026-11-21'
VERIFY = ROOT / 'runs/e2-progressive-validation-verification-2026-11-24'
UNIT = 'kenggo-e2-progressive-validation-resume-20261124.service'
OLD_UNITS = ('kenggo-e2-progressive-recovery-20261121.service',
             'kenggo-e2-final-verification-20261121.service',
             'kenggo-e2-progressive-resume-20261124.service',
             'kenggo-e2-progressive-resume-verification-20261124.service',
             'kenggo-e2-validation-source-20261124.service')
COMMIT = '59d29abf57d8c03eec783a30222fa288403c8e7b'
OLD_MANIFEST_SHA = '2c94dfb5ac5e06486cd43e18d3826c841d2fa32deffc446b0e58651c60074888'
PREP_ID = 'e2-validation-resume-preparation-20261124-v1'
SUPERVISOR_ID = 'e2-validation-resume-supervisor-20261124-v1'
MARGIN_CPU = 600
PREPARATION_WALL = 7200


def now():
    return datetime.now(timezone.utc)


def load(filename):
    return json.loads(Path(filename).read_text())


def digest(filename):
    with Path(filename).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def group_cpu(group):
    rows = dict(line.split() for line in (Path('/sys/fs/cgroup') / group.lstrip('/') / 'cpu.stat').read_text().splitlines())
    return int(rows['usage_usec']) / 1e6


def own_group():
    return Path('/proc/self/cgroup').read_text().strip().removeprefix('0::')


def stop_record():
    # systemd owns this service's full process tree, including fatal-import paths.
    record = {'utc': now().isoformat(), 'service': UNIT, 'cgroup': own_group(),
              'cpu_seconds': group_cpu(own_group()),
              'service_result': os.environ.get('SERVICE_RESULT'),
              'exit_code': os.environ.get('EXIT_CODE'), 'exit_status': os.environ.get('EXIT_STATUS'),
              'production_signals_sent_by_hook': False,
              'accounting': 'No reservations settled here. Missing summary or open reservation requires root assessment.'}
    with (OUT / 'stop-record.json').open('x') as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write('\n')


if __name__ == '__main__' and sys.argv[1:] == ['stop']:
    stop_record()
    raise SystemExit(0)


from kenggo import progressive_training as original
from kenggo.checkpoints import json_bytes, save_json
from kenggo.continuation_data import path, read, reference
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import WandbSink

spec = importlib.util.spec_from_file_location('old_e2_recovery', OLD / 'recovery.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def size(filename):
    return original._output_bytes(filename) if filename.is_dir() else filename.stat().st_size


def unit_status(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState',
                             '-p', 'SubState', '-p', 'MainPID', '-p', 'ControlGroup'],
                            check=True, capture_output=True, text=True, timeout=15)
    return dict(line.split('=', 1) for line in result.stdout.splitlines())


def live_scope_pids(group):
    """Read current identities after a scope snapshot; exited/zombie PIDs cannot write."""
    if not group:
        return set()
    return {pid for pid in old.cgroup_pids(group)
            if (info := old.proc(pid)) is not None and info['state'] not in ('Z', 'X')}


def require_old_stopped():
    observations = {}
    for unit in OLD_UNITS:
        item = unit_status(unit)
        if item.get('ActiveState') not in ('inactive', 'failed') or item.get('MainPID') != '0':
            raise ValueError('Prior production/verification service is not stopped: ' + unit)
        if live_scope_pids(item.get('ControlGroup', '')):
            raise ValueError('Prior service still has live processes: ' + unit)
        observations[unit] = item
    config = load(OLD / 'recovery-config.json')
    if live_scope_pids(config['worker_cgroup']):
        raise ValueError('Original growth scope still has live processes')
    # A production writer escaped from its original scope must also block reuse.
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            arguments = entry.joinpath('cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if (any(x in (b'kenggo.progressive_jobs', b'kenggo.continuation_training',
                      b'kenggo.progressive_training') for x in arguments)
                and any(str(PROD).encode() in arg for arg in arguments)):
            raise ValueError('A prior production command is still running: ' + entry.name)
    return observations


def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def composite_module():
    return load_helper('e2_composite_validation', OUT / 'composite-validation.py')


def recovery_jobs():
    return load_helper('e2_validation_recovery_jobs', OUT / 'recovered_jobs.py')


def dispatch_job(command, *args, **kwargs):
    from kenggo.pilot_jobs import run_job
    if command[1:3] == ['-m', 'kenggo.progressive_jobs']:
        command = [command[0], str(OUT / 'recovered_jobs.py'), *command[3:]]
    return run_job(command, *args, **kwargs)


def preparation_charge(incident, end):
    start = datetime.fromisoformat(incident['observed_utc'])
    elapsed = (end - start).total_seconds()
    if not 0 <= elapsed <= PREPARATION_WALL:
        raise ValueError('Resume preparation exceeded its two-hour window')
    processors = os.cpu_count()
    if type(processors) is not int or processors < incident['logical_cpus']:
        raise ValueError('Host CPU count cannot conservatively bound preparation')
    return {'start_utc': start.isoformat(), 'end_utc': end.isoformat(),
            'elapsed_seconds': elapsed, 'host_logical_cpus': processors,
            'cpu_seconds_upper': math.ceil(elapsed * processors) + MARGIN_CPU,
            'method': 'All host logical CPUs times elapsed preparation, plus 600 CPU seconds for earlier probing and launch. Upper charge, not measured CPU.'}


def verify_manifest(filename, expected):
    if digest(filename) != expected:
        raise ValueError('Resume manifest changed')
    manifest = load(filename)
    if manifest.get('schema') != 'e2_validation_resume_manifest_v1' or manifest.get('source_commit') != COMMIT:
        raise ValueError('Unexpected resume manifest')
    for name, pin in manifest['files'].items():
        if digest(ROOT / name) != pin:
            raise ValueError('Pinned resume/input file changed: ' + name)
    old.verify_manifest(OLD / 'manifest-pass3.json', OLD_MANIFEST_SHA)
    old.source_frozen()
    return manifest


def initial_ledger(manifest):
    frozen = read(manifest['ledger_before'])
    actual = load(PROD / 'resources.json')
    if actual != frozen or frozen['reservations'] or frozen['budget_breached']:
        raise ValueError('Settled original ledger changed or has unresolved reservations')
    return frozen


def lock_once():
    handle = (OUT / 'resume.lock').open('ab')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'service-start.json').exists():
            raise ValueError('Resume already started; automatic restart is forbidden')
        return handle
    except BaseException:
        handle.close()
        raise


def immutable_request(filename, value, *, replace=False):
    filename = Path(filename)
    if filename == PROD / 'tier1-growth-request.json':
        if replace or filename.read_bytes() != json_bytes(value):
            raise ValueError('Original growth request must remain byte-identical')
        return digest(filename)
    return save_json(filename, value, replace=replace)


def check_prefix(state, before):
    if state['entries'][:len(before['entries'])] != before['entries']:
        raise ValueError('Prior ledger entries changed')
    if any(state[key] != before[key] for key in ('schema_version', 'budget_id', 'gpu_seconds')):
        raise ValueError('Original ledger identity changed')


class ResumedStudy(original.Study):
    def __init__(self, manifest_ref, manifest, prep):
        self.manifest_ref, self.manifest, self.preparation = manifest_ref, manifest, prep
        self.before = initial_ledger(manifest)
        self.session = read(manifest['session'])
        self.plan = read(self.session['plan'])
        self.output = PROD
        self.limits = self.session['limits']
        self.launch = datetime.fromisoformat(self.session['launch_utc'])
        self.deadline = datetime.fromisoformat(self.session['deadline_utc'])
        if (self.launch.isoformat() != '2026-11-21T14:13:03.148730+00:00'
                or self.deadline - self.launch != timedelta(seconds=604800)
                or self.limits != self.plan['parameters']['limits']
                or self.limits != {'cpu_seconds': 604800, 'gpu_seconds': 604800,
                    'wall_seconds': 604800, 'output_bytes': 160 * 1024**3, 'publication_seconds': 21600}):
            raise ValueError('The original study deadline or limits changed')
        self.production_end = self.deadline - timedelta(seconds=self.limits['publication_seconds'])
        self.started = time.monotonic() - (now() - self.launch).total_seconds()
        self.cpu_started, self.accounted_cpu, self.closed = time.process_time(), 0, False
        self.group = own_group()
        if not self.group.endswith('/' + UNIT):
            raise ValueError('Continuation must run in its reviewed durable service')
        self.ledger = BudgetLedger.open(PROD / 'resources.json', budget_id=self.plan['budget_id'],
                                        gpu_seconds=self.limits['gpu_seconds'])
        self.base_cpu = sum(e['cpu_seconds'] for e in self.before['entries'])
        self.prep_cpu = prep['cpu_seconds_upper'] + max(0, manifest['prior_cpu_upper'] - self.base_cpu)
        self.state = read(manifest['previous_summary'])
        if set(self.state['stages']) != {'tier0-readiness', 'tier1-sources', 'tier1-growth'} or self.state['status'] != 'failed':
            raise ValueError('Resume only supports the settled validation-coverage failure before training')
        if any(tier.get('training') for tier in self.state['tiers']):
            raise ValueError('This recovery cannot restart an already trained arm')
        self.state.update(status='running', reason=None, tiers=[], active_stage=None,
            continuation={'manifest': manifest_ref, 'previous_failed_summary': manifest['previous_summary'],
                'growth_receipt': manifest['growth_receipt'], 'growth_audit': manifest['growth_audit'],
                'source_phase': manifest['source_phase'], 'source_summary': manifest['source_summary'],
                'composite_request': manifest['composite_request'], 'preparation': prep,
                'accounting': 'Original allowance and all settled costs retained; new service descendants charged once.'})
        self.stop_event = threading.Event()
        self.guard_error = None
        self.phase = 'validating_saved_preparation_inputs'
        self.finalizing = False

    def output_bytes(self):
        total = self.manifest['sealed_output_bytes']
        sealed = set(self.manifest['sealed_production_names'])
        total += sum(size(p) for p in PROD.iterdir() if p.name not in sealed)
        return total + size(OUT) + (size(VERIFY) if VERIFY.exists() else 0)

    def costs(self):
        ledger = self.ledger.snapshot()
        check_prefix(ledger['state'], self.before)
        service = group_cpu(self.group)
        measured_upper = self.base_cpu + self.prep_cpu + service
        recorded = sum(e['cpu_seconds'] for e in ledger['state']['entries'])
        return {'elapsed_seconds': max(time.monotonic() - self.started, (now() - self.launch).total_seconds()),
                'cpu_seconds': max(measured_upper, recorded) + (0 if self.closed else MARGIN_CPU),
                'service_cpu_seconds': service, 'preparation_cpu_upper_seconds': self.prep_cpu,
                'output_bytes': self.output_bytes(), 'ledger': ledger}

    def check(self):
        if self.guard_error:
            raise original.StopStudy(self.guard_error)
        costs = self.costs()
        if (costs['elapsed_seconds'] >= self.limits['wall_seconds'] - 120
                or costs['cpu_seconds'] >= self.limits['cpu_seconds'] - MARGIN_CPU
                or costs['output_bytes'] >= self.limits['output_bytes'] - 256 * 1024**2
                or costs['ledger']['budget_breached']):
            raise original.StopStudy('Original shared resource boundary reached')
        return costs

    def persist(self):
        self.state['costs'] = self.costs()
        save_json(OUT / 'progress.json', self.state, replace=True)

    def heartbeat(self):
        while not self.stop_event.wait(5):
            try:
                costs = self.check()
                save_json(OUT / 'heartbeat.json', {'utc': now().isoformat(), 'pid': os.getpid(),
                    'cgroup': self.group, 'phase': self.phase, 'deadline_utc': self.deadline.isoformat(),
                    'active_stage': self.state.get('active_stage'),
                    'costs': {key: value for key, value in costs.items() if key != 'ledger'}}, replace=True)
            except BaseException as error:
                if self.stop_event.is_set():
                    return
                self.guard_error = type(error).__name__ + ': ' + str(error)
                save_json(OUT / 'guard-stop.json', {'utc': now().isoformat(), 'reason': self.guard_error})
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def account_preparation(self):
        self.check()
        if self.ledger.snapshot()['state'] != self.before:
            raise ValueError('Ledger changed before resume preparation settlement')
        self.ledger.reserve(PREP_ID, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                            metadata={'manifest': self.manifest_ref, 'preparation': self.preparation,
                                      'prior_cpu_rounding_upper': self.prep_cpu - self.preparation['cpu_seconds_upper']})
        self.ledger.record(PREP_ID, allocations=[], outcome='completed', arm_shares={'shared': 1},
                           cpu_seconds=self.prep_cpu, wall_seconds=0)
        self.ledger.reserve(SUPERVISOR_ID, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                            metadata={'purpose': 'Resume service and descendants not already charged by jobs',
                                      'manifest': self.manifest_ref})

    def coverage(self, tier):
        helper = composite_module()
        previous = [item['preparation'] for item in self.state['tiers']
                    if item['tier'] < tier and item.get('preparation')]
        try:
            return helper.preflight(path(self.manifest['composite_request']),
                self.manifest['composite_request']['sha256'], tier=tier, previous=previous)
        except ValueError as error:
            raise original.StopStudy('Fresh stopping-validation coverage is insufficient: ' + str(error)) from error

    def stage(self, name, config, **options):
        self.check()
        verify_manifest(path(self.manifest_ref), self.manifest_ref['sha256'])
        if name in self.state['stages']:
            stage = self.state['stages'][name]
            old.matching_stage(read(stage['config']), config, options.get('output_bytes', 2 * 1024**3))
            read(stage['provenance'])
            original.require_completed(stage)
            return stage, read(stage['summary'])
        if config.get('purpose') == 'progressive-sources':
            self.coverage(config['parameters']['tier'])
        physical = name + '-validation-recovery' if config.get('purpose') == 'progressive-preparation' else name
        if (PROD / physical).exists():
            raise ValueError('An existing unregistered stage prevents continuation: ' + physical)
        reservations = set(self.ledger.snapshot()['state']['reservations']) - {SUPERVISOR_ID}
        if reservations:
            raise ValueError('Unresolved reservations block new work: ' + ', '.join(sorted(reservations)))
        old.gpu_idle()
        self.phase = physical
        config = deepcopy(config)
        config['inputs'].update(resume_manifest=self.manifest_ref,
            composite_request=self.manifest['composite_request'],
            validation_supplement=self.manifest['source_summary'])
        return super().stage(physical, config, **options)

    def validate_saved_start(self):
        receipt, audit = read(self.manifest['growth_receipt']), read(self.manifest['growth_audit'])
        stage = self.state['stages']['tier1-growth']
        summary = read(stage['summary'])
        if (receipt['status'] != 'validated_saved_artifacts' or audit['status'] != 'passed'
                or receipt['worker_summary'] != stage['summary']
                or audit['dataset']['dataset'] != summary['dataset_summary']
                or audit['dataset']['rows'] != 3999997 or audit['dataset']['games'] != 18126):
            raise ValueError('Saved growth audit or accepted receipt changed')
        from_module = recovery_jobs()
        from_module.assert_fitting_disjoint(summary['dataset_summary'], self.manifest['source_summary'])
        self.coverage(1)
        save_json(OUT / 'saved-inputs-verified.json', {'status': 'verified_saved_pins_and_coverage',
            'growth_receipt': self.manifest['growth_receipt'], 'growth_audit': self.manifest['growth_audit'],
            'source_summary': self.manifest['source_summary'], 'composite_request': self.manifest['composite_request'],
            'original_board_audit_repeated': False, 'training_performed': False, 'utc': now().isoformat()})
        self.persist()

    def finish(self, status, reason):
        self.finalizing = True
        self.phase = 'finalizing'
        self.state.update(status=status, reason=reason, active_stage=None)
        unresolved = set(self.ledger.snapshot()['state']['reservations']) - {SUPERVISOR_ID}
        if unresolved:
            self.persist()
            raise RuntimeError('Unsettled job reservations retained; final accounting is unavailable')
        old.gpu_idle()
        self.persist()
        provisional = OUT / 'pre-export-summary.json'
        save_json(provisional, self.state)
        from kenggo.progressive_review import export
        try:
            export(reference(provisional), OUT / 'review.html')
        except BaseException as error:
            save_json(OUT / 'export-failure.json', {'type': type(error).__name__, 'reason': str(error)})
            self.state['export_failure'] = reference(OUT / 'export-failure.json')
        entries = self.ledger.snapshot()['state']['entries'][len(self.before['entries']):]
        new_job_cpu = sum(e['cpu_seconds'] for e in entries if e['run_id'] not in (PREP_ID, SUPERVISOR_ID))
        overhead = max(0, group_cpu(self.group) - new_job_cpu) + MARGIN_CPU
        self.ledger.record(SUPERVISOR_ID, allocations=[], outcome='completed' if status == 'completed' else 'partial',
                           arm_shares={'shared': 1}, cpu_seconds=overhead, wall_seconds=0,
                           reason=None if status == 'completed' else reason)
        self.closed = True
        self.persist()
        save_json(OUT / 'summary.json', self.state)
        return self.state


def freeze(filename):
    old.source_frozen()
    require_old_stopped()
    old.gpu_idle()
    preparation_charge(load(OUT / 'incident.json'), now())
    before = load(OUT / 'before-resources.json')
    if load(PROD / 'resources.json') != before or before['reservations'] or before['budget_breached']:
        raise ValueError('Original ledger changed or remains unsettled')
    phase = load(SOURCE_PHASE / 'phase-summary.json')
    source_summary = SOURCE_PHASE / 'production/worker/source-work/summary.json'
    if phase['status'] != 'completed' or load(source_summary)['usable_games'] < 2:
        raise ValueError('Source supplement is not complete and sufficient')
    files = dict(load(SOURCE_PHASE / 'phase-manifest.json')['files'])
    for directory in (SOURCE_PHASE, PREVIOUS, PREVIOUS_VERIFY):
        for item in directory.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                files[str(item.relative_to(ROOT))] = digest(item)
    for directory in (OUT, VERIFY):
        for item in directory.iterdir():
            if item.is_file() and item.name not in ('manifest.json', 'review-approved.json') and not item.name.startswith('.'):
                files[str(item.relative_to(ROOT))] = digest(item)
    for item in (SOURCE_PHASE / 'production/worker/source-work').iterdir():
        if item.is_file():
            files[str(item.relative_to(ROOT))] = digest(item)
    for item in (PROD / 'tier1-preparation').rglob('*'):
        if item.is_file():
            files[str(item.relative_to(ROOT))] = digest(item)
    sealed = sorted(p.name for p in PROD.iterdir() if p.name != 'resources.json')
    sealed_size = sum(size(PROD / name) for name in sealed)
    other = [p for p in (ROOT / 'runs').glob('e2-progressive*') if p not in (OUT, VERIFY, PROD)]
    sealed_size += sum(size(p) for p in other) + size(SOURCE_PHASE)
    manifest = {'schema': 'e2_validation_resume_manifest_v1', 'source_commit': COMMIT,
        'session': reference(PROD / 'session.json'), 'ledger_before': reference(OUT / 'before-resources.json'),
        'previous_summary': reference(PREVIOUS / 'summary.json'),
        'growth_receipt': reference(PREVIOUS / 'growth-recovered.json'),
        'growth_audit': reference(PREVIOUS / 'audit-saved-growth.json'),
        'source_phase': reference(SOURCE_PHASE / 'phase-summary.json'), 'source_summary': reference(source_summary),
        'composite_request': reference(OUT / 'composite-request.json'),
        'prior_cpu_upper': load(OUT / 'incident.json')['prior_cpu_upper_seconds'],
        'sealed_production_names': sealed, 'sealed_output_bytes': sealed_size,
        'files': dict(sorted(files.items()))}
    save_json(filename, manifest)
    print(json.dumps(reference(filename)))


def preflight(filename, expected, *, review=False):
    manifest = verify_manifest(filename, expected)
    require_old_stopped()
    initial_ledger(manifest)
    old.gpu_idle()
    prep = preparation_charge(load(OUT / 'incident.json'), now())
    if (OUT / 'service-start.json').exists():
        raise ValueError('Resume already started')
    if review:
        verdict = load(OUT / 'review-approved.json')
        if verdict.get('manifest_sha256') != expected or verdict.get('verdict') != 'no significant findings':
            raise ValueError('Fresh independent review of these exact files is required')
    return manifest, prep


def run(filename, expected):
    manifest, _ = preflight(filename, expected, review=True)
    lock = lock_once()
    study = ResumedStudy(reference(filename), manifest,
                         preparation_charge(load(OUT / 'incident.json'), STARTED_UTC))
    study.check()
    save_json(OUT / 'service-start.json', {'utc': STARTED_UTC.isoformat(), 'pid': os.getpid(),
        'start_ticks': old.proc(os.getpid())['start'], 'cgroup': study.group,
        'deadline_utc': study.deadline.isoformat(), 'manifest': reference(filename),
        'preparation': study.preparation})
    study.account_preparation()
    monitor = threading.Thread(target=study.heartbeat, daemon=True)
    def terminated(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        study.guard_error = study.guard_error or 'Resume service termination requested'
        study.stop_event.set()
        raise original.StopStudy(study.guard_error)
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    monitor.start()
    status, reason = 'failed', 'Resume did not finish'
    old_save, old_runner = original.save_json, original.run_job
    try:
        study.validate_saved_start()
        original.save_json = immutable_request
        original.run_job = dispatch_job
        reason = original.execute(study)
        status = 'completed'
    except original.StopStudy as error:
        status, reason = 'budget_or_evidence_stop', str(error)
    except BaseException as error:
        reason = type(error).__name__ + ': ' + str(error)
        save_json(OUT / 'resume-failure.json', {'type': type(error).__name__, 'reason': str(error)})
    finally:
        original.save_json, original.run_job = old_save, old_runner
        try:
            study.finish(status, reason)
        finally:
            study.stop_event.set()
            monitor.join(timeout=6)
            lock.close()
    return 1 if status == 'failed' else 0


def launch(filename, expected):
    filename = Path(filename).resolve()
    manifest, prep = preflight(filename, expected, review=True)
    if (OUT / 'launch-command.json').exists():
        raise ValueError('A launch was already attempted; no automatic retry')
    session = read(manifest['session'])
    remaining = (datetime.fromisoformat(session['deadline_utc']) - now()).total_seconds() - 120
    if remaining <= 120 or remaining > 604800:
        raise ValueError('Original deadline cannot support continuation')
    command = ['systemd-run', '--user', '--unit=' + UNIT, '--property=Type=exec',
        '--property=Restart=no', '--property=CPUAccounting=yes', '--property=KillMode=control-group',
        '--property=TimeoutStopSec=30s', '--property=RuntimeMaxSec=' + str(int(remaining)) + 's',
        '--property=StandardOutput=append:' + str(OUT / 'service.stdout.log'),
        '--property=StandardError=append:' + str(OUT / 'service.stderr.log'),
        '--property=ExecStopPost=' + ' '.join((str(ROOT / '.venv/bin/python'), str(OUT / 'resume.py'), 'stop')),
        '--working-directory=' + str(ROOT), '--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=UV_OFFLINE=1', '--setenv=CUDA_VISIBLE_DEVICES=',
        str(ROOT / '.venv/bin/python'), str(OUT / 'resume.py'), 'run', str(filename), expected]
    save_json(OUT / 'launch-command.json', {'utc': now().isoformat(), 'command': command,
        'remaining_runtime_seconds': int(remaining), 'preparation_upper_at_launch': prep})
    subprocess.run(command, check=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'check', 'run', 'launch'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('sha256', nargs='?')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.manifest)
    elif not args.sha256:
        parser.error('The exact manifest SHA256 is required')
    elif args.mode == 'check':
        preflight(args.manifest, args.sha256)
        print('Frozen source, pins, original stopped services and settled ledger verified')
    elif args.mode == 'launch':
        launch(args.manifest, args.sha256)
    else:
        raise SystemExit(run(args.manifest, args.sha256))


if __name__ == '__main__':
    main()
