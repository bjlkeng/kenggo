"""Run one reviewed validation-source check against the original study allowance."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path('/home/bojji/code/kenggo')
OUT = ROOT / 'runs/e2-validation-gap-2026-11-24'
PROD = ROOT / 'runs/e2-progressive-2026-11-21'
UNIT = 'kenggo-e2-validation-source-20261124.service'
HEAD = '59d29abf57d8c03eec783a30222fa288403c8e7b'
DEADLINE = datetime.fromisoformat('2026-11-28T14:13:03.148730+00:00')
PREP_ID = 'e2-validation-source-preparation-20261124-v1'
JOB_ID = 'e2-validation-source-20261124-v1'
SUPERVISOR_ID = 'e2-validation-source-supervisor-20261124-v1'
OLD_UNITS = ('kenggo-e2-progressive-recovery-20261121.service', 'kenggo-e2-final-verification-20261121.service',
             'kenggo-e2-progressive-resume-20261124.service', 'kenggo-e2-progressive-resume-verification-20261124.service')
START = datetime.now(timezone.utc)


def now():
    return datetime.now(timezone.utc)


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ref(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p)}


def write(name, obj, *, replace=False):
    target = OUT / name
    if target.exists() and not replace:
        raise FileExistsError(target)
    temporary = target.with_name('.' + target.name + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, target)


def group():
    return Path('/proc/self/cgroup').read_text().strip().removeprefix('0::')


def cpu():
    fields = dict(line.split() for line in (Path('/sys/fs/cgroup') / group().lstrip('/') / 'cpu.stat').read_text().splitlines())
    return int(fields['usage_usec']) / 1e6


def stop_record():
    write('stop-record.json', {'utc': now().isoformat(), 'cpu_seconds': cpu(), 'cgroup': group(),
          'service_result': os.environ.get('SERVICE_RESULT'), 'exit_status': os.environ.get('EXIT_STATUS'),
          'note': 'This hook never settles outstanding reservations or signals prior services.'})


if __name__ == '__main__' and sys.argv[1:] == ['stop']:
    stop_record()
    raise SystemExit(0)


from kenggo.pilot_jobs import JobLimits, run_job, _output_bytes
from kenggo.resource_ledger import BudgetLedger
from kenggo.tracking import Experiment, capture_provenance


def state(unit):
    text = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID', '-p', 'ControlGroup'], text=True, timeout=15)
    return dict(line.split('=', 1) for line in text.splitlines())


def source():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=15).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True, timeout=15).strip()
    if head != HEAD or dirty:
        raise ValueError('Frozen source changed')


def verify(expected):
    if sha(OUT / 'phase-manifest.json') != expected:
        raise ValueError('Phase manifest changed')
    manifest = load(OUT / 'phase-manifest.json')
    for name, value in manifest['files'].items():
        if sha(ROOT / name) != value:
            raise ValueError('Phase file changed: ' + name)
    source()
    return manifest


def old_stopped():
    for unit in OLD_UNITS:
        s = state(unit)
        if s['ActiveState'] not in ('failed', 'inactive') or s['MainPID'] != '0':
            raise ValueError('Prior service is still active: ' + unit)
        cg = s.get('ControlGroup') or '/user.slice/user-1001.slice/user@1001.service/app.slice/' + unit
        directory = Path('/sys/fs/cgroup') / cg.lstrip('/')
        if directory.exists() and any(p.read_text().strip() for p in directory.rglob('cgroup.procs')):
            raise ValueError('Prior service still has processes: ' + unit)
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(a in (b'kenggo.progressive_training', b'kenggo.progressive_jobs', b'kenggo.continuation_training') for a in args) and any(str(PROD).encode() in a for a in args):
            raise ValueError('Prior production worker remains')


def preparation(incident, end):
    elapsed = (end - datetime.fromisoformat(incident['observed_utc'])).total_seconds()
    if not 0 <= elapsed <= 7200:
        raise ValueError('Two-hour source preparation window expired')
    processors = os.cpu_count()
    if processors is None or processors < incident['logical_cpus']:
        raise ValueError('Cannot bound preparation CPU')
    return {'cpu_seconds_upper': math.ceil(elapsed * processors) + 600, 'wall_seconds': elapsed,
            'method': 'All host logical CPUs times preparation wall time plus600CPU seconds; conservative charge, not measured CPU'}


def over_limit(usage, elapsed, size, utc):
    return usage >= 1740 or elapsed >= 3540 or size >= 1024**3 or utc >= DEADLINE - timedelta(seconds=120)


def preflight(expected, *, approved=True):
    manifest = verify(expected)
    old_stopped()
    before = load(OUT / 'before-resources.json')
    if load(PROD / 'resources.json') != before or before['reservations'] or before['budget_breached']:
        raise ValueError('Original settled ledger changed or remains reserved')
    if (OUT / 'service-start.json').exists():
        raise ValueError('Source phase already started; automatic retry forbidden')
    prep = preparation(load(OUT / 'incident.json'), now())
    if load(OUT / 'incident.json')['prior_cpu_upper_seconds'] + prep['cpu_seconds_upper'] + 2400 > 604800:
        raise ValueError('Original CPU budget cannot fund source check')
    if (DEADLINE - now()).total_seconds() < 3600 + 21600:
        raise ValueError('Original remaining time cannot fund source check and final reserve')
    if manifest['sealed_output_bytes'] + _output_bytes(OUT) + 1024**3 + 8 * 1024**3 > 160 * 1024**3:
        raise ValueError('Original output budget cannot fund source check')
    if approved:
        verdict = load(OUT / 'review-approved.json')
        if verdict.get('manifest_sha256') != expected or verdict.get('verdict') != 'no significant findings':
            raise ValueError('Fresh independent review must approve this exact manifest')
    return manifest, before, prep


def run(expected, credentials_file):
    import fcntl
    manifest, before, _ = preflight(expected)
    if not group().endswith('/' + UNIT):
        raise ValueError('Source phase requires its durable service')
    lock = (OUT / 'phase.lock').open('ab')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'service-start.json').exists():
        raise ValueError('Source phase already started')
    prep = preparation(load(OUT / 'incident.json'), START)
    old_cpu = sum(e['cpu_seconds'] for e in before['entries'])
    prep_charge = prep['cpu_seconds_upper'] + max(0, load(OUT / 'incident.json')['prior_cpu_upper_seconds'] - old_cpu)
    ledger = BudgetLedger.open(PROD / 'resources.json', budget_id=before['budget_id'], gpu_seconds=before['gpu_seconds'])
    write('service-start.json', {'utc': START.isoformat(), 'pid': os.getpid(), 'cgroup': group(),
          'manifest': ref(OUT / 'phase-manifest.json'), 'original_deadline_utc': DEADLINE.isoformat(), 'preparation': prep})
    ledger.reserve(PREP_ID, estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={'manifest': ref(OUT / 'phase-manifest.json'), 'preparation': prep})
    ledger.record(PREP_ID, allocations=[], outcome='completed', arm_shares={'shared': 1}, cpu_seconds=prep_charge, wall_seconds=0)
    ledger.reserve(SUPERVISOR_ID, estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={'purpose': 'Source supervisor and tracking not already charged by source job; includes final margin'})
    stop = threading.Event()
    phase_start = time.monotonic()
    outcome, reason, job = 'failed', 'Source phase did not finish', None

    def terminated(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop.set()
        raise InterruptedError('Source phase resource guard or service termination')

    def watchdog():
        while not stop.wait(5):
            try:
                usage, elapsed, size = cpu(), time.monotonic() - phase_start, _output_bytes(OUT)
                write('heartbeat.json', {'utc': now().isoformat(), 'status': 'running', 'cpu_seconds': usage, 'phase_wall_seconds': elapsed,
                      'global_elapsed_seconds': (now() - datetime.fromisoformat(load(OUT / 'incident.json')['original_launch_utc'])).total_seconds(),
                      'output_bytes': size}, replace=True)
                if over_limit(usage, elapsed, size, now()):
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
            except BaseException as error:
                write('guard-failure.json', {'reason': type(error).__name__ + ': ' + str(error)})
                os.kill(os.getpid(), signal.SIGTERM)
                return

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    monitor = threading.Thread(target=watchdog, daemon=True)
    monitor.start()
    try:
        config = load(OUT / 'phase-config.json')
        provenance, patch = capture_provenance(ROOT, config['inputs'])
        with Experiment(OUT / 'production', config, provenance, patch) as experiment:
            request = manifest['request']
            command = [str(ROOT / '.venv/bin/python'), str(OUT / 'source-helper.py'), 'run', request['path'], request['sha256'],
                       str(OUT / 'production/worker'), '--credentials-file', credentials_file]
            job = run_job(command, OUT / 'production/worker', cwd=ROOT, limits=JobLimits(3300, 1500, 1024**3),
                          ledger=ledger, run_id=JOB_ID, arm_shares={'shared': 1}, device=None,
                          environment=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                          metadata={'request': request, 'phase_manifest': ref(OUT / 'phase-manifest.json')})
            saved = OUT / 'production/worker/summary.json'
            source_summary = load(saved) if saved.exists() else {}
            outcome = 'completed' if job['status'] == source_summary.get('status') == 'completed' else 'partial'
            reason = None if outcome == 'completed' else source_summary.get('reason') or job.get('reason') or 'No complete source summary'
            experiment.log(1, {'source/job_status': job['status'], 'cost/cpu_seconds': job['costs']['cpu_seconds'],
                               'cost/wall_seconds': job['costs']['wall_seconds'], 'source/usable_games': source_summary.get('usable_games', 0)})
            experiment.finish(outcome, reason=reason, summary={'source_summary': ref(saved) if saved.exists() else None})
    except BaseException as error:
        outcome = 'failed'
        reason = type(error).__name__ + ': ' + str(error)
        write('phase-failure.json', {'reason': reason})
    finally:
        try:
            pending = set(ledger.snapshot()['state']['reservations']) - {SUPERVISOR_ID}
            if pending:
                raise RuntimeError('Unknown source allocation retained; cannot finalize: ' + ', '.join(sorted(pending)))
            charged_job_cpu = sum(e['cpu_seconds'] for e in ledger.snapshot()['state']['entries'][len(before['entries']):]
                                 if e['run_id'] not in (PREP_ID, SUPERVISOR_ID))
            overhead = max(0, cpu() - charged_job_cpu) + 600
            ledger.record(SUPERVISOR_ID, allocations=[], outcome=outcome, arm_shares={'shared': 1},
                          cpu_seconds=overhead, wall_seconds=0, reason=reason)
            final = ledger.snapshot()
            if final['state']['entries'][:len(before['entries'])] != before['entries']:
                raise ValueError('Prior cost entries changed')
            write('phase-summary.json', {'status': outcome, 'reason': reason, 'job': job,
                  'source_summary': ref(OUT / 'production/worker/summary.json') if (OUT / 'production/worker/summary.json').exists() else None,
                  'preparation': prep, 'measured_service_cpu_seconds': cpu(), 'phase_wall_seconds': time.monotonic() - phase_start,
                  'cumulative_cpu_seconds': sum(e['cpu_seconds'] for e in final['state']['entries']), 'gpu_seconds_added': 0,
                  'original_deadline_utc': DEADLINE.isoformat(), 'ledger': final,
                  'scope': 'Validation-source feasibility only; no training or new allowance'})
        finally:
            stop.set()
            monitor.join(timeout=6)
            lock.close()
    return 0 if outcome == 'completed' else 1


def launch(expected, credentials_file):
    preflight(expected)
    if (OUT / 'launch-command.json').exists():
        raise ValueError('Launch already attempted; automatic retry forbidden')
    argv = ['systemd-run', '--user', '--unit=' + UNIT, '--property=Type=exec', '--property=Restart=no',
            '--property=CPUAccounting=yes', '--property=KillMode=control-group', '--property=TimeoutStopSec=30s',
            '--property=RuntimeMaxSec=3600s', '--property=LimitCORE=0',
            '--property=StandardOutput=append:' + str(OUT / 'service.stdout.log'), '--property=StandardError=append:' + str(OUT / 'service.stderr.log'),
            '--property=ExecStopPost=' + str(ROOT / '.venv/bin/python') + ' ' + str(OUT / 'phase-runner.py') + ' stop',
            '--working-directory=' + str(ROOT), '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=CUDA_VISIBLE_DEVICES=',
            '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1', '--setenv=UV_OFFLINE=1',
            str(ROOT / '.venv/bin/python'), str(OUT / 'phase-runner.py'), 'run', expected, credentials_file]
    write('launch-command.json', {'utc': now().isoformat(), 'command': argv})
    subprocess.run(argv, cwd=ROOT, check=True, timeout=30)


if __name__ == '__main__':
    mode, expected, *extra = sys.argv[1:]
    if mode == 'check':
        preflight(expected, approved=False)
        print('Original source, stopped services, ledger and phase pins checked')
    elif mode == 'launch':
        launch(expected, extra[0])
    elif mode == 'run':
        raise SystemExit(run(expected, extra[0]))
    else:
        raise ValueError('Unknown phase operation')
