"""Wait for resumed production, then verify saved evidence within its original allowance."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/bojji/code/kenggo')
OUT = ROOT / 'runs/e2-progressive-resume-verification-2026-11-24'
RESUME = ROOT / 'runs/e2-progressive-resume-2026-11-24'
PROD = ROOT / 'runs/e2-progressive-2026-11-21'
PRODUCTION_UNIT = 'kenggo-e2-progressive-resume-20261124.service'
UNIT = 'kenggo-e2-progressive-resume-verification-20261124.service'
LAUNCH = datetime.fromisoformat('2026-11-21T14:13:03.148730+00:00')
DEADLINE = datetime.fromisoformat('2026-11-28T14:13:03.148730+00:00')
HEAD = '59d29abf57d8c03eec783a30222fa288403c8e7b'
CPU_CAP, WALL_CAP, MARGIN = 1800, 3600, 600
CHILD = None


def now():
    return datetime.now(timezone.utc)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': digest(path)}


def write(name, record):
    target = OUT / name
    temporary = target.with_name('.' + target.name + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, target)


def source():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=15).strip()
    state = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True, timeout=15).strip()
    if head != HEAD or state:
        raise ValueError('Frozen production source changed')
    return head


def validate_pins(expected=None):
    expected = expected or os.environ['E2_VERIFY_MANIFEST_SHA']
    if digest(OUT / 'manifest.json') != expected:
        raise ValueError('Verification manifest changed')
    manifest = load(OUT / 'manifest.json')
    for filename, pin in manifest['files'].items():
        if digest(ROOT / filename) != pin:
            raise ValueError('Verification pin changed: ' + filename)
    return {'manifest': ref(OUT / 'manifest.json'), 'files': len(manifest['files'])}


def unit_state(unit):
    output = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID',
                                      '-p', 'ControlGroup', '-p', 'Result'], text=True, timeout=15)
    return dict(line.split('=', 1) for line in output.splitlines())


def own_group():
    return Path('/proc/self/cgroup').read_text().strip().removeprefix('0::')


def stop_record():
    write('stop-record.json', {'utc': now().isoformat(), 'cgroup': own_group(),
          'service_cpu_seconds': cpu(own_group()), 'service_result': os.environ.get('SERVICE_RESULT'),
          'exit_code': os.environ.get('EXIT_CODE'), 'exit_status': os.environ.get('EXIT_STATUS'),
          'production_signalled': False, 'costs_settled': (OUT / 'final-resources.json').exists()})


def cpu(group):
    fields = dict(line.split() for line in (Path('/sys/fs/cgroup') / group.lstrip('/') / 'cpu.stat').read_text().splitlines())
    return int(fields['usage_usec']) / 1e6


def size(directory):
    total = 0
    for base, _, files in os.walk(directory, followlinks=False):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def stopped():
    state = unit_state(PRODUCTION_UNIT)
    if state.get('ActiveState') not in ('failed', 'inactive') or int(state.get('MainPID', '0')):
        return False
    group = Path('/sys/fs/cgroup') / own_group().lstrip('/')
    group = group.parent / PRODUCTION_UNIT
    if group.exists() and any(p.read_text().strip() for p in group.rglob('cgroup.procs')):
        return False
    modules = (b'kenggo.progressive_training', b'kenggo.progressive_jobs', b'kenggo.continuation_training')
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            command = (entry / 'cmdline').read_bytes()
            if str(PROD).encode() in command and any(module in command for module in modules):
                return False
        except FileNotFoundError:
            pass
    return True


def settled(summary, ledger):
    if not summary or summary.get('status') in (None, 'running'):
        raise ValueError('No final resumed production summary')
    if ledger['reservations']:
        raise ValueError('Production reservations remain unsettled')
    if summary['costs']['ledger']['state'] != ledger:
        raise ValueError('Summary and settled ledger differ')
    if datetime.fromisoformat(summary['deadline_utc']) != DEADLINE:
        raise ValueError('Original deadline changed')
    return max(summary['costs']['cpu_seconds'], sum(e['cpu_seconds'] for e in ledger['entries']))


def budget_reason(usage, active, production_cpu=0, production_bytes=0, output_bytes=0):
    if usage >= CPU_CAP - 30 or active >= WALL_CAP - 30:
        return 'Verification CPU or active wall limit reached'
    if (DEADLINE - now()).total_seconds() <= 120:
        return 'Original deadline reached'
    if production_cpu + usage + MARGIN + 30 >= 604800:
        return 'Original CPU allowance cannot fund verification'
    if output_bytes >= 1000 * 1024**2 or production_bytes + output_bytes >= 160 * 1024**3 - 16 * 1024**2:
        return 'Original or verification output allowance reached'
    return None


def stop_child():
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        try:
            os.killpg(CHILD.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            CHILD.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(CHILD.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            CHILD.wait()


def interrupted(signum, frame):
    stop_child()
    raise InterruptedError('Verification interrupted; production was not signalled')


def run_helper(name, argv, group, active_start, production_cpu, production_bytes):
    global CHILD
    start = time.monotonic()
    reason = budget_reason(cpu(group), start - active_start, production_cpu, production_bytes, size(OUT))
    if reason:
        raise ValueError(reason)
    with (OUT / (name + '.stdout.log')).open('xb') as stdout, (OUT / (name + '.stderr.log')).open('xb') as stderr:
        CHILD = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
        while CHILD.poll() is None:
            reason = budget_reason(cpu(group), time.monotonic() - active_start, production_cpu, production_bytes, size(OUT))
            if reason:
                stop_child()
                break
            time.sleep(1)
        result = {'returncode': CHILD.returncode, 'reason': reason, 'wall_seconds': time.monotonic() - start,
                  'service_cpu_seconds': cpu(group)}
    CHILD = None
    write(name + '-attempt.json', result)
    return result


def account(group, active_start, initial, summary):
    from kenggo.resource_ledger import BudgetLedger
    if not stopped() or load(PROD / 'resources.json') != initial:
        raise ValueError('Production or ledger changed before verification settlement')
    usage = cpu(group)
    upper = settled(summary, initial)
    active = time.monotonic() - active_start
    reason = budget_reason(usage, active, upper, summary['costs']['output_bytes'], size(OUT))
    if reason:
        raise ValueError(reason)
    ledger = BudgetLedger.open(PROD / 'resources.json', budget_id=initial['budget_id'], gpu_seconds=initial['gpu_seconds'])
    passed = ((OUT / 'verification.json').exists() and load(OUT / 'verification.json').get('status') == 'saved_evidence_checks_finished'
              and (OUT / 'browser-evidence.json').exists() and load(OUT / 'browser-evidence.json').get('status') == 'passed')
    entry = 'e2-resume-verification-20261124-v1'
    ledger.reserve(entry, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                   metadata={'purpose': 'Saved-evidence verification service, descendants and final margin; preparation already charged by resumed production',
                             'manifest': ref(OUT / 'manifest.json')})
    ledger.record(entry, allocations=[], outcome='completed' if passed else 'partial', arm_shares={'shared': 1},
                  cpu_seconds=usage + MARGIN, wall_seconds=active,
                  reason=None if passed else 'One or more saved-evidence or browser checks failed or were unavailable')
    final = ledger.snapshot()
    if final['state']['entries'][:-1] != initial['entries']:
        raise ValueError('Prior ledger entries changed')
    write('final-resources.json', {'schema': 'e2_resumed_final_verification_costs_v1',
          'measured_service_cpu_seconds': usage, 'verification_cpu_upper_seconds': usage + MARGIN,
          'preparation_charged_again': False, 'gpu_seconds': 0, 'active_wall_seconds': active,
          'global_elapsed_seconds': (now() - LAUNCH).total_seconds(),
          'cumulative_cpu_upper_seconds': upper + usage + MARGIN,
          'cumulative_gpu_seconds': final['charged_gpu_seconds'], 'verification_output_bytes': size(OUT),
          'ledger': ref(PROD / 'resources.json'), 'production_summary': ref(RESUME / 'summary.json')})


def run():
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    source()
    validate_pins()
    group = own_group()
    if not group.endswith('/' + UNIT):
        raise ValueError('Verifier must run in its own durable service')
    write('service-start.json', {'utc': now().isoformat(), 'pid': os.getpid(), 'cgroup': group})
    active_start = None
    while True:
        reason = budget_reason(cpu(group), 0, output_bytes=size(OUT))
        if reason:
            raise ValueError(reason)
        write('status.json', {'status': 'waiting', 'utc': now().isoformat(), 'cpu_seconds': cpu(group),
                              'reason': 'Waiting for resumed production and every worker to stop'})
        if stopped():
            time.sleep(5)
            if stopped():
                break
        time.sleep(60)
    active_start = time.monotonic()
    initial = load(PROD / 'resources.json')
    summary = load(RESUME / 'summary.json')
    production_cpu = settled(summary, initial)
    write('production-resources.json', initial)
    write('production-summary.json', summary)
    validate_pins()
    write('status.json', {'status': 'verifying', 'utc': now().isoformat()})
    production_bytes = summary['costs']['output_bytes']
    run_helper('saved-evidence', [str(ROOT / '.venv/bin/python'), str(OUT / 'checks.py')],
               group, active_start, production_cpu, production_bytes)
    if (OUT / 'review.html').exists():
        run_helper('browser', ['/home/bojji/.nvm/versions/node/v25.9.0/bin/node', str(OUT / 'browser.cjs'), str(OUT / 'review.html')],
                   group, active_start, production_cpu, production_bytes)
    source()
    validate_pins()
    account(group, active_start, initial, summary)
    result = {'status': 'finished_requires_root_acceptance', 'utc': now().isoformat(),
              'production_status': summary['status'], 'production_reason': summary.get('reason'),
              'checks': ref(OUT / 'verification.json') if (OUT / 'verification.json').exists() else None,
              'browser': ref(OUT / 'browser-evidence.json') if (OUT / 'browser-evidence.json').exists() else None,
              'owner_page': ref(OUT / 'review.html') if (OUT / 'review.html').exists() else None,
              'costs': ref(OUT / 'final-resources.json'),
              'root_work': 'Assess failures and missing checks, obtain fresh independent completion review, deliver reports/source/Beads; E2 stays open.'}
    write('handoff.json', result)
    write('status.json', result)


def launch(expected):
    source()
    validate_pins(expected)
    if (OUT / 'launch-record.json').exists() or unit_state(UNIT).get('ActiveState') == 'active':
        raise ValueError('Verifier already launched; refusing duplicate')
    if unit_state(PRODUCTION_UNIT).get('ActiveState') != 'active':
        raise ValueError('Start verifier only after resumed production is active')
    remaining = (DEADLINE - now()).total_seconds()
    if remaining < 3600:
        raise ValueError('Original deadline leaves insufficient verification reserve')
    argv = ['systemd-run', '--user', '--unit=' + UNIT, '--property=Type=exec', '--property=Restart=no',
            '--property=CPUAccounting=yes', '--property=CPUQuota=100%', '--property=MemoryMax=6G',
            '--property=LimitCORE=0', '--property=KillMode=control-group', '--property=TimeoutStopSec=10s',
            '--property=RuntimeMaxSec=' + str(math.floor(remaining) - 30) + 's',
            '--property=StandardOutput=append:' + str(OUT / 'service.stdout.log'),
            '--property=StandardError=append:' + str(OUT / 'service.stderr.log'),
            '--property=ExecStopPost=' + str(ROOT / '.venv/bin/python') + ' ' + str(OUT / 'verify_resume.py') + ' stop',
            '--working-directory=' + str(ROOT),
            '--setenv=E2_VERIFY_MANIFEST_SHA=' + expected, '--setenv=PYTHONDONTWRITEBYTECODE=1',
            '--setenv=CUDA_VISIBLE_DEVICES=', '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1',
            '--setenv=MKL_NUM_THREADS=1', '--setenv=UV_OFFLINE=1',
            '--setenv=NODE_PATH=/home/bojji/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules',
            str(ROOT / '.venv/bin/python'), str(OUT / 'verify_resume.py'), 'run', expected]
    write('launch-record.json', {'utc': now().isoformat(), 'argv': argv, 'preparation_costs': 'Covered by resumed-production upper charge'})
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=20)
    write('launch-result.json', {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    if result.returncode:
        raise RuntimeError('Verifier systemd launch failed; retained launch result')


if __name__ == '__main__':
    if sys.argv[1:] == ['stop']:
        stop_record()
        raise SystemExit(0)
    mode, expected = sys.argv[1:]
    os.environ['E2_VERIFY_MANIFEST_SHA'] = expected
    try:
        if mode == 'launch':
            launch(expected)
        elif mode == 'run':
            run()
        else:
            raise ValueError('Unknown mode')
    except BaseException as error:
        stop_child()
        try:
            observation = {'service_cpu_seconds': cpu(own_group()), 'global_elapsed_seconds': (now() - LAUNCH).total_seconds(),
                           'verification_output_bytes': size(OUT)}
        except Exception as observation_error:
            observation = {'unavailable': str(observation_error)}
        write('service-failure.json', {'status': 'blocked', 'utc': now().isoformat(),
              'reason': type(error).__name__ + ': ' + str(error), 'traceback': traceback.format_exc(),
              'production_signalled': False, 'costs_settled': (OUT / 'final-resources.json').exists(),
              'resource_observation': observation})
        raise
