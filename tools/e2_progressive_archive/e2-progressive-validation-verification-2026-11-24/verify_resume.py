"""Verify saved evidence after the exact validation-resume service has stopped."""
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
OUT = ROOT / 'runs/e2-progressive-validation-verification-2026-11-24'
RESUME = ROOT / 'runs/e2-progressive-validation-resume-2026-11-24'
PROD = ROOT / 'runs/e2-progressive-2026-11-21'
PRODUCTION_UNIT = 'kenggo-e2-progressive-validation-resume-20261124.service'
UNIT = 'kenggo-e2-progressive-validation-verification-20261124.service'
ENTRY = 'e2-validation-resume-verification-20261124-v1'
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


def write(name, record, *, exclusive=False):
    target = OUT / name
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with target.open('x') as stream:
            stream.write(text)
        return
    temporary = target.with_name('.' + target.name + '.tmp')
    temporary.write_text(text)
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
    if manifest.get('schema') != 'e2_validation_resume_verification_manifest_v1' or manifest.get('source_commit') != HEAD:
        raise ValueError('Unexpected verifier manifest identity')
    production = manifest['production_manifest']
    if Path(production['path']).resolve() != (RESUME / 'manifest.json').resolve() or digest(production['path']) != production['sha256']:
        raise ValueError('Production manifest changed')
    for filename, pin in manifest['files'].items():
        if digest(ROOT / filename) != pin:
            raise ValueError('Verification pin changed: ' + filename)
    return {'manifest': ref(OUT / 'manifest.json'), 'production_manifest': production, 'files': len(manifest['files'])}


def unit_state(unit):
    output = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'LoadState', '-p', 'ActiveState', '-p', 'MainPID',
                                      '-p', 'ControlGroup', '-p', 'Result'], text=True, timeout=15)
    return dict(line.split('=', 1) for line in output.splitlines())


def own_group():
    return Path('/proc/self/cgroup').read_text().strip().removeprefix('0::')


def cpu(group):
    fields = dict(line.split() for line in (Path('/sys/fs/cgroup') / group.lstrip('/') / 'cpu.stat').read_text().splitlines())
    return int(fields['usage_usec']) / 1e6


def process(pid):
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(fields[19]),
                'cgroup': (path / 'cgroup').read_text().strip().removeprefix('0::'),
                'command': (path / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')}
    except FileNotFoundError:
        return None


def group_processes(group):
    if not group or group == '/':
        raise ValueError('Missing production cgroup')
    path = Path('/sys/fs/cgroup') / group.lstrip('/')
    return sorted({int(pid) for file in path.rglob('cgroup.procs') for pid in file.read_text().split()}) if path.exists() else []


def expected_command(manifest):
    return [str(ROOT / '.venv/bin/python'), str(RESUME / 'resume.py'), 'run', str(RESUME / 'manifest.json'), manifest['sha256']]


def production_identity(manifest):
    record = load(RESUME / 'service-start.json')
    state = unit_state(PRODUCTION_UNIT)
    identity = {key: record[key] for key in ('pid', 'start_ticks', 'cgroup')}
    identity['command'] = expected_command(manifest)
    if (record['manifest'] != manifest or datetime.fromisoformat(record['deadline_utc']) != DEADLINE
            or not identity['cgroup'].endswith('/' + PRODUCTION_UNIT)
            or state.get('ActiveState') != 'active' or int(state.get('MainPID', '0')) != identity['pid']
            or state.get('ControlGroup') != identity['cgroup'] or process(identity['pid']) != identity):
        raise ValueError('Resumed production identity differs from its exact launch')
    return {'record': ref(RESUME / 'service-start.json'), 'identity': identity, 'manifest': manifest}


def stopped():
    expected = load(OUT / 'launch-record.json')['production']
    if ref(RESUME / 'service-start.json') != expected['record']:
        raise ValueError('Production service-start record changed')
    state = unit_state(PRODUCTION_UNIT)
    identity = expected['identity']
    if state.get('ActiveState') not in ('failed', 'inactive') or int(state.get('MainPID', '0')):
        if state.get('ActiveState') == 'active' and process(int(state.get('MainPID', '0'))) != identity:
            raise ValueError('Production service identity changed while waiting')
        return False
    if process(identity['pid']) == identity or group_processes(identity['cgroup']):
        return False
    for unit in ('kenggo-e2-progressive-recovery-20261121.service', 'kenggo-e2-progressive-resume-20261124.service'):
        old = unit_state(unit)
        if old.get('ActiveState') not in ('failed', 'inactive') or int(old.get('MainPID', '0')):
            return False
        if old.get('ControlGroup') and group_processes(old['ControlGroup']):
            return False
    modules = (b'kenggo.progressive_training', b'kenggo.progressive_jobs', b'kenggo.continuation_training',
               b'resume.py', b'composite', b'prepare.py', b'recovered_jobs.py')
    roots = tuple(str(p).encode() for p in (PROD, RESUME, ROOT / 'runs/e2-progressive-resume-2026-11-24'))
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            command = (entry / 'cmdline').read_bytes()
            if any(root in command for root in roots) and any(module in command for module in modules):
                return False
        except FileNotFoundError:
            pass
    return True


def size(directory):
    total = 0
    for base, _, files in os.walk(directory, followlinks=False):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


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
    if any(not math.isfinite(x) or x < 0 for x in (usage, active, production_cpu, production_bytes, output_bytes)):
        return 'Invalid resource observation'
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
    if CHILD is not None:
        try:
            os.killpg(CHILD.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            CHILD.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid, signal.SIGKILL)
            CHILD.wait(timeout=5)
        try:
            os.killpg(CHILD.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        CHILD = None


def stop_descendants(group):
    if group != own_group() or not group.endswith('/' + UNIT):
        raise ValueError('Refusing to signal any cgroup outside the verifier')
    def remaining():
        return [pid for pid in group_processes(group) if pid != os.getpid()]
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for pid in remaining():
            identity = process(pid)
            if identity and (identity['cgroup'] == group or identity['cgroup'].startswith(group + '/')):
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 5
        while remaining() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not remaining():
            return
    raise ValueError('Verifier descendants did not stop; costs remain observable in its cgroup')


def interrupted(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
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
        returncode = None
        try:
            while CHILD.poll() is None:
                reason = budget_reason(cpu(group), time.monotonic() - active_start, production_cpu, production_bytes, size(OUT))
                if reason:
                    break
                time.sleep(1)
            returncode = CHILD.poll()
        finally:
            stop_child()
            stop_descendants(group)
            result = {'returncode': returncode, 'reason': reason, 'wall_seconds': time.monotonic() - start,
                      'service_cpu_seconds': cpu(group)}
            write(name + '-attempt.json', result)
    return result


def reserve(initial):
    from kenggo.resource_ledger import BudgetLedger
    if not stopped() or load(PROD / 'resources.json') != initial:
        raise ValueError('Production or ledger changed before verification reservation')
    ledger = BudgetLedger.open(PROD / 'resources.json', budget_id=initial['budget_id'], gpu_seconds=initial['gpu_seconds'])
    ledger.reserve(ENTRY, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                   metadata={'purpose': 'Saved-evidence verification, passive wait, descendants and final margin',
                             'preparation_charged_again': False, 'manifest': ref(OUT / 'manifest.json')})
    return ledger.snapshot()['state']


def account(group, active_start, service_start, initial, reserved, summary, failure=None):
    from kenggo.resource_ledger import BudgetLedger
    if not stopped() or load(PROD / 'resources.json') != reserved:
        raise ValueError('Production or ledger changed before verification settlement')
    if set(reserved['reservations']) != {ENTRY} or reserved['entries'] != initial['entries']:
        raise ValueError('Unknown reservations or changed prior ledger entries; retained without mutation')
    usage = cpu(group)
    upper = settled(summary, initial)
    active = max(0, time.monotonic() - active_start)
    total_wall = max(active, time.monotonic() - service_start)
    reason = failure or budget_reason(usage, active, upper, summary['costs']['output_bytes'], size(OUT))
    passed = (not reason and (OUT / 'verification.json').exists()
              and load(OUT / 'verification.json').get('status') == 'saved_evidence_checks_finished'
              and (OUT / 'browser-evidence.json').exists() and load(OUT / 'browser-evidence.json').get('status') == 'passed')
    ledger = BudgetLedger.open(PROD / 'resources.json', budget_id=initial['budget_id'], gpu_seconds=initial['gpu_seconds'])
    ledger.record(ENTRY, allocations=[], outcome='completed' if passed else 'partial', arm_shares={'shared': 1},
                  cpu_seconds=usage + MARGIN, wall_seconds=total_wall,
                  reason=None if passed else reason or 'Saved-evidence or browser checks failed or were unavailable')
    final = ledger.snapshot()
    if final['state']['entries'][:-1] != initial['entries']:
        raise ValueError('Prior ledger entries changed')
    write('final-resources.json', {'schema': 'e2_validation_resumed_final_verification_costs_v1',
          'measured_service_cpu_seconds': usage, 'verification_cpu_upper_seconds': usage + MARGIN,
          'preparation_charged_again': False, 'gpu_seconds': 0, 'active_wall_seconds': active,
          'service_wall_seconds_including_wait': total_wall,
          'global_elapsed_seconds': (now() - LAUNCH).total_seconds(),
          'cumulative_cpu_upper_seconds': upper + usage + MARGIN,
          'cumulative_gpu_seconds': final['charged_gpu_seconds'], 'verification_output_bytes': size(OUT),
          'outcome': 'completed' if passed else 'partial', 'failure': reason,
          'ledger': ref(PROD / 'resources.json'), 'production_summary': ref(RESUME / 'summary.json')})


def run():
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    source()
    validate_pins()
    group = own_group()
    if not group.endswith('/' + UNIT):
        raise ValueError('Verifier must run in its own durable service')
    write('service-start.json', {'utc': now().isoformat(), 'pid': os.getpid(), 'cgroup': group}, exclusive=True)
    service_start = time.monotonic()
    active_start = initial = reserved = summary = None
    failure = None
    try:
        while True:
            if stopped():
                time.sleep(5)
                if stopped():
                    break
            current = load(PROD / 'resources.json')
            production_cpu = sum(e['cpu_seconds'] for e in current['entries'])
            reason = budget_reason(cpu(group), 0, production_cpu, output_bytes=size(OUT))
            if reason:
                raise ValueError(reason)
            write('status.json', {'status': 'waiting', 'utc': now().isoformat(), 'cpu_seconds': cpu(group),
                                  'reason': 'Waiting for the exact new production service and every worker to stop'})
            time.sleep(30)
        active_start = time.monotonic()
        initial = load(PROD / 'resources.json')
        summary = load(RESUME / 'summary.json')
        production_cpu = settled(summary, initial)
        expected = load(OUT / 'launch-record.json')['production']['manifest']
        if summary['continuation']['manifest'] != expected:
            raise ValueError('Final summary does not belong to the exact resumed production launch')
        write('production-resources.json', initial, exclusive=True)
        write('production-summary.json', summary, exclusive=True)
        reserved = reserve(initial)
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
    except BaseException as error:
        failure = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        stop_child()
        stop_descendants(group)
        if reserved is not None:
            account(group, active_start, service_start, initial, reserved, summary, failure)
    result = {'status': 'finished_requires_root_acceptance', 'utc': now().isoformat(),
              'production_status': summary['status'], 'production_reason': summary.get('reason'),
              'checks': ref(OUT / 'verification.json') if (OUT / 'verification.json').exists() else None,
              'browser': ref(OUT / 'browser-evidence.json') if (OUT / 'browser-evidence.json').exists() else None,
              'owner_page': ref(OUT / 'review.html') if (OUT / 'review.html').exists() else None,
              'costs': ref(OUT / 'final-resources.json'),
              'root_work': 'Assess failures and missing checks, obtain fresh independent completion review, deliver reports/source/Beads; E2 stays open.'}
    write('handoff.json', result)
    write('status.json', result)


def stop_record():
    write('stop-record.json', {'utc': now().isoformat(), 'cgroup': own_group(),
          'service_cpu_seconds': cpu(own_group()), 'service_result': os.environ.get('SERVICE_RESULT'),
          'exit_code': os.environ.get('EXIT_CODE'), 'exit_status': os.environ.get('EXIT_STATUS'),
          'production_signalled': False, 'costs_settled': (OUT / 'final-resources.json').exists(),
          'unsettled_cost_action': 'Root must retain and charge any observed unrecorded costs only after every production writer stops.'})


def launch(expected):
    source()
    pins = validate_pins(expected)
    review = load(OUT / 'review-approved.json')
    if review.get('manifest_sha256') != expected or review.get('verdict') != 'no significant findings':
        raise ValueError('Fresh review does not approve this exact verifier manifest')
    if (OUT / 'launch-record.json').exists() or (OUT / 'service-start.json').exists() or unit_state(UNIT).get('ActiveState') == 'active':
        raise ValueError('Verifier already launched; refusing duplicate')
    production = production_identity(pins['production_manifest'])
    remaining = (DEADLINE - now()).total_seconds()
    if remaining < WALL_CAP:
        raise ValueError('Original deadline leaves insufficient verification reserve')
    argv = ['systemd-run', '--user', '--unit=' + UNIT, '--property=Type=exec', '--property=Restart=no',
            '--property=CPUAccounting=yes', '--property=CPUQuota=100%', '--property=MemoryMax=6G',
            '--property=LimitCORE=0', '--property=KillMode=control-group', '--property=TimeoutStopSec=15s',
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
    write('launch-record.json', {'utc': now().isoformat(), 'argv': argv, 'production': production,
          'preparation_costs': 'Covered once by validation-resumed production upper charge'}, exclusive=True)
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=20)
    write('launch-result.json', {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    if result.returncode:
        raise RuntimeError('Verifier systemd launch failed; retained launch result; no automatic retry')


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
