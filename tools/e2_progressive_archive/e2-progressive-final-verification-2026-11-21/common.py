"""Read-only production observation and bounded verification support."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess, time

ROOT = Path('/home/bojji/code/kenggo')
OUT = ROOT / 'runs/e2-progressive-final-verification-2026-11-21'
PROD = ROOT / 'runs/e2-progressive-2026-11-21'
REC = ROOT / 'runs/e2-progressive-recovery-2026-11-21'
SERVICE = 'kenggo-e2-progressive-recovery-20261121.service'
UNIT = 'kenggo-e2-final-verification-20261121.service'
HEAD = '59d29abf57d8c03eec783a30222fa288403c8e7b'
LAUNCH = datetime.fromisoformat('2026-11-21T14:13:03.148730+00:00')
DEADLINE = datetime.fromisoformat('2026-11-28T14:13:03.148730+00:00')
PINNED = {'handoff.json':'291c932fda84be522ccd8639a66e0d8d4c5da872f8a59a6fbe485d85c4c3fae3', 'manifest-pass3.json':'2c94dfb5ac5e06486cd43e18d3826c841d2fa32deffc446b0e58651c60074888','recovery.py':'6b4ca08c0d21921046bd15cff7fdbe211a36954b140d761ef9bad7e47c79b433'}

def now(): return datetime.now(timezone.utc)
def load(p): return json.loads(Path(p).read_text())
def write(name, data):
    p = OUT / name
    t = p.with_name('.' + p.name + '.tmp')
    t.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(t, p)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def ref(p): return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def cmd(args): return subprocess.check_output(args, cwd=ROOT, text=True, timeout=20).strip()
def cpu(group):
    return int(dict(line.split() for line in (Path('/sys/fs/cgroup'+group)/'cpu.stat').read_text().splitlines())['usage_usec']) / 1e6
def own_group(): return Path('/proc/self/cgroup').read_text().strip().split('::')[1]
def size(directory):
    total = 0
    for base, dirs, files in os.walk(directory, followlinks=False):
        for name in files:
            try: total += (Path(base)/name).stat().st_size
            except FileNotFoundError: pass
    return total

def source():
    head=cmd(['git','rev-parse','HEAD'])
    state=cmd(['git','status','--porcelain','--untracked-files=all'])
    if head != HEAD or state: raise ValueError('Required source commit or clean state changed')
    return {'head':head,'porcelain':state}

def pins():
    for name, expected in PINNED.items():
        if sha(REC/name) != expected: raise ValueError('Recovery pin differs: '+name)
    hand=load(REC/'handoff.json'); manifest=load(REC/'manifest-pass3.json')
    refs=[{'path':str(ROOT/p),'sha256':h} for p,h in manifest['files'].items()]
    refs += list(hand['artifacts'].values())
    for r in refs:
        if sha(r['path']) != r['sha256']: raise ValueError('Pin differs: '+r['path'])
    return {'manifest_pins':len(manifest['files']), 'handoff_artifacts':len(hand['artifacts']), 'handoff':ref(REC/'handoff.json')}

def process(pid):
    p=Path('/proc')/str(pid)
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),'start_ticks':int(fields[19]),
                'cgroup':(p/'cgroup').read_text().strip().split('::')[1],
                'command':(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')}
    except FileNotFoundError: return None

def group_processes(group):
    p=Path('/sys/fs/cgroup'+group)
    if not p.exists(): return []
    ids=set()
    for f in p.rglob('cgroup.procs'): ids.update(int(x) for x in f.read_text().split())
    return sorted(ids)

def observation():
    hand=load(REC/'handoff.json'); expected=hand['service']; original=load(PROD/'tier1-growth/worker/job.json')
    properties=cmd(['systemctl','--user','show',SERVICE,'-p','LoadState','-p','ActiveState','-p','SubState','-p','MainPID','-p','ControlGroup','-p','Result','-p','ExecMainStatus','-p','CPUUsageNSec','-p','ExecMainStartTimestampMonotonic'])
    state=dict(line.split('=',1) for line in properties.splitlines())
    main=process(int(state.get('MainPID','0'))) if int(state.get('MainPID','0')) else None
    worker=process(expected['worker_pid'])
    issues=[]
    if state.get('ActiveState')=='active':
        if not main or main['pid']!=expected['pid'] or main['start_ticks']!=68123060 or main['cgroup']!=expected['cgroup'] or main['command'][1:]!=[str(REC/'recovery.py'),'run',str(REC/'manifest-pass3.json'),PINNED['manifest-pass3.json']]:
            issues.append('Recovery service identity mismatch')
    if worker and (worker['start_ticks']!=expected['worker_start_ticks'] or worker['cgroup']!=expected['worker_cgroup'] or worker['command']!=original['command']):
        issues.append('Original worker PID exists with another identity; do not infer its original exit status')
    groups={g:group_processes(g) for g in (expected['cgroup'],expected['worker_cgroup'])}
    matches=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            if p.stat().st_uid != os.getuid(): continue
            raw=(p/'cmdline').read_bytes()
            if str(PROD).encode() in raw and any(m in raw for m in (b'kenggo.progressive_jobs',b'kenggo.continuation_training',b'kenggo.progressive_training')):
                item=process(int(p.name))
                if item: matches.append(item)
        except FileNotFoundError: continue
        except PermissionError: issues.append('Cannot inspect same-user process '+p.name)
    hb=load(REC/'heartbeat.json') if (REC/'heartbeat.json').exists() else None
    age=(now()-datetime.fromisoformat(hb['utc'])).total_seconds() if hb else None
    stopped=state.get('ActiveState') in ('inactive','failed') and not main and not any(groups.values()) and not matches and not worker and not issues
    return {'utc':now().isoformat(),'service':state,'main':main,'original_worker':worker,'cgroups':groups,'matching_workers':matches,'identity_issues':issues,'heartbeat':hb,'heartbeat_age_seconds':age,'fully_stopped':stopped}

def gate(obs, ledger, summary):
    if not obs['fully_stopped']: return 'Production is live or its process identity/stop state is unknown'
    if ledger.get('reservations'): return 'Unresolved production reservations: '+', '.join(sorted(ledger['reservations']))
    if not summary or summary.get('status') in (None,'running'): return 'Final production summary is absent or running'
    if summary.get('costs',{}).get('ledger',{}).get('state') != ledger: return 'Final summary does not contain the settled production ledger'
    if not (REC/'stop-hook.json').exists(): return 'Recovery stop-hook record is missing'
    return None
