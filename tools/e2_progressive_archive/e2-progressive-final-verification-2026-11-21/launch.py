"""Start only the new bounded verifier service. Never signal or restart production."""
from common import *
import ast, math
for p in OUT.glob('*.py'):ast.parse(p.read_text())
source();pins();obs=observation()
if obs['identity_issues'] or obs['service'].get('ActiveState')!='active':raise RuntimeError('Recovery identity/state changed before verifier launch')
current=load(OUT/'preparation-start.json');measured=cpu(current['preparation_cgroup'])
active=(now()-datetime.fromisoformat(current['started_utc_conservative'])).total_seconds()+120
if active>1800 or measured>600:raise RuntimeError('Preparation used too much of final verification allowance')
remaining=(DEADLINE-now()).total_seconds()
cost=obs['heartbeat']['costs']
if remaining<3600 or cost['cpu_seconds']+2400>604800 or cost['output_bytes']+1024**3>160*1024**3:raise RuntimeError('Original remaining allowance insufficient')
prep={'schema':'e2_final_verification_preparation_v1','finished_utc':now().isoformat(),'cpu_seconds':measured,'active_wall_seconds':active,
 'measurement_cgroup':current['preparation_cgroup'],'measurement_scope':'Whole current agent-scope lifetime through this timestamp. Later brief launch/handoff CPU is covered by the one 600 CPU-second helper margin; recovery 3600+600 is never added again.',
 'active_wall_method':'Conservative preparation start through this timestamp, plus 120 seconds for launch and handoff. Passive wait excluded.',
 'helper_pins':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.py','.cjs')},
 'preparation_tests':{'guard_tests':10,'browser_fixture_pages':2,'viewports':[1000,390],'selector_combinations':400,'fixture_browser_status':'passed','production_browser_status':'pending'},
 'retained_failed_attempts':['python executable unavailable before directory creation','rg wrong template path; no writes','baseline-probe-attempt1.log','baseline-probe-attempt2.log','fixture/browser-evidence.json','fixture/attempt2/browser-evidence.json'],
 'cpu_margin_seconds':600,'total_active_wall_cap_seconds':3600,'total_measured_cpu_cap_seconds':1800,'gpu_seconds':0,'output_cap_bytes':1024**3}
write('preparation-final.json',prep)
write('verification.json',{'status':'pending','reason':'Recovery and production workers still running; final saved-evidence verification has not run','preparation':ref(OUT/'preparation-final.json'),'root_acceptance':'pending; .16/.17/E2 remain open'})
write('browser-evidence.json',{'status':'pending','reason':'Production page has not been exported; only explicitly synthetic browser-harness fixtures were checked'})
write('resources-pending.json',{'status':'unsettled','preparation':prep,'production_observation':obs,'ledger_appended':False,'recovery_preparation_charged_again':False})
argv=['systemd-run','--user','--unit='+UNIT,'--property=Type=exec','--property=Restart=no','--property=CPUAccounting=yes','--property=CPUQuota=100%',
 '--property=MemoryMax=6G','--property=LimitCORE=0','--property=KillMode=control-group','--property=TimeoutStopSec=10s',
 '--property=RuntimeMaxSec='+str(math.floor(remaining)-30)+'s',
 '--property=StandardOutput=append:'+str(OUT/'service.stdout.log'),'--property=StandardError=append:'+str(OUT/'service.stderr.log'),
 '--property=ExecStopPost='+str(ROOT/'.venv/bin/python')+' '+str(OUT/'service.py')+' stop','--working-directory='+str(ROOT),
 '--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
 '--setenv=UV_OFFLINE=1','--setenv=NODE_PATH=/home/bojji/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules',
 str(ROOT/'.venv/bin/python'),str(OUT/'service.py')]
write('launch-command.json',{'utc':now().isoformat(),'argv':argv,'runtime_derived_from_original_deadline':True})
r=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=20)
write('launch-result.json',{'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'utc':now().isoformat()})
print(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'preparation_cpu_seconds':measured,'preparation_active_wall_seconds_reserved':active,'verifier':UNIT}))
raise SystemExit(r.returncode)
