"""Passive durable waiter; starts saved-evidence checks only after production settles."""
from common import *
import signal, sys, traceback, shutil

PREP=load(OUT/'preparation-final.json') if (OUT/'preparation-final.json').exists() else None
GROUP=own_group()
TMP_FD=os.open(OUT/'tmp',os.O_RDONLY|os.O_DIRECTORY)
os.environ['TMPDIR']=f'/proc/{os.getpid()}/fd/{TMP_FD}'
ACTIVE=None
CHILD=None
PRODUCTION_CPU=0
PRODUCTION_BYTES=0

def resources():
    elapsed=0 if ACTIVE is None else time.monotonic()-ACTIVE
    return {'utc':now().isoformat(),'preparation_cpu_seconds':PREP['cpu_seconds'],
            'service_cpu_seconds':cpu(GROUP),'measured_cpu_seconds':PREP['cpu_seconds']+cpu(GROUP),
            'cpu_margin_seconds':600,'active_wall_seconds':PREP['active_wall_seconds']+elapsed,
            'passive_wait_wall_seconds':max(0,(now()-datetime.fromisoformat(PREP['finished_utc'])).total_seconds()-elapsed),
            'global_elapsed_seconds':(now()-LAUNCH).total_seconds(),'gpu_seconds':0,'verification_output_bytes':size(OUT)}

def record_status(status,reason,observation_record=None):
    r={'schema':'e2_final_verification_service_v1','status':status,'reason':reason,'service':UNIT,
       'resources':resources(),'observation':observation_record,'root_acceptance':'pending; kenggo-ydo.16/.17/E2 remain open'}
    write('service-state.json',r);return r

def stop_own_child():
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid,signal.SIGTERM)
        try:CHILD.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()

def interrupted(signum,frame):
    stop_own_child()
    raise InterruptedError('Verification service interrupted; production was not signalled')

def remaining():
    r=resources();return min(3600-r['active_wall_seconds'],(DEADLINE-now()).total_seconds()-90),min(1800-r['measured_cpu_seconds'],604800-PRODUCTION_CPU-r['measured_cpu_seconds']-600)

def run_helper(name,argv):
    global CHILD
    before=resources();start=time.monotonic()
    with (OUT/(name+'.stdout.log')).open('xb') as stdout,(OUT/(name+'.stderr.log')).open('xb') as stderr:
        CHILD=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,start_new_session=True)
        reason=None
        while CHILD.poll() is None:
            wall,leftcpu=remaining()
            if wall<20 or leftcpu<20 or size(OUT)>min(1000*1024**2,160*1024**3-PRODUCTION_BYTES-16*1024**2):
                reason='Verification budget guard stopped this helper';stop_own_child();break
            time.sleep(1)
        result={'name':name,'argv':argv,'returncode':CHILD.returncode,'reason':reason,'wall_seconds':time.monotonic()-start,'before':before,'after':resources()}
    write(name+'-attempt.json',result);CHILD=None;return result

def account(obs):
    # Only these new uniquely named entries are ours. Never settle a production reservation.
    from kenggo.resource_ledger import BudgetLedger
    frozen=load(OUT/'production-resources.json');current=load(PROD/'resources.json')
    eq=frozen==current
    if not eq or current['reservations'] or not observation()['fully_stopped']:
        raise RuntimeError('Production ledger or stop identity changed before verification accounting')
    r=resources();charge=r['measured_cpu_seconds']+600
    used=sum(e['cpu_seconds'] for e in frozen['entries'])
    upper=max(used,load(OUT/'production-summary.json')['costs']['cpu_seconds'])
    if upper+charge>604800 or r['measured_cpu_seconds']>1800 or r['active_wall_seconds']>3600:
        raise RuntimeError('Verification costs exceed remaining original allowance; retained outside ledger for root')
    ledger=BudgetLedger.open(PROD/'resources.json',budget_id=frozen['budget_id'],gpu_seconds=frozen['gpu_seconds'])
    entries=[('e2-final-verification-20261121-measured-v1',r['measured_cpu_seconds'],r['active_wall_seconds'],
              {'purpose':'Saved-evidence verification preparation, passive-wait CPU, export, browser and all helper attempts','measurement':r}),
             ('e2-final-verification-20261121-margin-v1',600,0,{'purpose':'Conservative final helper, shutdown and measurement margin; distinct from recovery 3600+600 charges'})]
    outcome='completed' if (OUT/'verification.json').exists() and load(OUT/'verification.json').get('status')=='saved_evidence_checks_finished' and (OUT/'browser-evidence.json').exists() and load(OUT/'browser-evidence.json').get('status')=='passed' else 'partial'
    for run_id,seconds,wall,meta in entries:
        ledger.reserve(run_id,estimate_gpu_seconds=0,shutdown_gpu_seconds=0,metadata=meta)
        ledger.record(run_id,allocations=[],outcome=outcome,arm_shares={'shared':1},cpu_seconds=seconds,wall_seconds=wall,reason=None if outcome=='completed' else 'Incomplete or failing verification checks; retain evidence for root')
    after=ledger.snapshot()
    if after['state']['entries'][:len(frozen['entries'])]!=frozen['entries']:raise RuntimeError('Existing ledger entries changed')
    write('final-resources.json',{'schema':'e2_final_verification_resources_v1','production_snapshot':ref(OUT/'production-resources.json'),
      'production_cpu_seconds':used,'production_cpu_upper_seconds':upper,'production_gpu_seconds':sum(e['gpu_seconds'] for e in frozen['entries']),
      'verification_measured_through_accounting':r,'verification_charged_cpu_seconds':charge,'verification_gpu_seconds':0,
      'cumulative_cpu_seconds':used+charge,'cumulative_cpu_upper_seconds':upper+charge,'cumulative_gpu_seconds':after['charged_gpu_seconds'],
      'global_elapsed_seconds':(now()-LAUNCH).total_seconds(),'global_output_bytes':size(PROD)+size(REC)+size(OUT),
      'ledger':ref(PROD/'resources.json'),'ledger_entries_added':[x[0] for x in entries],
      'recovery_charges_added_again':False,'final_production_summary_preserved':ref(PROD/'summary.json'),
      'note':'Production summary and original review remain immutable; cumulative ledger includes these separate verification entries. Passive waits are not added to global elapsed time.'})

def handoff(status,reason):
    files={p.name:ref(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('handoff.json','service.stdout.log','service.stderr.log','service-state.json','stop-record.json') and not p.name.startswith('.')}
    write('handoff.json',{'schema':'e2_final_verification_handoff_v1','status':status,'reason':reason,'updated_utc':now().isoformat(),
      'files':files,'owner_page':ref(OUT/'review-settled-production.html') if (OUT/'review-settled-production.html').exists() else (ref(PROD/'review.html') if (PROD/'review.html').exists() else None),
      'root_work':['Inspect verification.json, browser-evidence.json and final-resources.json; unavailable or failing checks remain unresolved.',
      'Preserve the original page and production summary; use the new owner page only if its export and browser QA passed.',
      'Perform fresh independent Astra/max completion review, acceptance and safe versioned delivery. Keep kenggo-ydo.16/.17 and E2 open until those gates pass.'],
      'resource_observation':resources(),'no_production_signals':True,'no_new_inference_or_training':True})

def main():
    global ACTIVE, PRODUCTION_CPU, PRODUCTION_BYTES
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    for name,pinvalue in PREP['helper_pins'].items():
        if sha(OUT/name)!=pinvalue:raise ValueError('Verification helper changed: '+name)
    pins();source()
    write('service-start.json',{'utc':now().isoformat(),'pid':os.getpid(),'cgroup':GROUP,'preparation':ref(OUT/'preparation-final.json')})
    while True:
        obs=observation();r=resources()
        record_status('waiting','Production/final accounting must finish before saved-evidence verification',obs)
        if r['measured_cpu_seconds']>1750:raise RuntimeError('Passive wait reached verification CPU reserve')
        observed_costs=(obs.get('heartbeat') or {}).get('costs',{})
        if observed_costs.get('cpu_seconds',0)+r['measured_cpu_seconds']+600>=604800:raise RuntimeError('Original remaining CPU allowance cannot cover verifier costs')
        if observed_costs.get('output_bytes',0)+r['verification_output_bytes']>=160*1024**3:raise RuntimeError('Original output allowance cannot cover verification evidence')
        if (DEADLINE-now()).total_seconds()<120:raise RuntimeError('Original absolute deadline leaves insufficient verification time')
        if obs['identity_issues']:raise RuntimeError('; '.join(obs['identity_issues']))
        if obs['service'].get('ActiveState') in ('inactive','failed'):
            if not obs['fully_stopped']:
                write('production-stop-blocker.json',obs)
                raise RuntimeError('Recovery service stopped with live or unknown production workers; no reservation settled')
            time.sleep(5)
            again=observation()
            ledger=load(PROD/'resources.json');summary=load(PROD/'summary.json') if (PROD/'summary.json').exists() else None
            why=gate(again,ledger,summary)
            if why:write('production-stop-blocker.json',{'observation':again,'ledger':ledger,'reason':why});raise RuntimeError(why)
            hook=load(REC/'stop-hook.json')
            if hook.get('status')!='stop_requested':raise RuntimeError('Stop-hook identity or outcome needs root assessment')
            break
        time.sleep(60)
    ACTIVE=time.monotonic()
    write('production-stop.json',again)
    for sourcefile,target in ((PROD/'resources.json','production-resources.json'),(PROD/'summary.json','production-summary.json')):
        with (OUT/target).open('xb') as dest:dest.write(sourcefile.read_bytes())
    budget=resources();used=sum(e['cpu_seconds'] for e in ledger['entries']);PRODUCTION_CPU=max(used,summary['costs']['cpu_seconds']);PRODUCTION_BYTES=size(PROD)+size(REC);wall,leftcpu=remaining()
    if wall<180 or leftcpu<60 or PRODUCTION_CPU+budget['measured_cpu_seconds']+600+60>604800 or size(PROD)+size(REC)+size(OUT)+64*1024**2>160*1024**3:
        raise RuntimeError('Insufficient original allowance for bounded final verification')
    # The service watchdog limits the aggregate, including failed helper attempts.
    record_status('verifying','Production fully stopped; final production ledger snapshot retained',again)
    run_helper('saved-evidence',[str(ROOT/'.venv/bin/python'),str(OUT/'verify.py')])
    page=OUT/'review-settled-production.html'
    if not page.exists():page=PROD/'review.html'
    if page.exists() and remaining()[0]>60 and remaining()[1]>30:
        run_helper('browser',['/home/bojji/.nvm/versions/node/v25.9.0/bin/node',str(OUT/'browser.cjs'),*list(dict.fromkeys(str(p) for p in (PROD/'review.html',page) if p.exists()))])
    else:write('browser-evidence.json',{'status':'unavailable','reason':'No saved owner page or insufficient original verification allowance'})
    source();pins()
    account(again)
    status='verification_finished_requires_root_acceptance'
    reason='Read passed/failed/unavailable checks; service exit is not experiment success'
    record_status(status,reason,again);handoff(status,reason)

if __name__=='__main__':
    if sys.argv[1:]==['stop']:
        record={'utc':now().isoformat(),'cpu_seconds_at_stop_hook':cpu(GROUP),'service_result':os.environ.get('SERVICE_RESULT'),'exit_code':os.environ.get('EXIT_CODE'),'exit_status':os.environ.get('EXIT_STATUS'),'production_signalled':False}
        write('stop-record.json',record)
        if not (OUT/'handoff.json').exists():write('interrupted-verification.json',{'status':'blocked','reason':'Verifier stopped before final handoff','stop':record})
    else:
        try:main()
        except BaseException as e:
            stop_own_child();reason=type(e).__name__+': '+str(e)
            write('service-failure.json',{'reason':reason,'traceback':traceback.format_exc()})
            write('unsettled-verification-resources.json',{'status':'not_appended','reason':reason,'measured_resources':resources(),'note':'Do not infer production reservations or GPU stop times. Root must assess any later accounting.'})
            record_status('blocked',reason);handoff('blocked',reason)
            raise
