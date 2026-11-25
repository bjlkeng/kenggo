"""Verify saved fitting/development evidence; never load a model or protected targets."""
from common import *
from collections import Counter, defaultdict
import math, re, traceback

RESULT={'schema':'e2_saved_evidence_verification_v1','status':'running','checks':[], 'limits':[
 'No new training, inference, data acquisition, full game replay or complete-game panel.',
 'Protected targets are not decoded. Their shard bytes and membership metadata may be checked.',
 'Checkpoint contents are hash-checked, not deserialized. Optimizer reuse is supported by saved provenance and pinned loader code.',
 'One seed, development selection only. No promotion, stronger-play or human-rank claim.',
 'Premature passing and earlier 0.5-komi value concerns remain open.'], 'started_utc':now().isoformat()}
HASHES={}
def persist(): write('verification.json',RESULT)
def check(name, f):
    try:
        value=f(); item={'name':name,'status':'passed','evidence':value}
    except Exception as e:
        item={'name':name,'status':'failed','error':type(e).__name__+': '+str(e),'traceback':traceback.format_exc()}
    RESULT['checks'].append(item);persist();return item

def unavailable(name, why):
    RESULT['checks'].append({'name':name,'status':'unavailable','reason':why});persist()

def pin(r):
    p=(ROOT/r['path']).resolve()
    if '.config' in p.parts or 'credential' in p.name.lower(): raise ValueError('Refusing credential path')
    expected=r['sha256']; key=str(p)
    if key not in HASHES: HASHES[key]=sha(p)
    if HASHES[key]!=expected: raise ValueError('Hash differs: '+key)
    return p

def read(r): return load(pin(r))
def refs(obj):
    if isinstance(obj,dict):
        if isinstance(obj.get('path'),str) and isinstance(obj.get('sha256'),str): pin(obj)
        for k,v in obj.items():
            if 'credential' not in k: refs(v)
    elif isinstance(obj,list):
        for v in obj: refs(v)

def eq(a,b,label):
    if a!=b: raise ValueError(label)

def close(a,b,label):
    if isinstance(a,dict) and isinstance(b,dict):
        eq(a.keys(),b.keys(),label+' keys')
        for k in a: close(a[k],b[k],label+'/'+k)
    elif isinstance(a,list) and isinstance(b,list):
        eq(len(a),len(b),label+' length')
        for i,(x,y) in enumerate(zip(a,b)): close(x,y,label+str(i))
    elif type(a) in (float,int) and type(b) in (float,int):
        if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10): raise ValueError(label+' numeric difference')
    else: eq(a,b,label)

def stage_checks():
    records=[]
    for name,stage in SUMMARY['stages'].items():
        refs(stage)
        config=read(stage['config']); provenance=read(stage['provenance']); worker=read(stage['summary'])
        eq(config['inputs'],provenance['inputs'],name+' input provenance')
        eq(provenance['source'],{'commit':HEAD,'patch_sha256':hashlib.sha256(b'').hexdigest()},name+' source identity')
        refs(config['inputs'])
        records.append({'name':name,'stage_status':stage['status'],'worker_status':worker['status'],
                        'job_status':stage.get('job',{}).get('status'),'completion_basis':stage.get('completion_basis'),
                        'config':stage['config'],'summary':stage['summary']})
        if name=='tier1-growth':
            eq(stage['completion_basis'],'validated_saved_artifacts_original_exit_unknown','Growth exit distinction')
    failures=SUMMARY.get('failed_attempts',[]);refs(failures)
    for f in sorted(PROD.glob('*/worker/*failure*.json')): records.append({'failure':ref(f),'record':load(f)})
    # Include directories that never reached the summary's completed-stage map.
    seen={s['name'] for s in records if 'name' in s}
    for f in sorted(PROD.glob('*/worker/job.json')):
        if f.parent.parent.name not in seen:
            item={'name':f.parent.parent.name,'job':ref(f),'status':load(f).get('status'),'registered_stage':False}
            for candidate in ('training_summary.json','training_progress.json'):
                saved=f.parent/candidate
                if saved.exists():
                    attempt=load(saved);refs(attempt)
                    item[candidate]={'reference':ref(saved),'status':attempt.get('status'),'attempted_updates':attempt.get('attempted_updates'),'completed_updates':attempt.get('completed_updates'),'uncertain_updates':attempt.get('uncertain_updates'),'sampler_step':attempt.get('progress',{}).get('sampler_step'),'commits':attempt.get('commits',[])}
            records.append(item)
    ledger=load(OUT/'production-resources.json')
    original=load(REC/'before-resources.json')
    eq(ledger['entries'][:len(original['entries'])],original['entries'],'Original settled entries changed')
    if (REC/'growth-recovered.json').exists():
        growth_entry=next(e for e in ledger['entries'] if e['run_id']=='tier1-growth')
        eq(growth_entry['outcome'],'partial','Original growth attempt must stay partial')
        eq(load(REC/'growth-recovered.json')['exit_code'],None,'Original exit code must stay unknown')
        eq(load(REC/'tracking-recovered.json')['status'],'partial','Tracking interruption must remain partial')
    return {'stages_and_failures':records,'failed_attempts':failures,'recovery_records':{p.name:{'reference':ref(p),'record':load(p)} for p in [REC/'growth-recovered.json',REC/'tracking-recovered.json',REC/'recovery-failure.json',REC/'export-failure.json',REC/'stop-hook.json'] if p.exists()}}

MEMBERS={}
def membership(r):
    if r['sha256'] not in MEMBERS:
        from kenggo.data_splits import read_membership
        MEMBERS[r['sha256']]=read_membership(pin(r).parent,r['sha256'])[1]
    return MEMBERS[r['sha256']]

DATASETS={}
def dataset(r):
    from kenggo.continuation_data import validate_lineage
    from kenggo.progressive_training import dataset_inputs
    import numpy as np
    import pyarrow.parquet as pq
    from kenggo.canonical_data import digest
    inputs=dataset_inputs(r,BASE)
    refs(inputs)
    manifest,normalizer=validate_lineage(inputs)
    publication=read(r)
    if 'request' in publication:
        refs(publication['artifacts']);refs(publication['request'])
        eq(publication['actual_train_rows'],manifest['counts']['train']['examples'],'Growth publication row count')
        for name,h in publication['code_sha256'].items():pin({'path':name,'sha256':h})
    members=membership(inputs['split']); original=read(BASE['examples'])
    root=pin(inputs['examples']).parent
    expected={row['source_url']:row for row in members['train']}
    game_counts=Counter(); rank_counts=Counter(); accounts=defaultdict(set); bits={}; counts=Counter(); komi_counts=Counter()
    for role,shards in manifest['artifacts'].items():
        if role not in ('train','excluded'):
            eq(shards,original['artifacts'][role],'Protected/development original shard list changed: '+role)
        for shard in shards:
            for kind in ('arrays','metadata'): pin({'path':str(root/shard[kind]),'sha256':shard[kind+'_sha256']})
            if role!='train' or not shard['rows']: continue
            rows=pq.read_table(root/shard['metadata']).to_pylist()
            eq(len(rows),shard['rows'],'Shard metadata rows')
            with np.load(root/shard['arrays'],allow_pickle=False) as z:
                arrays={k:z[k] for k in ('policy_valid','value_valid','duration_valid','scalars','scalar_valid','value','action')}
            for k,a in arrays.items():
                eq(a.shape[0],len(rows),'Array row count '+k)
                if not np.isfinite(a).all(): raise ValueError('Nonfinite '+k)
            if not arrays['policy_valid'].all() or arrays['duration_valid'].any(): raise ValueError('Policy/duration masks differ')
            if arrays['scalar_valid'][:,12:].any(): raise ValueError('Dynamic clock/overtime masks changed')
            scalar_names=manifest['scalars']
            for i,name in enumerate(scalar_names):
                if any(word in name for word in ('remaining','elapsed','dynamic')) and arrays['scalar_valid'][:,i].any(): raise ValueError('Dynamic clock mask changed: '+name)
            if not np.isin(np.abs(arrays['scalars'][:,1]),[.5,6.5]).all(): raise ValueError('Komi support changed')
            if not np.isin(arrays['value'][arrays['value_valid']],[-1,1]).all(): raise ValueError('Invalid recorded mover outcome')
            counts.update(rows=len(rows),value_valid=int(arrays['value_valid'].sum()),duration_valid=int(arrays['duration_valid'].sum()))
            for i,row in enumerate(rows):
                source_id=row['source_url']; ply=row['ply']
                eq(row['row_key'],digest([source_id,ply]),'Fitting row key')
                if source_id not in expected or row['role']!='train': raise ValueError('Unexpected fitting game/role')
                if not 0<=ply<10000: raise ValueError('Unbounded or invalid ply')
                bit=1<<ply
                if bits.get(source_id,0)&bit: raise ValueError('Duplicate fitting position')
                bits[source_id]=bits.get(source_id,0)|bit
                game_counts[source_id]+=1;rank_counts[row['source_rank_mover']]+=1;accounts[row['source_rank_mover']].add(row['group_mover'])
                komi_counts[str(abs(float(arrays['scalars'][i,1])))]+=1
                if bool(arrays['value_valid'][i])!=(expected[source_id]['value_actions']>0): raise ValueError('Value mask differs from membership')
    eq(game_counts,Counter({s:r['policy_actions'] for s,r in expected.items()}),'Whole-game counts')
    eq(counts['rows'],manifest['counts']['train']['examples'],'Manifest train rows')
    eq(counts['value_valid'],manifest['counts']['train']['value_valid'],'Manifest valid outcomes')
    for s,n in game_counts.items():
        if bits[s] != (1<<n)-1: raise ValueError('Fitting game plies are not whole and contiguous')
    DATASETS[r['sha256']]=set(expected)
    return {'dataset':r,'rows':counts['rows'],'games':len(game_counts),'rank_rows':rank_counts,'rank_accounts':{k:len(v) for k,v in accounts.items()},'komi_rows':komi_counts,'value_valid':counts['value_valid'],'normalizer_fit_rows':original['counts']['train']['examples'],'normalizer_fit_examples':BASE['examples'],'lineage':'validate_lineage passed','protected_targets':'hashed only; never decoded','board_legality_replay':'not repeated; hash integrity and metadata/small-array checks only'}

FIXED={}; FIXED_JOIN={}; DIAGNOSTICS={}; DIAGNOSTIC_INPUT_HASHES={}
def fixed_checks():
    import numpy as np
    import pyarrow.parquet as pq
    from kenggo.encoding import INPUT_KEYS,apply_normalizer
    from kenggo.examples import LABEL_SCHEMA
    from kenggo.recorded_data import _inputs_hash
    from kenggo.canonical_data import digest
    frozen=read(PLAN['inputs']['fixed_comparison']);games=set();rows=scope=0;details=[];wanted={}
    for common in frozen['common_sets']:
        manifest=read(common['manifest']);spec=common['spec'];actual=manifest['rows']
        eq(set(spec['source_ids']),{r['source_url'] for r in actual},'Fixed game IDs')
        eq(digest(actual),manifest['rows_sha256'],'Frozen row manifest digest')
        FIXED[spec['id']]=manifest;FIXED_JOIN[spec['id']]={}
        for row in actual:
            if row['row_key'] in wanted:raise ValueError('Fixed scopes share a row')
            wanted[row['row_key']]=(spec['id'],row)
    base=read(BASE['examples']);root=pin(BASE['examples']).parent;normalizer=read(BASE['normalizer'])
    for role in ('population_development','validation_development'):
        for shard in base['artifacts'][role]:
            metadata=pq.read_table(pin({'path':str(root/shard['metadata']),'sha256':shard['metadata_sha256']})).to_pylist()
            indices=[i for i,row in enumerate(metadata) if row['row_key'] in wanted]
            if not indices:continue
            with np.load(pin({'path':str(root/shard['arrays']),'sha256':shard['arrays_sha256']}),allow_pickle=False) as z:
                arrays={k:z[k] for k in (*INPUT_KEYS,*LABEL_SCHEMA)}
            normalized=apply_normalizer({k:arrays[k][indices] for k in INPUT_KEYS},normalizer)
            for offset,i in enumerate(indices):
                row=metadata[i];identity,before=wanted[row['row_key']]
                for k in ('row_key','source_url','canonical_sha256','sgf_sha256','state_sha256','ply','mover','rank_hidden'):eq(row[k],before[k],'Frozen metadata '+k)
                labels={k:arrays[k][i].item() for k in LABEL_SCHEMA}
                for k in ('policy_valid','value_valid','duration_valid'):eq(labels[k],before[k],'Frozen output mask '+k)
                for k,v in before['input_masks'].items():eq(arrays[k][i].tolist(),v,'Frozen input mask '+k)
                eq(_inputs_hash({k:arrays[k][i] for k in INPUT_KEYS}),before['inputs_sha256'],'Frozen raw input hash')
                FIXED_JOIN[identity][row['row_key']]={**row,**labels,'normalized_inputs_sha256':_inputs_hash({k:normalized[k][offset] for k in INPUT_KEYS})}
    for common in frozen['common_sets']:
        name=common['spec']['id'];m=FIXED[name];joined=FIXED_JOIN[name]
        eq(set(joined),{r['row_key'] for r in m['rows']},'Frozen rows found in original development shards')
        targets=[{k:joined[r['row_key']][k] for k in LABEL_SCHEMA} for r in m['rows']]
        # The frozen replay manifest serialized integral outcome/duration labels as JSON integers; arrays store float32.
        for target in targets:
            for key in ('value','duration'):
                if target[key]!=int(target[key]):raise ValueError('Nonintegral frozen outcome/duration')
                target[key]=int(target[key])
        eq(digest(targets),m['targets_sha256'],'Frozen target digest')
        games.update(common['spec']['source_ids']);rows+=len(joined);scope+=sum(r['source_rank_mover'] in RANKS for r in joined.values())
        details.append({'id':name,'games':len(common['spec']['source_ids']),'rows':len(joined),'manifest':common['manifest']})
    eq((len(games),rows,scope),(539,120450,111591),'Fixed comparison coverage')
    return {'games':len(games),'rows':rows,'in_scope':scope,'excluded':rows-scope,'samples':details,'method':'Original allowed-role shards joined by frozen row key; exact stored input and target hashes, including masks. No replay.'}

def diagnostic(r,purpose):
    if r['sha256'] in DIAGNOSTICS:return DIAGNOSTICS[r['sha256']]
    import numpy as np
    from kenggo.encoding import INPUT_KEYS
    from kenggo.recorded_data import _inputs_hash
    d=read(r);eq(d['purpose'],purpose,'Diagnostic purpose');eq(d['aggregate_ranks'],RANKS,'Rank scope')
    eq(d['normalizer_sha256'],BASE['normalizer']['sha256'],'Diagnostic normalizer')
    if d['role'] not in ('train','population_development','validation_development'):raise ValueError('Protected diagnostic')
    member=membership(d['data']['split']);allowed={m['source_url']:m for m in member[d['role']]}
    ids=set(d['source_ids']);eq(len(ids),len(d['source_ids']),'Repeated diagnostic game')
    counts=Counter(x['source_url'] for x in d['rows']);eq(counts,Counter({s:allowed[s]['policy_actions'] for s in ids}),'Diagnostic whole games')
    eq(len({x['row_key'] for x in d['rows']}),len(d['rows']),'Duplicate diagnostic rows')
    if not set(RANKS)<={x['source_rank_mover'] for x in d['rows']}: raise ValueError('Missing required diagnostic ranks')
    with np.load(pin(d['arrays']),allow_pickle=False) as z:
        arrays={k:z[k] for k in ('action','policy_valid','value','value_valid','duration_valid','legal')}
        inputs={k:z[k] for k in INPUT_KEYS}
    DIAGNOSTIC_INPUT_HASHES[r['sha256']]={row['row_key']:_inputs_hash({k:inputs[k][i] for k in INPUT_KEYS}) for i,row in enumerate(d['rows'])}
    for a in arrays.values():eq(a.shape[0],len(d['rows']),'Diagnostic label size')
    DIAGNOSTICS[r['sha256']]=(d,arrays);return d,arrays

def preparation_checks():
    used=set();fitting=set().union(*DATASETS.values()) if DATASETS else set()
    fixed_games={x['source_url'] for m in FIXED.values() for x in m['rows']};details=[]
    for tier in SUMMARY['tiers']:
        if 'preparation' not in tier:continue
        p=read(tier['preparation']);valid,va=diagnostic(p['epoch_validation'],'epoch_validation');monitor,ma=diagnostic(p['train_monitor'],'train_monitor')
        ids=set(valid['source_ids'])
        if ids&(used|fixed_games|fitting|set(monitor['source_ids'])):raise ValueError('Stopping validation overlaps fitting/fixed/previous stopping')
        used|=ids
        if not set(monitor['source_ids'])<=DATASETS.get(PLAN['inputs']['parent_summary']['sha256'],set()):raise ValueError('Monitor is outside original fitting population')
        for name,r in p['assessments'].items():
            d,arrays=diagnostic(r,'assessment');old=FIXED_JOIN[name]
            eq(set(old),{x['row_key'] for x in d['rows']},'Fixed row keys')
            for i,row in enumerate(d['rows']):
                before=old[row['row_key']]
                eq(DIAGNOSTIC_INPUT_HASHES[r['sha256']][row['row_key']],before['normalized_inputs_sha256'],'Fixed normalized inputs and legal masks')
                for k in ('source_url','ply','mover','group_mover','group_opponent','source_rank_mover'):eq(row[k],before[k],'Fixed metadata '+k)
                for k,a in arrays.items():
                    if k in before: eq(a[i].tolist(),before[k],'Fixed labels/masks '+k)
        details.append({'tier':tier['tier'],'stopping_games':len(ids),'stopping_rows':len(valid['rows']),'monitor_games':len(monitor['source_ids']),'monitor_rows':len(monitor['rows']),'epoch_validation':p['epoch_validation']})
    return details

def training_checks(tier):
    from kenggo.continuation_training import observe_validation
    details={};paired={};schedule={};commits_by_arm={}
    for arm in ('control','candidate'):
        stopping={'best_score':None,'tolerance_reference':None,'bad_rounds':0};best=None;last=None;measures=[];committed=[];attempts=[]
        for attempt in [a for a in tier['training'] if a['arm']==arm]:
            stage=attempt['stage'];cfg=read(stage['config']);worker=read(stage['summary']);refs(worker)
            if attempt['epoch']==0:eq(cfg['inputs']['parent_checkpoint']['sha256'],tier['parent_checkpoint']['sha256'],'Matched initial parent')
            elif last:eq(cfg['inputs']['parent_checkpoint']['sha256'],last['sha256'],'Continuation latest checkpoint')
            schedule[(arm,attempt['epoch'])]=cfg['parameters']['round_updates']
            eq(cfg['parameters']['round_updates'],math.ceil(tier['training_rows']/128),'Scheduled interval updates')
            eq(cfg['inputs']['normalizer'],BASE['normalizer'],'Original normalizer')
            eq(cfg['parameters']['model'],PLAN['parameters']['model'],'Frozen model architecture')
            for cr in worker.get('commits',[]):
                c=read(cr);m=read(c['metrics']);refs(c)
                stopping,new,meaningful=observe_validation(stopping,m['validation']['nll'])
                eq(c['stopping'],stopping,'Committed patience');eq(m['new_raw_best'],new,'Best flag');eq(m['meaningful_improvement'],meaningful,'Tolerance flag')
                if new:best=c['checkpoint']
                eq(c['best_checkpoint']['sha256'],best['sha256'],'Committed minimum validation checkpoint')
                eq(m['sampler_step'],m['round']*cfg['parameters']['round_updates'],'Committed update cursor')
                for name in ('validation','training_monitor'):
                    if not all(math.isfinite(m[name][k]) for k in ('nll','top1','value_mse')):raise ValueError('Nonfinite diagnostic')
                    input_name='epoch_validation' if name=='validation' else 'train_monitor'
                    eq(m[name]['diagnostic_sha256'],cfg['inputs'][input_name]['sha256'],'Committed diagnostic pin')
                    eq(m[name]['aggregate_ranks'],RANKS,'Committed rank scope')
                measures.append(m);committed.append(c)
            last=worker.get('latest_checkpoint')
            if stage['status']=='completed':paired[(arm,attempt['epoch'])]=worker
            attempts.append({'epoch':attempt['epoch'],'status':stage['status'],'completed_updates':worker.get('completed_updates'),'uncertain_updates':worker.get('uncertain_updates'),'sampler_step':worker.get('progress',{}).get('sampler_step'),'latest_checkpoint':last,'best_checkpoint':worker.get('best_checkpoint')})
        selected=tier.get('selected',{}).get(arm)
        if selected:
            last_pair=tier['completed_intervals'];through=[c for c in committed if c['round']<=last_pair]
            if not through:raise ValueError('Selected checkpoint lacks complete matched pair')
            eq(selected['sha256'],through[-1]['best_checkpoint']['sha256'],'Selection must use last complete pair')
        commits_by_arm[arm]=committed
        details[arm]={'attempts':attempts,'committed_measurements':measures,'selected':selected,'selected_exposure_note':'Selected updates may differ between arms; equal opportunities are not identical selected exposures.'}
    if 'completed_intervals' in tier:
        final=tier['completed_intervals']
        for epoch in range(final+1):
            a,b=paired[('control',epoch)],paired[('candidate',epoch)]
            eq(schedule[('control',epoch)],schedule[('candidate',epoch)],'Unequal scheduled opportunities')
            eq(a['progress']['sampler_step'],b['progress']['sampler_step'],'Unequal completed pair updates')
        if tier.get('training_stop')=='validation_plateau':
            if not all(paired[(a,final)]['progress']['stopping']['bad_rounds']>=3 for a in ('control','candidate')):raise ValueError('Plateau without both patience counters')
    elif tier.get('selected') or tier.get('comparison_status')=='completed':raise ValueError('Comparison without initial matched pair')
    hypotheses=[]
    for arm,v in details.items():
        m=v['committed_measurements']
        if len(m)>1:
            train=m[-1]['training_monitor']['nll']-m[0]['training_monitor']['nll'];valid=m[-1]['validation']['nll']-m[0]['validation']['nll']
            hypotheses.append({'arm':arm,'training_monitor_nll_change':train,'stopping_nll_change':valid,'hypothesis_not_fact': 'Fitting improved while validation worsened; overfitting or sample mismatch is a hypothesis.' if train<0 and valid>0 else 'Both trajectories need interpretation alongside fixed comparisons; these curves alone do not identify a cause.'})
    details['hypotheses']=hypotheses;details['last_complete_matched_interval']=tier.get('completed_intervals');return details

def assessment_checks(tier):
    from kenggo.recorded_statistics import _estimate
    from kenggo.progressive_evaluation import paired_losses, decide
    reports={};details={}
    for arm,r in tier['assessments'].items():
        a=read(r);refs(a);reports[arm]={};details[arm]={}
        wanted=tier['parent_checkpoint'] if arm=='parent' else tier['selected'][arm]
        eq(a['checkpoint']['sha256'],wanted['sha256'],'Assessment checkpoint')
        for name,rr in a['reports'].items():
            report=read(rr);reports[arm][name]=report
            prep=read(tier['preparation']);eq(report['diagnostic_sha256'],prep['assessments'][name]['sha256'],'Assessment diagnostic')
            diagnostic_data,arrays=diagnostic(prep['assessments'][name],'assessment')
            index={row['row_key']:i for i,row in enumerate(diagnostic_data['rows'])}
            eq(set(index),{row['row_key'] for row in report['predictions']},'Assessment complete keys')
            eq(len(index),len(report['predictions']),'Repeated assessment row')
            for row in report['predictions']:
                i=index[row['row_key']]
                for k in ('policy_valid','value_valid'):eq(row[k],arrays[k][i].item(),'Assessment mask')
                for k,mask in (('action','policy_valid'),('value','value_valid')):
                    if row[mask]:eq(row[k],arrays[k][i].item(),'Assessment target')
            rows=[r for r in report['predictions'] if r['source_rank_mover'] in RANKS]
            for metric in ('nll','top1','value_mse'):
                mask='value_valid' if metric=='value_mse' else 'policy_valid'
                estimates=_estimate(rows,{r['row_key']:r[metric] for r in rows if r[mask]},RANKS,mask)
                close(estimates,report['report']['overall']['metrics'][metric],name+'/'+metric)
            details[arm][name]={k:report[k] for k in ('nll','top1','value_mse','rows','evaluated_rows','excluded_rank_rows','policy_count','value_count')}
    if 'comparison' in tier:
        saved=read(tier['comparison'])
        recomputed={arm:{name:paired_losses(candidate,reports[arm][name]) for name,candidate in reports['candidate'].items()} for arm in ('control','parent')}
        close(recomputed,saved['comparisons'],'Saved paired loss comparison');eq(decide(recomputed),saved['decision'],'Saved decision');eq(tier['decision'],saved['decision'],'Tier decision')
        details['decision']=saved['decision'];details['paired']=recomputed
        details['gate_limitation']='Pinned exploratory gate blocks a supported clear regression when its lower endpoint exceeds +0.02; this is not proof that every upper endpoint is below +0.02 and is not a promotion gate.'
    else:details['comparison']='unavailable; no new inference or fabricated comparison'
    return details

def pages():
    from kenggo.continuation_data import reference
    from kenggo.progressive_review import export
    original=PROD/'review.html'
    result={'original_page':ref(original) if original.exists() else None,'production_costs':SUMMARY['costs']}
    provisional=REC/'pre-export-summary.json'
    if provisional.exists():
        before=load(provisional)['costs'];result['pre_export_costs']=before
        result['settlement_cpu_difference_seconds']=SUMMARY['costs']['cpu_seconds']-before['cpu_seconds']
    new=OUT/'review-settled-production.html'
    export(reference(PROD/'summary.json'),new)
    result['owner_page']=ref(new)
    text=new.read_text();match=re.search(r'<script id="learning-data" type="application/json">(.*?)</script>',text,re.S)
    if not match:raise ValueError('Shared learning component absent')
    data=json.loads(match.group(1));current=data['runs'][-1]
    if len(data['runs'])<2:raise ValueError('Prior run history absent')
    for s in ('training-monitor','stopping-validation','selected checkpoint','masked','premature'):
        if s.lower() not in text.lower():raise ValueError('Page explanation missing '+s)
    result['history_runs']=[{'id':r['id'],'description':r.get('description'),'notes':r.get('notes')} for r in data['runs']]
    result['current_training_models']=list(current['training']);result['verification_costs']='Excluded from page; separate final-resources.json owns cumulative totals.'
    write('page-evidence.json',result);return result

if __name__=='__main__':
    PLAN=load(ROOT/'configs/e2_progressive_training_v1.json');BASE=PLAN['parameters']['base_inputs'];RANKS=[f'{n}k' for n in range(12,0,-1)]+['1d']
    SUMMARY=load(OUT/'production-summary.json')
    persist();check('Source and recovery pins',lambda:{'source':source(),'pins':pins()})
    check('Stages, failures and recovered unknown exit',stage_checks)
    check('Frozen fixed comparison coverage',fixed_checks)
    ds={PLAN['inputs']['parent_summary']['sha256']:PLAN['inputs']['parent_summary']}
    for stage in SUMMARY['stages'].values():
        s=read(stage['summary'])
        if 'dataset_summary' in s: ds[s['dataset_summary']['sha256']]=s['dataset_summary']
    for r in ds.values():check('Dataset '+r['sha256'][:12],lambda r=r:dataset(r))
    if any('preparation' in t for t in SUMMARY['tiers']):check('Disjoint diagnostics and frozen assessment labels/masks',preparation_checks)
    else:unavailable('Stopping samples','Production did not publish a stopping-sample preparation')
    for t in SUMMARY['tiers']:
        if t['training']:check('Matched training tier '+str(t['tier']),lambda t=t:training_checks(t))
        else:unavailable('Training tier '+str(t['tier']),'No training pair was published')
        if t['assessments']:check('Saved assessments tier '+str(t['tier']),lambda t=t:assessment_checks(t))
        else:unavailable('Fixed assessment tier '+str(t['tier']),'No new fixed-assessment reports were published')
    check('Preserved original page and settled-production export',pages)
    check('Final unchanged source and pins',lambda:{'source':source(),'pins':pins()})
    write('verified-hashes.json',HASHES)
    RESULT.update(status='failed_checks' if any(c['status']=='failed' for c in RESULT['checks']) else 'saved_evidence_checks_finished',completed_utc=now().isoformat(),production_status=SUMMARY['status'],production_reason=SUMMARY.get('reason'),verified_file_count=len(HASHES),browser='pending separate offline QA',root_acceptance='pending; .16/.17/E2 remain open')
    persist()
