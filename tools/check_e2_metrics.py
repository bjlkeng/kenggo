"""Independently recompute saved BASE metrics from already-opened development shards."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
from kenggo.checkpoints import file_hash, read_json, save_json
from kenggo.examples import read_shards

ROOT=Path.cwd().resolve()
ROLES=('population_development','validation_development')


def pinned(ref):
    return read_json(ROOT/ref['path'],ref['sha256'])


def expected_number(actual, expected):
    if expected is None or math.isnan(expected):
        assert actual=={'status':'undefined','value':None},(actual,expected)
    elif math.isinf(expected):
        assert actual=={'status':'positive_infinity' if expected>0 else 'negative_infinity','value':None},(actual,expected)
    else:
        assert actual['status']=='finite' and math.isclose(actual['value'],expected,rel_tol=1e-10,abs_tol=1e-10),(actual,expected)


def side_weights(rows, indices, ranks):
    groups=defaultdict(list)
    for index in indices:
        row=rows[index];groups[(row['source_rank_mover'],row['source_url'],row['mover'])].append(index)
    counts={rank:sum(key[0]==rank for key in groups) for rank in ranks}
    if any(not counts[rank] for rank in ranks):return None
    return {index:1/(len(ranks)*counts[key[0]]*len(members)) for key,members in groups.items() for index in members}


def weighted_mean(values, weights):
    if weights is None:return None
    with np.errstate(invalid='ignore'):
        value=np.sum(np.asarray([values[index]*weight for index,weight in weights.items()],dtype=np.float64))
    return None if np.isnan(value) else float(value)


def groups(report, rows, ranks):
    yield 'overall',report['overall'],list(range(len(rows))),ranks
    for rank,group in report['by_rank'].items():
        yield 'rank/'+rank,group,[i for i,row in enumerate(rows) if row['source_rank_mover']==rank],[rank]
    for phase,group in report['by_phase'].items():
        yield 'phase/'+phase,group,[i for i,row in enumerate(rows) if row['phase']==phase],ranks
    for clock,group in report['by_clock'].items():
        yield 'clock/'+clock,group,[i for i,row in enumerate(rows) if row['clock']==clock],ranks


def check_report(report, rows, losses, ranks, predictions, value_mask, policy_mask):
    checked=0
    for label,group,indices,scope in groups(report,rows,ranks):
        for metric,values in losses.items():
            mask=value_mask if metric=='value_mse' else policy_mask
            weights=side_weights(rows,[i for i in indices if mask[i]],scope)
            expected_number(group['metrics'][metric]['estimate'],weighted_mean(values,weights))
            assert group['metrics'][metric]['uncertainty']['status']=='inconclusive'
            assert group['metrics'][metric]['uncertainty']['interval95'] is None
            checked+=1
        weights=side_weights(rows,[i for i in indices if value_mask[i]],scope)
        for name,values in predictions.items():
            bins=group['value_calibration'][name]['bins']
            memberships=np.minimum(9,np.searchsorted(np.asarray((-1.,-.8,-.6,-.4,-.2,0.,.2,.4,.6,.8,1.)),values,side='right')-1)
            for b,observed in enumerate(bins):
                selected=[i for i in indices if value_mask[i] and memberships[i]==b]
                assert observed['raw_count']==len(selected)
                mass=None if weights is None else math.fsum(weights[i] for i in selected)
                if mass is None:assert observed['weighted_mass'] is None
                else:assert math.isclose(observed['weighted_mass'],mass,rel_tol=1e-10,abs_tol=1e-10)
                conditional=None if not mass else {i:weights[i]/mass for i in selected}
                expected_number(observed['mean_prediction'],weighted_mean(values,conditional))
                expected_number(observed['mean_result'],weighted_mean(np.asarray([row['value'] for row in rows]),conditional))
                checked+=3
    return checked


def run(run_directory, output):
    started,cpu=time.monotonic(),time.process_time()
    run_directory=run_directory.resolve()
    main=read_json(run_directory/'summary.json',file_hash(run_directory/'summary.json'))
    assert main['status']=='completed'
    plan=pinned(main['plan']);refs=plan['inputs'];checked=0;coverage={}
    analysis=pinned(main['stages']['analysis']['summary'])
    for role in ROLES:
        labels={}
        for arrays,metadata in read_shards((ROOT/refs['examples']['path']).parent,refs['examples']['sha256'],role):
            for index,meta in enumerate(metadata):
                assert meta['role']==role and meta['row_key'] not in labels
                main_seconds=float(arrays['scalars'][index,9])
                assert not arrays['scalar_valid'][index,9] or 600 <= main_seconds <= 1800
                clock='missing' if not arrays['scalar_valid'][index,9] else '10-<20' if main_seconds<1200 else '20-30'
                phase='1-60' if meta['ply']<60 else '61-180' if meta['ply']<180 else '181+'
                labels[meta['row_key']]=dict(meta,phase=phase,clock=clock,action=int(arrays['action'][index]),
                    value=float(arrays['value'][index]),value_valid=bool(arrays['value_valid'][index]),
                    policy_valid=bool(arrays['policy_valid'][index]),legal=arrays['legal'][index].copy())
        spec=next(item for item in plan['parameters']['common_sets'] if item['role']==role)
        common_path=run_directory/'preparation/worker'/spec['id']/'manifest.json'
        prepared=pinned(main['stages']['preparation']['summary'])
        common_ref=next(item['manifest'] for item in prepared['common_sets'] if item['spec']['role']==role)
        common=read_json(common_path,common_ref['sha256']);keys=[row['row_key'] for row in common['rows']]
        assert set(labels)==set(keys) and len(keys)==spec['expected_rows']
        rows=[labels[key] for key in keys];ranks=spec['aggregate_ranks']
        action=np.asarray([row['action'] for row in rows]);target=np.asarray([row['value'] for row in rows])
        policy_mask=np.asarray([row['policy_valid'] for row in rows]);value_mask=np.asarray([row['value_valid'] for row in rows])
        legal=np.asarray([row['legal'] for row in rows]);losses={};values_by_arm={}
        for arm in ('conditioned','unconditioned','human-reference'):
            stage=main['stages']['evaluation-'+arm];summary=pinned(stage['summary']);worker=Path(stage['directory'])/'worker'
            for report in [item for item in summary['reports'] if item['role']==role]:
                name=arm if report['variant']=='ordinary' else 'conditioned_unknown_ranks'
                ref=report['artifacts']['predictions.npz'];path=worker/ref['path'];assert file_hash(path)==ref['sha256']
                with np.load(path,allow_pickle=False) as archive:
                    assert archive['row_keys'].tolist()==keys
                    logs=archive['log_policy'];values=archive['values'];samples=archive['sampled_actions']
                    assert np.isneginf(logs[~legal]).all() and np.all(np.abs(np.exp(logs).sum(axis=1)-1)<=.0005)
                    assert np.isfinite(values).all() and np.all(np.abs(values)<=1)
                    assert samples.shape==(len(rows),32) and np.all((samples>=0)&(samples<=361))
                    assert legal[np.arange(len(rows))[:,None],samples].all()
                current={'nll':-logs[np.arange(len(rows)),action], 'top1':(np.argmax(logs,axis=1)==action).astype(float),
                         'value_mse':np.square(values-target)}
                ref=report['artifacts']['metrics.json'];metrics=read_json(worker/ref['path'],ref['sha256'])
                checked+=check_report(metrics,rows,current,ranks,{'model':values},value_mask,policy_mask)
                losses[name],values_by_arm[name]=current,values
        for comparison,details in analysis['comparisons'][spec['id']].items():
            candidate,control=comparison.split('_minus_')
            with np.errstate(invalid='ignore'):
                differences={name:losses[candidate][name]-losses[control][name] for name in losses[candidate]}
            checked+=check_report(pinned(details['report']),rows,differences,ranks,
                {'candidate':values_by_arm[candidate],'control':values_by_arm[control]},value_mask,policy_mask)
        coverage[role]={'rows':len(rows),'games':len({row['source_url'] for row in rows}),'arms':list(losses)}
    result={'schema':'e2_independent_saved_metric_check_v1','status':'passed','checked_metric_and_calibration_values':checked,
        'coverage':coverage,'method':'Independent NumPy rank/game-side/row means and value-bin calculation from existing development shards.',
        'new_neural_queries':0,'gpu_seconds':0,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-started,
        'pilot_summary_sha256':file_hash(run_directory/'summary.json'),'script_sha256':file_hash(Path(__file__))}
    save_json(output/'summary.json',result)
    print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();run(args.run,args.output)
