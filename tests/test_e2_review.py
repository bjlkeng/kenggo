import importlib.util
from pathlib import Path
import tempfile
import unittest

from kenggo.checkpoints import file_hash, save_json
from kenggo.go import Game, PASS

spec = importlib.util.spec_from_file_location("review_collector", Path(__file__).resolve().parents[1] / "tools/build_e2_review.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def ref(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    return {"path": str(path), "sha256": save_json(path, value)}


class E2ReviewTests(unittest.TestCase):
    def test_pinned_results_replay_masks_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'pilot';run.mkdir()
            stages={};ledger={'charged_gpu_seconds':2.,'remaining_gpu_seconds':86398.,'state':{'reservations':{},'entries':[]}}
            for name in ('preparation','conditioned','unconditioned','evaluation-conditioned','evaluation-unconditioned','evaluation-human-reference','matches','analysis'):
                directory=run/name;worker=directory/'worker';worker.mkdir(parents=True)
                summary={'status':'completed'}
                if name in ('conditioned','unconditioned'):
                    summary.update(epoch_metrics=[{'epoch':1,'policy_loss':2.,'value_loss':.5}],completed_updates=1,presented_examples=2,
                        initial_tensor_sha256='same',row_keys_sha256='same',phase_seconds={},peak_cuda_allocated_bytes=1,checkpoint_selection='fixed_final_epoch')
                    ledger['state']['entries'].append({'run_id':'pilot-'+name,'gpu_seconds':1.,'cpu_seconds':1.,'wall_seconds':1.})
                elif name.startswith('evaluation-'):
                    artifacts={}
                    for kind in ('metrics','sampling','timing'):
                        path=worker/(kind+'.json');r=ref(path,{'synthetic':True})
                        artifacts[path.name]={'path':path.name,'sha256':r['sha256']}
                    summary['reports']=[{'variant':'ordinary','set':'population_development','artifacts':artifacts,'rows':1,'common_set_sha256':'same'}]
                elif name=='matches':
                    slot={'id':'s0','arm':'human_sl','profile_id':'weak_uniform','candidate_color':'B'}
                    other=dict(slot,id='s1')
                    game=Game.new(candidate_color='B').play('B',0).play('W',1).play('B',PASS)
                    record={'slot':slot,'state':game.action_record(),'snapshot':game.snapshot,'actual_actions':3,
                        'sgf_file':'s0.sgf','candidate_passes':[{'move_index':2,'verdict':'unjudgeable'}],
                        'candidate_profile':{'conditioning_rank':'5k'},'result':'U','status':'invalid','reason':'fixture_infrastructure',
                        'censored':True,'behavior':{'unresolved':True},'outcome_kind':'fault','winner':None,'disagreement':{},'fault':{},'context':{}}
                    match_dir=worker/'matches';match_dir.mkdir();sgf=match_dir/'s0.sgf';sgf.write_text('(;GM[1]SZ[19];B[aa];W[ba];B[])')
                    record_pin=ref(match_dir/'s0.json',record)['sha256']
                    auth={'records':{'s0':{'record_sha256':record_pin,'sgf_sha256':file_hash(sgf)}}}
                    summary.update(schedule_file_sha256=ref(worker/'schedule.json',{'slots':[slot,other]})['sha256'],
                        evidence_authorization_sha256=ref(worker/'evidence_authorization.json',auth)['sha256'],
                        records_replayed=1,recorded_games=1,unverified_slots=[])
                elif name=='analysis': summary['comparisons']={}
                stages[name]={'directory':str(directory),'summary':ref(worker/'summary.json',summary),
                    'config':ref(directory/'config.json',{}),'provenance':ref(directory/'provenance.json',{}),
                    'wandb_url':None,'job':{'costs':{'cpu_seconds':1.,'wall_seconds':1.}}}
            ref(run/'summary.json',{'status':'completed','matched_training':True,'stages':stages,'ledger':ledger,'plan':{}})
            result=module.collect(run)
            assert len(result['games'])==2
            game=result['games'][0]
            assert game['arm']=='human-reference' and game['source_arm']=='human_sl'
            assert len(game['frames'])==4 and game['frames'][1][0]==[[0,'B']]
            assert game['frames'][2][0]==[[1,'W']] and game['frames'][3][0]==[]
            assert game['passes'][0]['ply']==2 and game['conditioning_rank']=='5k'
            assert result['games'][1]['status']=='unfilled' and result['games'][1]['result']=='U'
            assert result['training']['conditioned']['costs']['gpu_seconds']==1
            (run/'matches/worker/matches/s0.sgf').write_text('tampered')
            try: module.collect(run)
            except ValueError as error: assert 'changed' in str(error)
            else: raise AssertionError('Changed SGF must be rejected')


if __name__ == "__main__":
    unittest.main()
