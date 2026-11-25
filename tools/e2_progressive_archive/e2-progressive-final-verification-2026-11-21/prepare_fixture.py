"""Browser harness fixture only. Synthetic curves are never study evidence."""
from common import *
from kenggo.continuation_data import reference
from kenggo.progressive_review import export
import re
fixture=OUT/'fixture';fixture.mkdir(exist_ok=False)
summary={'tiers':[],'status':'verification_fixture','reason':'Synthetic UI harness test; production is still running. No training occurred.','costs':{'ledger':{'charged_gpu_seconds':0},'elapsed_seconds':0,'cpu_seconds':0},'deadline_utc':DEADLINE.isoformat()}
(fixture/'summary.json').write_text(json.dumps(summary))
export(reference(fixture/'summary.json'),fixture/'empty-review.html')
page=(fixture/'empty-review.html').read_text();pattern=r'(<script id="learning-data" type="application/json">)(.*?)(</script>)'
m=re.search(pattern,page,re.S);data=json.loads(m[2]);run=data['runs'][-1]
diag={'nll':2.5,'top1':.3,'value_mse':.7,'rows':128,'policy_count':128,'value_count':120,'aggregate_ranks':['5k'],'diagnostic_sha256':'synthetic-only'}
run['training']['fixture-control']={'points':[{'step':32,'epoch':1,'updates':32,'examples':4096,'policy_count':4096,'value_count':4000,'nll':2.4,'value_mse':.65}],
 'updates':32,'presentations':4096,'rows':128,'checkpoint':None,'status':'partial','window':'synthetic UI fixture',
 'checkpoints':[{'step':0,'epoch':0,'lifetime_updates':10,'training_monitor':diag,'validation':diag,'new_raw_best':True,'meaningful_improvement':True},
 {'step':32,'epoch':1,'lifetime_updates':42,'training_monitor':{**diag,'nll':2.4},'validation':{**diag,'nll':2.6},'new_raw_best':False,'meaningful_improvement':False}],
 'selected_step':0,'selected_round':0,'selected_lifetime_updates':10}
run['description']='Synthetic UI fixture; no model executed and no production result.'
page=page[:m.start(2)]+json.dumps(data).replace('<','\\u003c')+page[m.end(2):]
(fixture/'curves-review.html').write_text(page)
print(json.dumps({'fixture':str(fixture),'synthetic_only':True,'model_execution':False}))
