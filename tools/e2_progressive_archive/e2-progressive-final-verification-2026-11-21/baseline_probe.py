from pathlib import Path
import common,verify,json
verify.PLAN=common.load(common.ROOT/'configs/e2_progressive_training_v1.json');verify.BASE=verify.PLAN['parameters']['base_inputs'];verify.RANKS=[f'{n}k' for n in range(12,0,-1)]+['1d']
from kenggo.continuation_data import validate_lineage
from kenggo.progressive_training import dataset_inputs
m,n=validate_lineage(dataset_inputs(verify.PLAN['inputs']['parent_summary'],verify.BASE))
f=verify.fixed_checks()
common.write('preparation-baseline-check.json',{'scope':'Read-only baseline contract check during preparation; production final verification remains pending','normalizer_lineage':'passed','original_train_rows':m['counts']['train']['examples'],'fixed_comparison':f,'cpu_seconds_cgroup_through_this_probe':common.cpu(common.own_group()),'new_training':False,'new_inference':False})
print(json.dumps({'lineage':'passed','fixed_counts':{k:f[k] for k in ('games','rows','in_scope','excluded')},'cpu_seconds':common.cpu(common.own_group()),'output_bytes':common.size(common.OUT)}))
