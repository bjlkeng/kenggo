"""Read-only checks of the completed dataset; retain a separate audit receipt."""
import importlib.util
import json
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU, (600, 610))
ROOT=Path('/home/bojji/code/kenggo')
OUT=ROOT/'runs/e2-progressive-resume-2026-11-24'
started=time.monotonic(); cpu=time.process_time()
result={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running', 'checks':[]}
def module(name, file):
    spec=importlib.util.spec_from_file_location(name,file); obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
try:
    rec=module('audit_old_recovery',ROOT/'runs/e2-progressive-recovery-2026-11-21/recovery.py')
    rec.verify_manifest(ROOT/'runs/e2-progressive-recovery-2026-11-21/manifest-pass3.json','2c94dfb5ac5e06486cd43e18d3826c841d2fa32deffc446b0e58651c60074888')
    rec.source_frozen()
    result['checks'].append('frozen source and all original recovery manifest pins')
    plan=rec.load(ROOT/'configs/e2_progressive_training_v1.json'); stage=ROOT/'runs/e2-progressive-2026-11-21/tier1-growth'
    saved=rec.validate_growth(stage/'worker',rec.load(stage/'config.json')['inputs']['request'],plan['parameters']['base_inputs'])
    result['checks'].append('growth publication, original request, code pins, shard hashes and original normalizer lineage')
    sys.path.insert(0,str(ROOT/'runs/e2-progressive-final-verification-2026-11-21'))
    verify=module('audit_saved_dataset',ROOT/'runs/e2-progressive-final-verification-2026-11-21/verify.py')
    verify.BASE=plan['parameters']['base_inputs']
    result['dataset']=verify.dataset(saved['dataset_summary'])
    result['checks'].append('all fitting row identities, complete game counts, no duplicate positions, masks, scalar validity, komi support and value outcomes')
    rec.source_frozen()
    result['status']='passed'
    result['limits']=['No training or inference','No full board/legal-mask replay','Final and confirmation targets hashed only; never decoded','Original worker exit remains unknown; accepting saved publication only']
except BaseException as error:
    result['status']='failed';result['error']=type(error).__name__+': '+str(error);result['traceback']=traceback.format_exc()
finally:
    result['wall_seconds']=time.monotonic()-started;result['cpu_seconds']=time.process_time()-cpu;result['completed_utc']=datetime.now(timezone.utc).isoformat()
    with (OUT/'audit-saved-growth.json').open('x') as f: json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['dataset','traceback']},indent=2))
raise SystemExit(0 if result['status']=='passed' else 1)
