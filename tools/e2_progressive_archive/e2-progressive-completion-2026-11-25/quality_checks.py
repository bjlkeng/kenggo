"""Record bounded CPU-only repository checks, including failures."""
import json
import os
from pathlib import Path
import resource
import subprocess
import time

root=Path('/home/bojji/code/kenggo');out=root/'runs/e2-progressive-completion-2026-11-25'
env=dict(os.environ,UV_CACHE_DIR=str(root/'.cache/uv'),UV_PYTHON_INSTALL_DIR=str(root/'.cache/python'),UV_OFFLINE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
checks=[('lock',['uv','lock','--check'],60),('sync',['uv','sync','--locked','--check'],60),('smoke',['uv','run','--locked','python','-m','kenggo.smoke'],120),('tests',['uv','run','--locked','python','-m','unittest','discover','-s','tests','-v'],1800),('build',['uv','build','--no-sources'],120),('diff',['git','diff','--check'],60)]
results=[]
for name,command,limit in checks:
    start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (out/f'quality-{name}.log').open('w') as stream:
        try:
            code=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=limit).returncode
        except subprocess.TimeoutExpired:
            stream.write('\nCHECK TIMED OUT\n');code=124
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    results.append({'name':name,'command':command,'exit_code':code,'wall_seconds':time.monotonic()-start,'cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,'log':str(out/f'quality-{name}.log')})
    (out/'quality-checks.json').write_text(json.dumps({'checks':results,'status':'running' if len(results)<len(checks) else ('passed' if all(c['exit_code']==0 for c in results) else 'failed')},indent=2)+'\n')
    print(name,code,flush=True)
