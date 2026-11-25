from common import *
import subprocess
folder=OUT/'fixture/attempt3';folder.mkdir()
fd=os.open(OUT/'tmp',os.O_RDONLY|os.O_DIRECTORY)
env=dict(os.environ,TMPDIR=f'/proc/{os.getpid()}/fd/{fd}',E2_QA_OUTPUT=str(folder),NODE_PATH='/home/bojji/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules')
with (OUT/'fixture-browser-attempt3.log').open('xb') as log:
    r=subprocess.run(['/home/bojji/.nvm/versions/node/v25.9.0/bin/node',str(OUT/'browser.cjs'),str(OUT/'fixture/empty-review.html'),str(OUT/'fixture/curves-review.html')],env=env,stdout=log,stderr=log,timeout=120)
print(r.returncode)
os.close(fd)
