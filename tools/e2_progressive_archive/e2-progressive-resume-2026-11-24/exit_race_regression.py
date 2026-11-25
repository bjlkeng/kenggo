"""Reproduce the old exit-check race without starting or signalling a process."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

directory = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('resume', directory / 'resume.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
live = True


def exited():
    global live
    live = False
    return False


def process(pid):
    return {'state': 'R', 'group': 17, 'session': 17} if live else None


job = {'allocation_start_utc': r.now().isoformat(),
       'limits': {'wall_seconds': 10000, 'cpu_seconds': 10000, 'output_bytes': 10000}}
study = SimpleNamespace(check=lambda: {'old_scope_cpu_upper_seconds': 0})
worker = SimpleNamespace(pid=17, alive=exited)
with patch.object(r.old, 'load', return_value=job), \
        patch.object(r.old, 'cgroup_pids', return_value={17}), \
        patch.object(r.old, 'proc', side_effect=process), \
        patch.object(r.old.original, '_output_bytes', return_value=0):
    if sys.argv[1:] == ['old']:
        r.old.adopt(study, worker, {'worker_cgroup': '/fixture'})
    else:
        exited()
        assert r.live_scope_pids('/fixture') == set()
        print('Normal process exit is settled even if the scope snapshot still listed the exited leader.')
