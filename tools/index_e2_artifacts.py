"""Index completed pilot artifacts without decoding data or rerunning models."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

ROOT = Path.cwd().resolve()


def reference(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Evidence must be a local repository artifact')
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def verify(ref):
    actual = reference(ROOT / ref['path'])
    if actual['sha256'] != ref['sha256']:
        raise ValueError('Changed artifact: ' + ref['path'])
    return actual


def index(run, output):
    started, cpu = time.monotonic(), time.process_time()
    run = run.resolve()
    main = json.loads((run / 'summary.json').read_text())
    if main['status'] != 'completed' or main['ledger']['state']['reservations']:
        raise ValueError('Only a completed, settled pilot may be indexed')
    sources, stages = set(), []
    for name, stage in main['stages'].items():
        directory = Path(stage['directory']).resolve()
        if directory != run / name:
            raise ValueError('Stage directory differs from its owner')
        for key in ('config', 'provenance', 'summary'):
            if not (ROOT / stage[key]['path']).resolve().is_relative_to(directory):
                raise ValueError('Stage evidence leaves its owner')
            verify(stage[key])
        provenance = json.loads((ROOT / stage['provenance']['path']).read_text())
        source = provenance['source']
        patch = reference(directory / 'source.patch')
        if patch['sha256'] != source['patch_sha256']:
            raise ValueError('Source patch differs from provenance')
        sources.add((source['commit'], source['patch_sha256']))
        tracked = json.loads((directory / 'run.json').read_text())
        if not tracked.get('tracking_finished') or tracked['status'] not in ('completed', 'partial'):
            raise ValueError('W&B tracking did not finish')
        stages.append({'name': name, 'status': tracked['status'], 'wandb_url': stage['wandb_url'],
                       'config': stage['config'], 'provenance': stage['provenance'],
                       'summary': stage['summary'], 'job': reference(directory / 'worker/job.json')})
    if len(sources) != 1:
        raise ValueError('Pilot stages used different source snapshots')
    artifacts = []
    for path in sorted(run.rglob('*')):
        relative = path.relative_to(run)
        if 'wandb' in relative.parts or '.writer.lock' in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError('Unexpected evidence symlink: ' + str(relative))
        if path.is_file():
            artifacts.append(reference(path))
    commit, patch = next(iter(sources))
    result = {'schema': 'e2_base_artifact_index_v1', 'pilot': reference(run / 'summary.json'),
              'plan': verify(main['plan']), 'source': {'commit': commit, 'patch_sha256': patch},
              'stages': stages, 'artifacts': artifacts, 'files': len(artifacts),
              'bytes': sum(item['bytes'] for item in artifacts),
              'excluded': 'W&B internal telemetry/cache and writer lock; local run state and metrics are retained.',
              'costs': {'cpu_seconds': time.process_time() - cpu, 'wall_seconds': time.monotonic() - started,
                        'gpu_seconds': 0}, 'script': reference(Path(__file__))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'files': result['files'], 'bytes': result['bytes'], 'output': reference(output),
                      'costs': result['costs']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    index(args.run, args.output)
