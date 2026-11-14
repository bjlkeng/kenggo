"""Inspect trained rank parameters and existing data masks without new inference."""

import argparse
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import time

import pyarrow.parquet as pq
import torch

from kenggo.checkpoints import file_hash, read_json, save_json
from kenggo.kgs_source import RANKS

ROOT = Path.cwd().resolve()


def pinned(ref):
    return read_json(ROOT / ref['path'], ref['sha256'])


def checkpoint(stage, ref):
    path = Path(stage['directory']) / 'worker' / ref['path']
    manifest = read_json(path, ref['sha256'])
    assert manifest['config_sha256'] == stage['config']['sha256']
    assert manifest['provenance_sha256'] == stage['provenance']['sha256']
    state = path.parent / manifest['file']
    assert state.name == 'state.pt' and file_hash(state) == manifest['sha256']
    payload = torch.load(BytesIO(state.read_bytes()), map_location='cpu', weights_only=True)
    assert payload['step'] == manifest['step']
    assert payload['config_sha256'] == manifest['config_sha256']
    assert payload['provenance_sha256'] == manifest['provenance_sha256']
    embedding = payload['model']['rank_embedding.weight']
    assert embedding.device.type == 'cpu' and embedding.shape == (len(RANKS) + 1, 16)
    assert torch.isfinite(embedding).all()
    return embedding


def run(directory, output):
    started, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    main = read_json(directory / 'summary.json', file_hash(directory / 'summary.json'))
    assert main['status'] == 'completed' and not torch.cuda.is_initialized()
    plan = pinned(main['plan'])
    ref = plan['inputs']['examples']
    manifest = pinned(ref)
    base = (ROOT / ref['path']).parent
    rows, visible, hidden, sides, accounts = 0, Counter(), Counter(), defaultdict(set), defaultdict(set)
    for shard in manifest['artifacts']['train']:
        path = (base / shard['metadata']).resolve()
        assert path.is_relative_to(base.resolve() / 'train')
        assert file_hash(path) == shard['metadata_sha256']
        table = pq.read_table(path, columns=['role', 'source_url', 'mover', 'source_rank_mover',
            'source_rank_opponent', 'group_mover', 'rank_hidden'])
        assert len(table) == shard['rows']
        for row in table.to_pylist():
            assert row['role'] == 'train'
            rows += 1
            rank = row['source_rank_mover']
            sides[rank].add((row['source_url'], row['mover']))
            accounts[rank].add(row['group_mover'])
            for side, rank in enumerate((rank, row['source_rank_opponent'])):
                if row['rank_hidden'][side]:
                    hidden[side] += 1
                    visible['unknown'] += 1
                else:
                    visible[rank] += 1
    assert rows == manifest['counts']['train']['examples'] == 22916
    assert hidden[0] == manifest['counts']['train']['hidden_mover_rank']
    assert hidden[1] == manifest['counts']['train']['hidden_opponent_rank']
    embeddings, changes = {}, {}
    for arm in ('conditioned', 'unconditioned'):
        stage = main['stages'][arm]
        training = pinned(stage['summary'])
        initial_ref = next(ref for ref in training['checkpoints'] if ref['step'] == 0)
        initial, final = checkpoint(stage, initial_ref), checkpoint(stage, training['final_checkpoint'])
        embeddings[arm] = initial
        delta = (final - initial).abs().amax(dim=1).tolist()
        changes[arm] = dict(zip(('unknown',) + tuple(RANKS), delta, strict=True))
        if arm == 'unconditioned':
            assert torch.equal(initial[1:], final[1:])
        for rank, difference in changes[arm].items():
            if arm == 'conditioned' and not visible[rank]:
                assert difference == 0
    assert torch.equal(embeddings['conditioned'], embeddings['unconditioned'])
    assert not torch.cuda.is_initialized()
    result = {'schema': 'e2_conditioning_inspection_v1', 'status': 'passed',
        'scope': 'Post hoc inspection of fixed checkpoints and training metadata; no model query or parameter change.',
        'training_rows': rows, 'training_passes': manifest['counts']['train']['passes'],
        'hidden_mover_rows': hidden[0], 'hidden_opponent_rows': hidden[1],
        'visible_rank_inputs': dict(visible),
        'training_by_mover_rank': {rank: {'game_sides': len(sides[rank]), 'mover_accounts': len(accounts[rank])} for rank in RANKS},
        'maximum_absolute_embedding_change': changes,
        'new_neural_queries': 0, 'gpu_seconds': 0,
        'cpu_seconds': time.process_time() - cpu, 'wall_seconds': time.monotonic() - started,
        'pilot_summary_sha256': file_hash(directory / 'summary.json'), 'script_sha256': file_hash(Path(__file__))}
    save_json(output / 'summary.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.run, args.output)
