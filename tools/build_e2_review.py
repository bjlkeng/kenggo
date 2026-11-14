"""Build an offline review bundle from completed, pinned BASE outputs only."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from kenggo.checkpoints import file_hash, read_json
from kenggo.go import Game

ROOT = Path.cwd().resolve()


def read_owned(reference, owner):
    path = (ROOT / reference['path']).resolve()
    if not path.is_relative_to(owner.resolve()):
        raise ValueError(f'Artifact leaves its owning run: {path}')
    return read_json(path, reference['sha256'])


def collect(run):
    run = run.resolve()
    main_path = run / 'summary.json'
    main = read_json(main_path, file_hash(main_path))
    if main['status'] != 'completed' or not main['matched_training']:
        raise ValueError('A completed, matched pilot is required for this review bundle')
    stages, summaries, evidence = main['stages'], {}, []
    for name, stage in stages.items():
        directory = Path(stage['directory']).resolve()
        if directory != run / name:
            raise ValueError('Stage directory differs from its declared name')
        summaries[name] = read_owned(stage['summary'], directory)
        read_owned(stage['config'], directory)
        read_owned(stage['provenance'], directory)
        evidence.append({'label': name, 'path': stage['summary']['path'],
                         'sha256': stage['summary']['sha256'], 'url': stage['wandb_url']})
    evaluations = {}
    for name in ('conditioned', 'unconditioned', 'human-reference'):
        summary = summaries['evaluation-' + name]
        if summary['status'] != 'completed':
            raise ValueError('Recorded-position evaluation is incomplete')
        worker = run / ('evaluation-' + name) / 'worker'
        for report in summary['reports']:
            arm = name if report['variant'] == 'ordinary' else 'conditioned_unknown_ranks'
            artifacts = report['artifacts']
            outputs = {}
            for kind in ('metrics', 'sampling', 'timing'):
                ref = artifacts[kind + '.json']
                path = (worker / ref['path']).resolve()
                if not path.is_relative_to(worker):
                    raise ValueError('Evaluation artifact leaves its worker')
                outputs[kind] = read_json(path, ref['sha256'])
            outputs['rows'] = report['rows']
            outputs['common_set_sha256'] = report['common_set_sha256']
            evaluations.setdefault(report['set'], {})[arm] = outputs
    training = {arm: {key: summaries[arm][key] for key in
                ('epoch_metrics', 'completed_updates', 'presented_examples', 'initial_tensor_sha256',
                 'row_keys_sha256', 'phase_seconds', 'peak_cuda_allocated_bytes', 'checkpoint_selection')}
                | {'costs': stages[arm]['job']['costs']}
                for arm in ('conditioned', 'unconditioned')}
    ledger = main['ledger']
    if ledger['state']['reservations']:
        raise ValueError('The pilot still has unsettled allocations')
    for arm in training:
        entry = next(item for item in ledger['state']['entries'] if item['run_id'] == run.name + '-' + arm)
        training[arm]['costs']['gpu_seconds'] = entry['gpu_seconds']
    costs = {'gpu_seconds': ledger['charged_gpu_seconds'],
             'cpu_seconds': sum(item['cpu_seconds'] for item in ledger['state']['entries']),
             'wall_seconds': sum(item['wall_seconds'] for item in ledger['state']['entries']),
             'remaining_gpu_seconds': ledger['remaining_gpu_seconds']}
    match_summary = summaries['matches']
    worker = run / 'matches/worker'
    schedule = read_json(worker / 'schedule.json', match_summary['schedule_file_sha256'])
    authorization = read_json(worker / 'evidence_authorization.json', match_summary['evidence_authorization_sha256'])
    if (match_summary['records_replayed'] != match_summary['recorded_games']
            or match_summary.get('unverified_slots')):
        raise ValueError('Generated matches have not all passed independent replay')
    games = []
    for slot in schedule['slots']:
        identity = slot['id']
        if identity not in authorization['records']:
            games.append({'id': identity, 'arm': 'human-reference' if slot['arm'] == 'human_sl' else slot['arm'], 'source_arm': slot['arm'], 'opponent': slot['profile_id'],
                'candidate_color': slot['candidate_color'], 'result': 'U', 'status': 'unfilled',
                'reason': 'Unfilled under the fixed budget', 'censored': True, 'actual_actions': 0,
                'behavior': {'unresolved': True}, 'moves': [], 'frames': [[[], 'B', 0, 0, 0]], 'passes': [], 'sgf': ''})
            continue
        pins = authorization['records'][identity]
        record = read_json(worker / 'matches' / (identity + '.json'), pins['record_sha256'])
        sgf = (worker / 'matches' / record['sgf_file']).resolve()
        if not sgf.is_relative_to(worker / 'matches') or sgf.name != identity + '.sgf':
            raise ValueError('Match SGF leaves its declared record')
        if file_hash(sgf) != pins['sgf_sha256'] or record['slot'] != slot:
            raise ValueError('Match record or SGF changed')
        game = Game.new(candidate_color=slot['candidate_color'])
        if record['state']['origin'] != 'empty' or record['state']['initial'] != json.loads(json.dumps(game.action_record()['initial'])):
            raise ValueError('The review accepts ordinary empty-board games only')
        for color, action in record['state']['moves']:
            game = game.play(color, action)
        if game.snapshot != record['snapshot'] or len(game.moves) != record['actual_actions']:
            raise ValueError('Review board replay differs from the pinned match')
        previous, frames = '.' * 361, []
        for position in game.positions:
            board = ''.join(position.board)
            frames.append([[[index, color] for index, color in enumerate(board) if color != previous[index]],
                           position.to_play, position.captured_black, position.captured_white, position.pass_streak])
            previous = board
        games.append({key: record[key] for key in ('result', 'status', 'reason', 'censored', 'actual_actions',
                                                   'behavior', 'outcome_kind', 'winner', 'disagreement', 'fault', 'context')}
            | {'id': identity, 'arm': 'human-reference' if slot['arm'] == 'human_sl' else slot['arm'], 'source_arm': slot['arm'], 'opponent': slot['profile_id'],
               'candidate_color': slot['candidate_color'], 'conditioning_rank': record['candidate_profile'].get('conditioning_rank'), 'moves': record['state']['moves'], 'frames': frames,
               'passes': [dict(item, ply=item['move_index']) for item in record['candidate_passes']],
               'sgf': sgf.read_text(), 'record_sha256': pins['record_sha256'], 'sgf_sha256': pins['sgf_sha256']})
    return {'bundle_id': file_hash(main_path), 'title': 'E2 baseline pilot',
        'summary': 'One fixed training seed; exploratory development comparison and raw play against two weak policies.',
        'limitations': [
            'The roles have 19 and 11 accounts. Every proposed rank cell lacks promotion support; confidence intervals are inconclusive.',
            'The all-16-rank aggregate is unavailable. Rank labels are source metadata or conditioning inputs, not calibrated strength.',
            'Only development games were evaluated. Confirmation and final targets remain sealed.',
            'The 12-game panel uses uniform and capture-first policies. It does not qualify a raw opponent or establish broad playing strength.',
            'HumanSL uses raw temperature-1 policy, fixed KGS-era/time defaults, and full legality history. Training overlap is unknown.',
            'Match results retain unresolved and unfilled slots. Referee adjudication is distinct from candidate actions and human source results.',
            'CPU accounting excludes some tracking-service work. GPU allocation includes imports, loading, checkpoint work and idle time.'],
        'roles': [{'id': 'population_development', 'label': 'Population development'},
                  {'id': 'validation_development', 'label': 'Unseen-account development'}],
        'arms': [{'id': 'conditioned', 'label': 'Conditioned'}, {'id': 'unconditioned', 'label': 'Unconditioned'},
                 {'id': 'human-reference', 'label': 'HumanSL'},
                 {'id': 'conditioned_unknown_ranks', 'label': 'Conditioned: ranks hidden'}],
        'evaluations': evaluations, 'training': training, 'costs': costs, 'games': games, 'evidence': evidence,
        'paired': summaries['analysis']['comparisons'], 'match_summary': match_summary,
        'plan': main['plan'], 'ledger': ledger}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('template', type=Path)
    parser.add_argument('decision', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    data = collect(args.run)
    data['decision'] = json.loads(args.decision.read_text())
    data['summary'] = data['decision'].get('summary', data['summary'])
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    encoded = encoded.replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    template = args.template.read_text()
    if template.count('__PILOT_DATA_JSON__') != 1:
        raise ValueError('The template must have exactly one data marker')
    output = template.replace('__PILOT_DATA_JSON__', encoded)
    if len(output.encode()) > 32 * 1024**2:
        raise ValueError('Review page exceeds its 32 MiB allowance')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(output)
    details = {'schema': 'e2_review_export_v1', 'html': str(args.output), 'html_sha256': file_hash(args.output),
               'template_sha256': file_hash(args.template), 'collector_sha256': file_hash(Path(__file__)),
               'pilot_summary_sha256': data['bundle_id'], 'decision_sha256': file_hash(args.decision),
               'games': len(data['games']), 'cpu_seconds': time.process_time() - cpu,
               'wall_seconds': time.monotonic() - started, 'gpu_seconds': 0}
    with args.output.with_suffix('.manifest.json').open('x') as stream:
        json.dump(details, stream, indent=2)
    print(json.dumps(details))


if __name__ == '__main__':
    main()
