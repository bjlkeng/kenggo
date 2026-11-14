"""Run frozen, CPU-only saved-artifact checks against the existing SL ledger."""

import argparse
import os
from pathlib import Path

from kenggo.checkpoints import file_hash, read_json, save_json
from kenggo.pilot_jobs import JobLimits, run_job
from kenggo.resource_ledger import BudgetLedger

ROOT = Path.cwd().resolve()


def run(config_path, expected_hash, output):
    config = read_json(config_path, expected_hash)
    if config['schema'] != 'e2_post_pilot_checks_v1' or config['gpu_seconds'] != 0:
        raise ValueError('Only the frozen CPU-only post-pilot checks may run')
    pilot_ref = config['pilot_summary']
    pilot = read_json(ROOT / pilot_ref['path'], pilot_ref['sha256'])
    if pilot['status'] != 'completed':
        raise ValueError('The pilot must complete before these checks')
    ledger = BudgetLedger.open(ROOT / config['ledger'], budget_id='e0-sl-pilot-v1', gpu_seconds=86400)
    if ledger.snapshot()['state']['reservations']:
        raise ValueError('The pilot has unfinished resource reservations')
    for job in config['jobs']:
        if file_hash(ROOT / job['script']['path']) != job['script']['sha256']:
            raise ValueError('A post-pilot check script changed')
        JobLimits(**job['limits'])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'config.json', config)
    jobs = []
    try:
        for job in config['jobs']:
            directory = output / job['id']
            arguments = [str(directory) if item == '{output}' else str(directory / item.removeprefix('{output}/'))
                         if item.startswith('{output}/') else item for item in job['arguments']]
            command = [str(ROOT / '.venv/bin/python'), str(ROOT / job['script']['path']), *arguments]
            result = run_job(command, directory, cwd=ROOT, limits=JobLimits(**job['limits']), ledger=ledger,
                run_id=output.name + '-' + job['id'], arm_shares={'shared': 1},
                environment=os.environ | {'CUDA_VISIBLE_DEVICES': ''},
                metadata={'purpose': 'post_pilot_saved_artifact_validation', 'config_sha256': expected_hash,
                          'script_sha256': job['script']['sha256'], 'pilot_summary_sha256': pilot_ref['sha256']})
            jobs.append(result)
            if result['status'] != 'completed':
                raise RuntimeError('Post-pilot check failed: ' + job['id'])
        save_json(output / 'summary.json', {'schema': config['schema'], 'status': 'completed',
                  'config_sha256': expected_hash, 'jobs': jobs, 'ledger': ledger.snapshot()})
    except BaseException as error:
        save_json(output / 'failure.json', {'status': 'failed', 'reason': type(error).__name__ + ': ' + str(error),
                  'config_sha256': expected_hash, 'jobs': jobs, 'ledger': ledger.snapshot()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run(args.config, args.sha256, args.output)
