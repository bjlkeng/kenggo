"""Check retained audit records and quarantine membership without reopening SGFs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from kenggo.kgs_eligibility import normalize_handle
from kenggo.kgs_source import player


def pinned(ref):
    path = Path(ref['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ref['sha256'], path
    return json.loads(path.read_text())


def run(config_path, config_pin, output):
    started, cpu = time.monotonic(), time.process_time()
    config = pinned({'path': str(config_path), 'sha256': config_pin})
    frozen = pinned(config['frozen'])
    root = Path(config['output'])
    execution = json.loads((root / 'summary.json').read_text())
    assert execution['execution_status'] == execution['job_status'] == 'completed'
    assert execution['full_execution_accounting']['budget_status'] == 'within_limits'
    summary = pinned(execution['audit_summary'])
    assert summary['status'] in ('completed', 'partial')
    assert summary['input_manifest_sha256'] == config['frozen']['sha256']
    records_path = root / 'job' / summary['records']['path']
    assert hashlib.sha256(records_path.read_bytes()).hexdigest() == summary['records']['sha256']
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    candidates = frozen['candidates']
    assert [r['source_url'] for r in records] == [c['archive_row']['url'] for c in candidates[:len(records)]]
    assert len(records) == summary['recorded_records'] == summary['attempted_records']
    assert summary['unattempted_records'] == len(candidates) - len(records)
    assert summary['attempted_unrecorded_records'] == 0
    assert summary['retrieved_records'] == sum(r['retrieval_status'] == 'retrieved' for r in records)
    assert summary['source_admitted_records'] == sum(r['source_admitted'] for r in records)
    groups = frozen['groups']
    reserved = set(pinned(frozen['inputs']['split_config'])['test_groups'])
    exclusions = pinned(frozen['inputs']['owner_exclusions'])['excluded_games']
    excluded_urls = {r['source_url'] for r in exclusions}
    excluded_bytes = {r['sgf_sha256'] for r in exclusions}
    rows = []
    for role, expected in frozen['candidate_counts'].items():
        selected = [r for r in records if r['provisional_role'] == role]
        admitted = [r for r in selected if r['source_admitted']]
        stats = summary['source_admitted_by_provisional_role'][role]
        assert len(admitted) == stats['games']
        handles = {r['metadata'][prop].lower() for r in admitted for prop in ('PB', 'PW') if r['metadata'][prop]}
        assert len(handles) == stats['distinct_handle_strings']
        rows.append({'role': role, 'candidates': expected, 'attempted': len(selected),
            'source_admitted': len(admitted), 'distinct_handles': len(handles),
            'explicit_passes': sum(r['counters'].get('explicit_passes', 0) for r in admitted),
            'two_pass_games': sum(r['final_two_explicit_passes'] for r in admitted)})
    for candidate, record in zip(candidates, records):
        row = candidate['archive_row']
        assert record['source_url'] not in excluded_urls
        assert record['provisional_role'] == candidate['provisional_role']
        assert record['input_manifest_sha256'] == config['frozen']['sha256']
        assert record['bundle_sha256'] == frozen['bundles'][row['page_key']]['artifact']['sha256']
        for side in ('black', 'white'):
            handle = normalize_handle(player(row[side])[0])
            assert groups[handle] not in reserved
        assert not record['training_ready'] and not record['replay_validated'] and not record['deduplicated']
        if record['source_admitted']:
            assert record['parse_status'] == 'parsed' and record['sgf_sha256'] not in excluded_bytes
            assert not any(record[k] for k in ('identity_valid', 'duration_targets_valid', 'dynamic_clock_inputs_valid'))
    exposure = pinned({'path': str(root / 'job' / summary['exposure']['path']), 'sha256': summary['exposure']['sha256']})
    assert exposure['source_ids'] == [c['archive_row']['url'] for c in candidates]
    assert not reserved.intersection(exposure['groups'])
    assert exposure['sgf_sha256'] == sorted({r['sgf_sha256'] for r in records if 'sgf_sha256' in r})
    assert not execution['ledger']['state']['reservations'] and execution['ledger']['charged_gpu_seconds'] == 0
    costs = {'cpu_seconds': time.process_time() - cpu, 'wall_seconds': time.monotonic() - started, 'gpu_seconds': 0}
    result = {'schema': 'e2_cached_coverage_verification_v1', 'status': 'passed',
        'execution_summary': {'path': str(root / 'summary.json'), 'sha256': hashlib.sha256((root / 'summary.json').read_bytes()).hexdigest()},
        'audit_status': summary['status'], 'records_checked': len(records), 'roles': rows,
        'scope': 'Retained JSON records, exact fixed prefix, role counts, hashes, quarantine and masks only. No SGF or model call.',
        'costs': costs, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    run(args.config, args.sha256, args.output)
