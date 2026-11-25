"""Original owned jobs with explicit stopping-data preparation and growth exclusion."""
import importlib.util
from pathlib import Path

from kenggo import data_expansion, data_growth, progressive_jobs
from kenggo.checkpoints import file_hash, read_json, save_json
from kenggo.continuation_data import path, read, reference
from kenggo.data_splits import read_membership

HERE = Path(__file__).resolve().parent
FIELDS = ('sgf_sha256', 'audit_sequence_sha256', 'trajectory_sha256')


def verify_runtime(config):
    manifest = read(config['inputs']['resume_manifest'])
    root = HERE.parents[1]
    for filename in ('recovered_jobs.py', 'composite-validation.py', 'resume.py'):
        current = HERE / filename
        if manifest['files'].get(str(current.relative_to(root))) != file_hash(current):
            raise ValueError('Reviewed recovery helper changed: ' + filename)


def supplement(ref):
    summary = read(ref)
    games = summary['accepted']
    ids = {item['source_url'] for item in games}
    if (summary['status'] != 'completed' or summary['yield_status'] != 'sufficient_yield'
            or len(games) != summary['usable_games'] or len(ids) != len(games) or len(games) < 2
            or any(item['membership']['role'] != 'validation_development'
                   or item['membership']['exclusions'] for item in games)
            or any(not item.get(field) for item in games for field in FIELDS)):
        raise ValueError('Stopping supplement must contain distinct admitted validation games')
    return ids, {field: {item[field] for item in games} for field in FIELDS}


def blocked_records(index, ids, hashes):
    return {row['source_url'] for row in index['records'] if row['source_url'] in ids
            or any(row.get(field) in values for field, values in hashes.items())}


def assert_fitting_disjoint(dataset, source_summary):
    refs = read(dataset)['artifacts']
    _, partitions = read_membership(path(refs['split_manifest']).parent, refs['split_manifest']['sha256'])
    index = read(refs['base_index'])
    ids, hashes = supplement(source_summary)
    blocked = blocked_records(index, ids, hashes)
    fitting = {row['source_url'] for row in partitions['train']}
    if fitting & blocked:
        raise ValueError('A stopping-validation source or duplicate entered fitting')
    return {'fitting_games': len(fitting), 'supplement_games': len(ids),
            'fitting_overlap': 0, 'dataset': dataset, 'supplement': source_summary,
            'method': 'Source, SGF, audit-sequence and full-trajectory metadata; no protected targets decoded'}


def augmented_memberships(original, ids, hashes, units, fingerprints, *args, **kwargs):
    fingerprints = {field: set(values) | hashes.get(field, set()) for field, values in fingerprints.items()}
    for unit in units:
        if unit['status'] == 'canonical' and unit['source_url'] in ids:
            entry = unit['entry']
            entry['exclusions'] = sorted(set(entry['exclusions']) | {'reserved_stopping_validation'})
    return original(units, fingerprints, *args, **kwargs)


def prepare(config, output, *, check_budget=lambda: None):
    spec = importlib.util.spec_from_file_location('composite_validation_job', HERE / 'composite-validation.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    ref = config['inputs']['composite_request']
    previous = [previous_ref for key, previous_ref in sorted(config['inputs'].items())
                if key.startswith('previous_preparation_')]
    # Keep the builder's immutable publication separate from the original job wrapper's summary.
    return helper.build(path(ref), ref['sha256'], Path(output) / 'composite', tier=config['parameters']['tier'],
                        previous=previous, check_budget=check_budget)


def run(config_path, config_pin, provenance_path, provenance_pin, output):
    config = read_json(config_path, config_pin)
    verify_runtime(config)
    old_prepare = progressive_jobs.prepare
    old_memberships, old_promotions = data_expansion._new_memberships, data_growth._promotion_ids
    try:
        if config['purpose'] == 'progressive-preparation':
            progressive_jobs.prepare = prepare
        if config['purpose'] == 'progressive-growth':
            source = config['inputs']['validation_supplement']
            ids, hashes = supplement(source)
            request = read(config['inputs']['request'])
            parent = read(request['parent_summary'])
            index = read(parent['artifacts']['base_index'])
            blocked = blocked_records(index, ids, hashes)
            assert_fitting_disjoint(request['parent_summary'], source)
            data_expansion._new_memberships = lambda units, fingerprints, *args, **kwargs: augmented_memberships(
                old_memberships, ids, hashes, units, fingerprints, *args, **kwargs)
            data_growth._promotion_ids = lambda *args, **kwargs: [
                key for key in old_promotions(*args, **kwargs) if key not in blocked]
        result = progressive_jobs.run(config_path, config_pin, provenance_path, provenance_pin, output)
        if config['purpose'] == 'progressive-growth':
            evidence = assert_fitting_disjoint(result['dataset_summary'], config['inputs']['validation_supplement'])
            save_json(Path(output) / 'stopping-exclusion-check.json', evidence)
            result['stopping_exclusion_check'] = reference(Path(output) / 'stopping-exclusion-check.json')
            save_json(Path(output) / 'summary.json', result, replace=True)
        return result
    finally:
        progressive_jobs.prepare = old_prepare
        data_expansion._new_memberships, data_growth._promotion_ids = old_memberships, old_promotions


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'provenance', 'output'):
        parser.add_argument(name)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--provenance-sha256', required=True)
    args = parser.parse_args()
    try:
        run(args.config, args.config_sha256, args.provenance, args.provenance_sha256, args.output)
    except BaseException as error:
        save_json(Path(args.output) / 'failure.json', {'status': 'failed', 'type': type(error).__name__, 'reason': str(error)})
        raise


if __name__ == '__main__':
    main()
