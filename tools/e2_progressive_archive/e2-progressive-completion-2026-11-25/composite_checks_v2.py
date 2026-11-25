"""Independent saved-array verification for composite stopping validation."""
from collections import Counter

import numpy as np

from kenggo.canonical_data import digest
from kenggo.encoding import INPUT_KEYS, apply_normalizer, array_schema
from kenggo.examples import ExampleConfig, LABEL_SCHEMA, examples_for_game, sealed_sources

SOURCE_SHA = 'd049b5fc9a001971213e6fb6e62d961c7356581d5cadcb2cc27dd974d156b092'
FINGERPRINTS = ('sgf_sha256', 'audit_sequence_sha256', 'trajectory_sha256')
ARRAY_SCHEMA = {**array_schema(0), **LABEL_SCHEMA}


def require(value, message):
    if not value:
        raise ValueError(message)


def unique(rows, name):
    result = {row['source_url']: row for row in rows}
    require(len(result) == len(rows), 'Repeated source in ' + name)
    return result


def equal_arrays(actual, expected, message):
    require(set(actual) == set(expected), message + ': array keys')
    for name, wanted in expected.items():
        got = actual[name]
        require(got.dtype == wanted.dtype and got.shape == wanted.shape
                and np.array_equal(got, wanted), message + ': ' + name)


def read_arrays(reference, check):
    with np.load(check.pin(reference), allow_pickle=False) as data:
        require(set(data.files) == set(ARRAY_SCHEMA), 'Diagnostic array schema changed')
        arrays = {name: data[name] for name in data.files}
    for name, (dtype, shape) in ARRAY_SCHEMA.items():
        array = arrays[name]
        require(array.dtype == np.dtype(dtype) and array.shape[1:] == shape
                and np.isfinite(array).all(), 'Invalid diagnostic array: ' + name)
    return arrays


def selection(inventory, ranks, tier, excluded, original_ids, supplemental_ids):
    by_source = unique(inventory, 'inventory')
    require(not set(original_ids) & set(supplemental_ids), 'Original and supplemental memberships overlap')
    seed = f'stopping-20261121-tier-{tier}'
    cells, selected = [], set()
    for komi in (1, 13):
        for rank in ranks:
            ordered = sorted((row for row in inventory if row['source_url'] not in excluded
                              and row['komi_halfpoints'] == komi and rank in row['ranks']),
                             key=lambda row: digest([seed, komi, rank, row['source_url']]))
            require(len(ordered) >= 2, f'Insufficient independent coverage: {rank}/{komi}')
            ids = [row['source_url'] for row in ordered[:2]]
            selected.update(ids)
            cells.append({'rank': rank, 'komi_halfpoints': komi, 'source_ids': ids, 'eligible_games': len(ordered)})
    return {'schema': 'composite_stopping_selection_v1', 'tier': tier, 'seed': seed, 'per_cell': 2,
            'ranks': list(ranks), 'komi_halfpoints': [1, 13], 'cells': cells,
            'cell_selections': 2 * len(cells), 'source_ids': sorted(selected), 'unique_games': len(selected),
            'expected_rows': sum(by_source[source]['rows'] for source in selected),
            'excluded_source_ids': sorted(excluded),
            'original_source_ids': sorted(selected & set(original_ids)),
            'supplemental_source_ids': sorted(selected & set(supplemental_ids)),
            'selection_uses_targets_or_predictions': False}


def original_inventory(check):
    import pyarrow.parquet as pq
    manifest = check.read(check.BASE['examples'])
    root = check.pin(check.BASE['examples']).parent
    games = {}
    for shard in manifest['artifacts']['validation_development']:
        rows = pq.read_table(check.pin({'path': str(root / shard['metadata']), 'sha256': shard['metadata_sha256']})).to_pylist()
        with np.load(check.pin({'path': str(root / shard['arrays']), 'sha256': shard['arrays_sha256']}), allow_pickle=False) as values:
            scalars = values['scalars']
        require(len(rows) == len(scalars) == shard['rows'], 'Original inventory shard counts differ')
        for i, row in enumerate(rows):
            komi = int(round(abs(float(scalars[i, 1])) * 2))
            game = games.setdefault(row['source_url'], {'source_url': row['source_url'], 'ranks': set(), 'rows': 0,
                                                        'komi_halfpoints': komi})
            require(game['komi_halfpoints'] == komi and row['role'] == 'validation_development', 'Original inventory role or komi changed')
            game['ranks'].add(row['source_rank_mover'])
            game['rows'] += 1
    return sorted(({**row, 'ranks': sorted(row['ranks'])} for row in games.values()), key=lambda row: row['source_url'])


def validate_unit(unit, record, plan):
    member = unit['membership']
    source = unit['source_url']
    gate, metadata, actions = record['eligibility'], record['metadata'], record['actions']
    colors = [sum(move['color'] == color for move in actions) for color in ('B', 'W')]
    groups = [gate[side + '_group'] for side in ('black', 'white')]
    identities = [gate[side + '_identity_valid'] for side in ('black', 'white')]
    ranks = sorted({metadata[name] for name, count in zip(('BR', 'WR'), colors) if count})
    require(unit['status'] == 'canonical' and member['source_url'] == source == record['source']['source_url']
            and member['canonical_sha256'] == digest(record)
            and member['role'] == 'validation_development' and not member['exclusions']
            and member['duplicate_representative'] == source and member['duplicate_sources'] == [source]
            and member['final_excluded'] is True and record.get('sealed_final') is False
            and member['groups'] == groups and member['identity_valid'] == identities and not any(identities)
            and gate['training_eligible'] is True and not member['known_persona_colors'] and not member['adaptation_colors']
            and set(groups) & set(plan['validation_groups']) and not set(groups) & set(plan['test_groups'])
            and member['date'] == metadata['DT'] and plan['train_before'] <= member['date'] < plan['confirmation_from']
            and len(actions) > 0 and member['actions'] == member['policy_actions'] == len(actions)
            and member['actions_by_color'] == colors and sum(colors) == len(actions)
            and member['value_actions'] == (len(actions) if record['recorded_value_supported'] else 0)
            and record['replay_validated'] is True and record['replay_pending_cap'] is None
            and metadata['RU'] == 'Japanese' and metadata['KM'] == 6.5 and '11k' in ranks
            and not record['dynamic_clock_input_valid'] and not record['duration_valid'],
            'Supplemental membership differs from complete admitted validation game')
    from kenggo.kgs_source import RANKS
    require(set(ranks) <= set(RANKS), 'Supplemental rank outside original admission policy')
    for name, value in {'sgf_sha256': record['source']['sgf_sha256'],
                        'audit_sequence_sha256': record['audit']['sequence_fingerprint'],
                        'trajectory_sha256': record['trajectory_sha256']}.items():
        require(value and value == unit[name], 'Supplemental fingerprint differs: ' + name)
    return {'source_url': source, 'ranks': ranks, 'rows': len(actions), 'komi_halfpoints': 13}


def historical_sources(check, summary_refs):
    blocked, hashes = set(), {name: set() for name in FINGERPRINTS}
    for reference in summary_refs:
        summary = check.read(reference)
        index = check.read(summary['artifacts']['base_index'])
        exposure = check.read(summary['artifacts']['exposure'])
        require(index['canonical_sha256'] == summary['artifacts']['canonical']['sha256']
                and index['split_sha256'] == summary['artifacts']['split_manifest']['sha256']
                and index['exposure'] == summary['artifacts']['exposure'], 'Historical index pins differ')
        blocked.update(index.get('processed_source_ids', []))
        for field in ('records', 'rejected', 'study_rejections'):
            for row in index.get(field, []):
                blocked.add(row['source_url'])
                for name in hashes:
                    if row.get(name):
                        hashes[name].add(row[name])
        blocked.update(exposure['source_ids'])
        hashes['sgf_sha256'].update(exposure['sgf_hashes'])
        hashes['audit_sequence_sha256'].update(exposure['sequence_hashes'])
    return blocked, hashes


def normalized_examples(units, records, sources, plan, example_config, normalizer):
    examples = []
    for source in sources:
        examples.extend(examples_for_game(records[source], units[source]['membership'], plan,
                                          ExampleConfig(**example_config), komi_study=True))
    require(bool(examples), 'Supplemental selection is empty')
    arrays = {name: np.asarray([row[0 if name in INPUT_KEYS else 1][name] for row in examples], dtype=dtype)
                  .reshape((len(examples), *shape)) for name, (dtype, shape) in ARRAY_SCHEMA.items()}
    rows = []
    for inputs, _, metadata in examples:
        raw = inputs['scalars']
        rows.append({**metadata, 'nominal_clock': [float(raw[9]), float(raw[10]), int(raw[11])]
                     if inputs['scalar_valid'][9:12].all() else None,
                     'komi_halfpoints': int(round(2 * abs(float(raw[1]))))})
    require(int(arrays['policy_valid'].sum()) == sum(units[source]['membership']['policy_actions'] for source in sources)
            and int(arrays['value_valid'].sum()) == sum(units[source]['membership']['value_actions'] for source in sources)
            and not arrays['duration_valid'].any(), 'Supplemental replay mask counts differ')
    arrays.update(apply_normalizer({name: arrays[name] for name in INPUT_KEYS}, normalizer))
    return arrays, rows


def verify_original_component(component, check):
    import pyarrow.parquet as pq
    original = check.read(component['diagnostic'])
    check.eq(original['purpose'], 'epoch_validation', 'Original component purpose')
    check.eq(original['role'], 'validation_development', 'Original component role')
    check.eq(original['aggregate_ranks'], check.RANKS, 'Original component rank scope')
    check.eq(original['normalizer_sha256'], check.BASE['normalizer']['sha256'], 'Original component normalization pin')
    check.eq(original['inputs_are_normalized'], True, 'Original component normalization state')
    check.eq(original['data'], check.BASE, 'Original component data provenance')
    check.eq(original['source_ids'], component['source_ids'], 'Original component sources')
    actual = read_arrays(original['arrays'], check)
    manifest = check.read(check.BASE['examples'])
    normalizer = check.read(check.BASE['normalizer'])
    root = check.pin(check.BASE['examples']).parent
    selected = set(component['source_ids'])
    parts = {key: [] for key in ARRAY_SCHEMA}
    expected_rows = []
    for shard in manifest['artifacts']['validation_development']:
        rows = pq.read_table(check.pin({'path': str(root / shard['metadata']), 'sha256': shard['metadata_sha256']})).to_pylist()
        indices = np.asarray([i for i, row in enumerate(rows) if row['source_url'] in selected], dtype=int)
        if not len(indices):
            continue
        with np.load(check.pin({'path': str(root / shard['arrays']), 'sha256': shard['arrays_sha256']}), allow_pickle=False) as data:
            raw = {name: data[name][indices] for name in ARRAY_SCHEMA}
        normalized = apply_normalizer({name: raw[name] for name in INPUT_KEYS}, normalizer)
        for name in parts:
            parts[name].append(normalized[name] if name in INPUT_KEYS else raw[name])
        for offset, i in enumerate(indices):
            scalars = raw['scalars'][offset]
            expected_rows.append({**rows[i], 'nominal_clock': [float(scalars[9]), float(scalars[10]), int(scalars[11])]
                                  if raw['scalar_valid'][offset, 9:12].all() else None,
                                  'komi_halfpoints': int(round(2 * abs(float(scalars[1]))))})
    expected = {name: np.concatenate(values) for name, values in parts.items()}
    check.eq(original['rows'], expected_rows, 'Original component metadata differs from source shards')
    check.eq(original['excluded_rank_rows'], sum(row['source_rank_mover'] not in check.RANKS for row in expected_rows), 'Original component excluded ranks')
    equal_arrays(actual, expected, 'Original component normalized exactly once from original arrays')
    return actual, expected_rows


def verify(diagnostic, check, *, prior_sources=(), fixture_only=False):
    check.eq(diagnostic['version'], 'continuation_diagnostic_v1', 'Diagnostic version')
    check.eq(diagnostic['purpose'], 'epoch_validation', 'Composite purpose')
    check.eq(diagnostic['role'], 'validation_development', 'Composite role')
    check.eq(diagnostic['aggregate_ranks'], check.RANKS, 'Composite rank scope')
    check.eq(diagnostic['inputs_are_normalized'], True, 'Composite normalization state')
    check.eq(diagnostic['normalizer_sha256'], check.BASE['normalizer']['sha256'], 'Composite normalizer')
    composite = diagnostic['composite']
    check.eq(composite['version'], 'composite_stopping_validation_v1', 'Composite version')
    check.eq(composite['original_fitting_data_unchanged'], True, 'Composite fitting declaration')
    check.eq(composite['normalization'], {'normalizer': check.BASE['normalizer'], 'fitting_examples': check.BASE['examples'],
                                         'fitting_split': check.BASE['split']}, 'Original normalization fitting population')
    request = check.read(composite['request'])
    refs = request['inputs']
    check.eq(request['schema'], 'composite_stopping_validation_request_v1', 'Composite request version')
    check.eq(request['fixture_only'], fixture_only, 'Composite fixture scope')
    check.eq(check.read(refs['original_config']), check.PLAN, 'Original experiment config')
    check.eq(refs['fixed_comparison'], check.PLAN['inputs']['fixed_comparison'], 'Fixed comparison pin')
    parent = check.PLAN['inputs']['parent_summary']
    check.eq(check.pin(refs['parent_summary']).resolve(), check.pin(parent).resolve(), 'Original parent path')
    check.eq(refs['parent_summary']['sha256'], parent['sha256'], 'Original parent hash')
    check.eq(request['ranks'], check.RANKS, 'Request rank scope')
    if not fixture_only:
        check.eq(refs['source_summary']['sha256'], SOURCE_SHA, 'Reviewed source summary pin')
        check.eq(request['source_commit'], check.HEAD, 'Composite frozen source')
        check.eq(request['deadline_utc'], check.DEADLINE.isoformat(), 'Composite original deadline')
    for filename, pin in request['code_sha256'].items():
        check.pin({'path': filename, 'sha256': pin})
    base_manifest, normalizer = check.read(check.BASE['examples']), check.read(check.BASE['normalizer'])
    check.eq(normalizer['examples_sha256'], check.BASE['examples']['sha256'], 'Normalizer fitting examples')
    check.eq(base_manifest['split_sha256'], check.BASE['split']['sha256'], 'Original split pin')
    members = check.membership(check.BASE['split'])
    allowed = unique(members['validation_development'], 'original validation membership')
    split = check.read(check.BASE['split'])
    source_summary = check.read(refs['source_summary'])
    accepted = check.read(refs['source_accepted'])['games']
    units = unique(accepted, 'accepted supplement')
    check.eq(accepted, source_summary['accepted'], 'Independent accepted source records')
    require(source_summary['status'] == 'completed' and source_summary['yield_status'] == 'sufficient_yield'
            and source_summary['accepted_games'] == source_summary['usable_games'] == len(units)
            and not source_summary['training_performed'] and not source_summary['model_queries']
            and not source_summary['protected_targets_decoded'], 'Supplemental source phase is incomplete or changed')
    check.eq(source_summary['request'], refs['source_request'], 'Source request pin')
    check.eq(source_summary['original_deadline_utc'], request['deadline_utc'], 'Source original deadline')
    source_review = check.read(refs['source_review'])
    require(source_review['request_sha256'] == refs['source_request']['sha256']
            and source_review['verdict'] == 'no significant findings' and source_review['model'] == 'gpt-6-astra'
            and source_review['reasoning'] == 'max', 'Independent source review is absent or mismatched')
    source_phase = check.read(refs['source_phase'])
    require(source_phase['status'] == 'completed' and check.read(source_phase['source_summary'])['source_summary'] == refs['source_summary'],
            'Source phase completion differs')
    source_request = check.read(refs['source_request'])
    check.refs(source_request)
    check.eq(source_request['inputs']['parent_summary'], refs['parent_summary'], 'Source original parent lineage')
    check.eq(source_request['inputs']['grown_summary'], refs['grown_summary'], 'Source original growth lineage')
    records = {source: check.read(unit['canonical']) for source, unit in units.items()}
    supplemental = [validate_unit(units[source], records[source], split['plan']) for source in sorted(units)]
    blocked, fingerprints = historical_sources(check, (refs['parent_summary'], refs['grown_summary']))
    for source, unit in units.items():
        require(source not in blocked, 'Supplemental source was already exposed or fitted')
        for name in FINGERPRINTS:
            require(unit[name] not in fingerprints[name], 'Supplemental fingerprint overlaps historical corpus: ' + name)
    for name in FINGERPRINTS:
        require(len({unit[name] for unit in units.values()}) == len(units), 'Supplemental fingerprint duplicate: ' + name)
    exposure = check.read(refs['source_exposure'])
    require(all(unit['source_url'] in exposure['source_ids'] and unit['sgf_sha256'] in exposure['sgf_hashes']
                and unit['audit_sequence_sha256'] in exposure['sequence_hashes'] for unit in units.values()), 'Incomplete supplemental exposure')
    grown = check.read(refs['grown_summary'])
    fitting = unique(check.membership(grown['artifacts']['split_manifest'])['train'], 'grown fitting')
    fixed = check.read(refs['fixed_comparison'])
    fixed_ids = {source for common in fixed['common_sets'] for source in common['spec']['source_ids']}
    previous = {source for previous in composite['previous_preparations']
                for source in check.read(check.read(previous)['epoch_validation'])['source_ids']}
    check.eq(previous, set(prior_sources), 'Previous stopping exclusions omitted or changed')
    protected = sealed_sources(split) | sealed_sources(base_manifest)
    exclusions = set(fitting) | fixed_ids | previous | protected
    require(not set(units) & (set(fitting) | fixed_ids | protected | {row['source_url'] for rows in members.values() for row in rows}),
            'Supplemental membership overlaps fitting, fixed, protected or original games')
    original = original_inventory(check)
    check.eq(original, sorted(check.read(refs['original_inventory']), key=lambda row: row['source_url']), 'Original inventory differs from saved arrays')
    check.eq(set(unique(original, 'original inventory')), set(allowed), 'Original inventory source membership')
    require(all(row['rows'] == allowed[row['source_url']]['policy_actions'] and not allowed[row['source_url']]['exclusions']
                for row in original), 'Original inventory whole-game counts')
    inventory = sorted(original + supplemental, key=lambda row: row['source_url'])
    check.eq(check.read(composite['inventory']), inventory, 'Composite inventory differs from independent memberships')
    chosen = selection(inventory, check.RANKS, composite['tier'], exclusions, allowed, units)
    check.eq(check.read(composite['selection']), chosen, 'Deterministic 26-cell selection changed')
    check.eq(diagnostic['source_ids'], chosen['source_ids'], 'Composite selected whole games')
    check.eq(diagnostic['data'], {'original': check.BASE, 'supplemental': {
        'source_summary': refs['source_summary'], 'accepted': refs['source_accepted'], 'exposure': refs['source_exposure']}}, 'Per-component data provenance')
    barrier = check.read(composite['growth_exclusions'])
    check.eq(barrier['source_ids'], sorted(units), 'Growth barrier must include all accepted games')
    for name in FINGERPRINTS:
        check.eq(barrier[name], sorted(unit[name] for unit in units.values()), 'Growth barrier fingerprints: ' + name)
    for name, field in (('source_summary', 'source_summary'), ('source_exposure', 'source_exposure'), ('source_accepted', 'source_accepted')):
        check.eq(barrier[name], refs[field], 'Growth barrier provenance')
    components = composite['components']
    require(len(components) == 2 and [component['kind'] for component in components] == ['original', 'supplemental'], 'Composite components are missing or reordered')
    old, extra = components
    check.eq(old['source_ids'], chosen['original_source_ids'], 'Original component selection')
    check.eq(old['memberships'], [allowed[source] for source in old['source_ids']], 'Original component membership evidence')
    check.eq(extra['source_ids'], chosen['supplemental_source_ids'], 'Supplemental component selection')
    check.eq(extra['source_summary'], refs['source_summary'], 'Supplemental component pin')
    check.eq(extra['games'], [units[source] for source in extra['source_ids']], 'Supplemental component membership evidence')
    old_arrays, old_rows = verify_original_component(old, check)
    extra_arrays, extra_rows = normalized_examples(units, records, extra['source_ids'], split['plan'], base_manifest['config'], normalizer)
    expected = {name: np.concatenate([old_arrays[name], extra_arrays[name]]) for name in ARRAY_SCHEMA}
    arrays = read_arrays(diagnostic['arrays'], check)
    equal_arrays(arrays, expected, 'Composite arrays differ from original plus independently replayed supplement')
    check.eq(diagnostic['rows'], old_rows + extra_rows, 'Composite metadata differs from legal replay')
    counts = Counter(row['source_url'] for row in diagnostic['rows'])
    check.eq(counts, Counter({source: (allowed[source] if source in allowed else units[source]['membership'])['policy_actions']
                             for source in chosen['source_ids']}), 'Composite complete-game counts')
    require(len({row['row_key'] for row in diagnostic['rows']}) == len(diagnostic['rows']) == chosen['expected_rows'], 'Composite duplicate or missing rows')
    check.eq(diagnostic['excluded_rank_rows'], sum(row['source_rank_mover'] not in check.RANKS for row in diagnostic['rows']), 'Composite excluded-rank rows')
    return arrays, {'selection': composite['selection'], 'growth_exclusions': composite['growth_exclusions'],
                    'selected_original_games': len(old['source_ids']), 'selected_supplemental_games': len(extra['source_ids']),
                    'all_accepted_supplemental_games': len(units), 'rows': len(diagnostic['rows']),
                    'value_valid': int(arrays['value_valid'].sum()), 'legal_replay': 'Every selected supplemental action and suffix',
                    'original_component': 'Exact pinned original arrays and metadata, original normalizer applied once',
                    'protected_targets': 'Not decoded'}
