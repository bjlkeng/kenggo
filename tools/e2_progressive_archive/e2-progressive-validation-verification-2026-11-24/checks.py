"""Inspect final saved evidence and publish a new composite-validation owner page."""
import html
import importlib.util
import json
import re
import sys
from pathlib import Path

from verify_resume import OUT, ROOT, RESUME, DEADLINE, validate_pins, ref
import composite_checks as composite

OLD = ROOT / 'runs/e2-progressive-final-verification-2026-11-21'
PREVIOUS = ROOT / 'runs/e2-progressive-resume-2026-11-24'
sys.path.insert(0, str(OLD))
import common

common.OUT = OUT
spec = importlib.util.spec_from_file_location('saved_checks', OLD / 'verify.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
ORIGINAL_DIAGNOSTIC = checks.diagnostic
COMPOSITE_DETAILS = {}
AUDIT = {'path': str(PREVIOUS / 'audit-saved-growth.json'),
         'sha256': 'b7848438f90edb0e63e7fd560cdd81a7dabf9746a7acd1439188cdbfa44a2b62'}


def recovery_receipt():
    summary = checks.SUMMARY
    continuation = summary['continuation']
    checks.refs(continuation)
    previous = checks.read(continuation['previous_failed_summary'])
    checks.eq(previous['status'], 'failed', 'Previous failed preparation was rewritten')
    original = checks.read(previous['continuation']['previous_failed_summary'])
    checks.eq(original['status'], 'failed', 'Original failure was rewritten')
    checks.eq(continuation['growth_audit'], AUDIT, 'Saved growth audit changed')
    audit = checks.read(AUDIT)
    receipt = checks.read(continuation['growth_receipt'])
    checks.eq(audit['status'], 'passed', 'Saved growth audit did not pass')
    checks.refs(receipt)
    checks.eq(receipt['exit_code'], None, 'Recovered growth exit must remain unknown')
    checks.eq(summary['stages']['tier1-growth']['job'], receipt, 'Recovered growth receipt differs')
    phase = checks.read(continuation['source_phase'])
    source = checks.read(continuation['source_summary'])
    checks.eq(continuation['source_summary']['sha256'], composite.SOURCE_SHA, 'Reviewed supplemental source summary changed')
    checks.eq(phase['status'], 'completed', 'Supplemental source phase incomplete')
    checks.eq(source['usable_games'], 7, 'Supplemental usable yield changed')
    checks.eq(source['training_performed'], False, 'Source check was relabeled as training')
    ledger = common.load(OUT / 'production-resources.json')
    growth = next(entry for entry in ledger['entries'] if entry['run_id'] == 'tier1-growth')
    checks.eq(growth['outcome'], 'partial', 'Original process attempt must remain partial')
    return {'original_exit_code': None, 'growth_receipt': continuation['growth_receipt'],
            'growth_audit': AUDIT, 'previous_failure': continuation['previous_failed_summary'],
            'original_failure': previous['continuation']['previous_failed_summary'],
            'original_ledger_outcome': growth['outcome'], 'source_phase': continuation['source_phase'],
            'source_summary': continuation['source_summary'], 'usable_supplemental_games': 7}


def saved_dataset(pin):
    from kenggo.continuation_data import validate_lineage
    from kenggo.progressive_training import dataset_inputs
    audit = checks.read(AUDIT)
    original_growth = audit['dataset']['dataset']
    if pin['sha256'] not in (checks.PLAN['inputs']['parent_summary']['sha256'], original_growth['sha256']):
        result = checks.dataset(pin)
        growth_barrier(pin)
        return result
    checks.eq(audit['status'], 'passed', 'Saved growth audit status')
    inputs = dataset_inputs(pin, checks.BASE)
    checks.refs(inputs)
    manifest, _ = validate_lineage(inputs)
    root = checks.pin(inputs['examples']).parent
    for shards in manifest['artifacts'].values():
        for shard in shards:
            for kind in ('arrays', 'metadata'):
                checks.pin({'path': str(root / shard[kind]), 'sha256': shard[kind + '_sha256']})
    members = checks.membership(inputs['split'])
    checks.DATASETS[pin['sha256']] = {row['source_url'] for row in members['train']}
    checks.eq(inputs['normalization_examples'], audit['dataset']['normalizer_fit_examples'], 'Saved audit fitting lineage')
    if pin['sha256'] == original_growth['sha256']:
        checks.eq(manifest['counts']['train']['examples'], audit['dataset']['rows'], 'Saved audit fitting rows')
        checks.eq(len(members['train']), audit['dataset']['games'], 'Saved audit fitting games')
    growth_barrier(pin)
    return {'dataset': pin, 'saved_complete_audit': AUDIT, 'rows': manifest['counts']['train']['examples'],
            'games': len(members['train']), 'normalizer_fit_examples': checks.BASE['examples'],
            'method': 'Recheck every saved shard hash and original fitting lineage; reuse exact completed 4M audit for unchanged bytes.',
            'protected_targets': 'Hashes only; no decoding', 'full_old_fitting_audit_repeated': False}


def growth_barrier(pin):
    source = checks.read(checks.SUMMARY['continuation']['source_summary'])
    units = source['accepted']
    summary = checks.read(pin)
    members = checks.membership(summary['artifacts']['split_manifest'])
    training = {row['source_url'] for row in members['train']}
    index = checks.read(summary['artifacts']['base_index'])
    records = composite.unique(index['records'], 'fitting index')
    composite.require(training <= set(records), 'Fitting index is missing member sources')
    protected_ids = {unit['source_url'] for unit in units}
    composite.require(not training & protected_ids, 'Supplemental validation source entered fitting')
    for name in composite.FINGERPRINTS:
        composite.require(all(isinstance(records[source].get(name), str) and len(records[source][name]) == 64
                              for source in training), 'Fitting index lacks a duplicate fingerprint: ' + name)
        protected = {unit[name] for unit in units}
        composite.require(not {records[source].get(name) for source in training} & protected,
                          'Supplemental validation fingerprint entered fitting: ' + name)
    return {'dataset': pin, 'accepted_validation_sources_excluded': len(units), 'fingerprints': list(composite.FINGERPRINTS)}


def diagnostic(pin, purpose):
    data = checks.read(pin)
    if 'composite' not in data:
        return ORIGINAL_DIAGNOSTIC(pin, purpose)
    checks.eq(purpose, 'epoch_validation', 'Composite sample used outside stopping validation')
    if pin['sha256'] in checks.DIAGNOSTICS:
        return checks.DIAGNOSTICS[pin['sha256']]
    number = data['composite']['tier']
    previous = set()
    for tier in checks.SUMMARY['tiers']:
        if tier['tier'] >= number:
            continue
        composite.require('preparation' in tier, 'A preceding tier has no frozen stopping sample')
        preparation = checks.read(tier['preparation'])
        previous.update(checks.read(preparation['epoch_validation'])['source_ids'])
    arrays, evidence = composite.verify(data, checks, prior_sources=previous)
    from kenggo.recorded_data import _inputs_hash
    from kenggo.encoding import INPUT_KEYS
    checks.DIAGNOSTIC_INPUT_HASHES[pin['sha256']] = {
        row['row_key']: _inputs_hash({name: arrays[name][i] for name in INPUT_KEYS}) for i, row in enumerate(data['rows'])}
    checks.DIAGNOSTICS[pin['sha256']] = (data, arrays)
    COMPOSITE_DETAILS[pin['sha256']] = evidence
    return data, arrays


def preparation_checks():
    details = checks.preparation_checks()
    for tier in checks.SUMMARY['tiers']:
        if 'preparation' not in tier:
            continue
        prepared = checks.read(tier['preparation'])
        data = checks.read(prepared['epoch_validation'])
        checks.eq(prepared['composite'], data['composite'], 'Preparation and diagnostic component provenance')
        checks.eq(prepared['growth_exclusions'], data['composite']['growth_exclusions'], 'Preparation growth barrier')
        checks.eq(prepared['selection_uses_targets_or_predictions'], False, 'Stopping selection used targets')
    return {'samples': details, 'composite_checks': COMPOSITE_DETAILS,
            'fixed_comparison': {'games': 539, 'rows': 120450, 'rank12k_to1d_rows': 111591}}


def recovery_page_section(summary):
    continuation = summary['continuation']
    source = checks.read(continuation['source_summary'])
    failed = checks.read(continuation['previous_failed_summary'])
    rows = []
    for tier in summary['tiers']:
        if 'preparation' not in tier:
            continue
        prep = checks.read(tier['preparation'])
        diagnostic = checks.read(prep['epoch_validation'])
        selected = checks.read(diagnostic['composite']['selection'])
        rows.append(f"<li>Round {tier['tier']}: {selected['unique_games']} whole stopping games, "
                    f"{selected['expected_rows']:,} positions, including {len(selected['supplemental_source_ids'])} supplemental games. "
                    f"{diagnostic['excluded_rank_rows']:,} positions are outside the displayed 12k–1d scope.</li>")
    esc = html.escape
    limits = ('The stopping sample joins separately pinned original and supplemental memberships. '
              'It is not a new unified train/validation split. Every accepted supplemental game stays outside fitting. '
              'The original 2,000,000-move training normalizer is applied once. '
              'The fixed comparison remains 539 games / 120,450 positions; 111,591 positions are in the 12k–1d scope. '
              'Losses average positions within game sides, then sides within each mover rank, then ranks equally. '
              'Policy and value masks and valid-label counts remain separate. '
              'Earlier model metrics use different populations unless a fixed paired comparison is explicitly shown. '
              'One seed and development selection support exploratory prediction conclusions only; premature passing, '
              'the earlier 0.5-komi value concern and complete-game qualification remain unresolved.')
    no_training = '' if any(tier['training'] for tier in summary['tiers']) else '<p>No new training occurred in this resumed attempt. New training and validation curves are not available.</p>'
    return (f'<section id="validation-recovery"><h2>Validation coverage repair · .17 and .18</h2>'
            f'<p>The previous continuation stopped before training because every existing 11k/6.5-komi validation game was already in the fixed comparison. '
            f'The bounded source check admitted {source["usable_games"]} additional whole games under the original account and date rules. '
            f'The source check did not train or evaluate a model. The original 28 November deadline and all prior charges remain in force.</p>'
            f'<p>Retained previous failure: {esc(str(failed.get("reason")))}</p><ul>{"".join(rows) or "<li>No composite stopping sample was published.</li>"}</ul>'
            f'{no_training}<p>{esc(limits)}</p><p>Final verification costs are recorded separately in final-resources.json; they are appended after this page and browser checks finish.</p>'
            f'<p><a href="{esc(str(Path(continuation["source_phase"]["path"]).resolve()), quote=True)}">Saved source phase</a> · '
            f'<a href="{esc(str(Path(continuation["previous_failed_summary"]["path"]).resolve()), quote=True)}">Preserved failed preparation</a></p></section>')


def page():
    from kenggo.progressive_review import export
    target = OUT / 'review.html'
    export(ref(OUT / 'production-summary.json'), target)
    text = target.read_text()
    text = text.replace('<main>', '<main>' + recovery_page_section(checks.SUMMARY), 1)
    target.write_text(text)
    metadata = common.load(target.with_suffix('.json'))
    metadata.update(page=ref(target), validation_recovery=checks.SUMMARY['continuation'])
    common.write('review.json', metadata)
    match = re.search(r'<script id="learning-data" type="application/json">(.*?)</script>', text, re.S)
    if not match:
        raise ValueError('Shared learning component absent')
    data = json.loads(match.group(1))
    if len(data['runs']) < 2:
        raise ValueError('Earlier run comparisons absent')
    for label in ('training-monitor', 'stopping-validation', 'selected checkpoint', 'masked', 'premature', 'validation-recovery'):
        if label not in text.lower():
            raise ValueError('Page explanation missing: ' + label)
    return {'owner_page': ref(target), 'production_summary': ref(OUT / 'production-summary.json'),
            'current_training_models': list(data['runs'][-1]['training']),
            'history_runs': [{'id': run['id'], 'description': run.get('description'), 'notes': run.get('notes')}
                             for run in data['runs']],
            'verification_costs': 'Separate final-resources.json includes the later verification charge.'}


def main():
    validate_pins()
    checks.PLAN = common.load(ROOT / 'configs/e2_progressive_training_v1.json')
    checks.BASE = checks.PLAN['parameters']['base_inputs']
    checks.RANKS = [f'{n}k' for n in range(12, 0, -1)] + ['1d']
    checks.SUMMARY = common.load(OUT / 'production-summary.json')
    checks.diagnostic = diagnostic
    checks.RESULT['limits'][0] = 'No training, inference, acquisition or complete-game panel. Every selected supplemental source is replayed and re-encoded.'
    checks.persist()
    checks.check('Frozen source, recovery and verification files', lambda: {'source': common.source(), 'pins': validate_pins()})
    checks.check('Stages, failures and original unknown exit', checks.stage_checks)
    checks.check('Saved growth, failed preparation and supplemental source chain', recovery_receipt)
    checks.check('Fixed comparison coverage and original target hashes', checks.fixed_checks)
    datasets = {checks.PLAN['inputs']['parent_summary']['sha256']: checks.PLAN['inputs']['parent_summary']}
    for stage in checks.SUMMARY['stages'].values():
        saved = checks.read(stage['summary'])
        if 'dataset_summary' in saved:
            datasets[saved['dataset_summary']['sha256']] = saved['dataset_summary']
    for pin in datasets.values():
        checks.check('Fitting lineage and supplemental exclusion ' + pin['sha256'][:12], lambda pin=pin: saved_dataset(pin))
    if any('preparation' in tier for tier in checks.SUMMARY['tiers']):
        checks.check('Composite stopping sample, independent replay and frozen fixed comparison', preparation_checks)
    else:
        checks.unavailable('Stopping validation', 'No stopping sample was published')
    for tier in checks.SUMMARY['tiers']:
        number = str(tier['tier'])
        if tier['training']:
            checks.check('Matched training tier ' + number, lambda tier=tier: checks.training_checks(tier))
        else:
            checks.unavailable('Training tier ' + number, 'No new training pair was published')
        if tier['assessments']:
            checks.check('Fixed comparisons tier ' + number, lambda tier=tier: checks.assessment_checks(tier))
        else:
            checks.unavailable('Fixed comparisons tier ' + number, 'No new fixed assessment was published')
    checks.check('Actual owner page with curves, previous runs and recovery limitations', page)
    checks.check('Final unchanged source and pinned evidence', lambda: {'source': common.source(), 'pins': validate_pins()})
    common.write('verified-hashes.json', checks.HASHES)
    failed = any(check['status'] == 'failed' for check in checks.RESULT['checks'])
    checks.RESULT.update(status='failed_checks' if failed else 'saved_evidence_checks_finished',
                         completed_utc=common.now().isoformat(), production_status=checks.SUMMARY['status'],
                         production_reason=checks.SUMMARY.get('reason'), verified_file_count=len(checks.HASHES),
                         browser='Separate browser-evidence.json records actual page checks',
                         root_acceptance='Pending fresh independent review; E2/.16/.17/.18 remain open')
    checks.persist()
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
