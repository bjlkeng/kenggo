"""Check resumed saved evidence using the previously tested verifier functions."""
import importlib.util
import json
import re
import sys
from pathlib import Path

from verify_resume import OUT, ROOT, RESUME, validate_pins, ref

OLD = ROOT / 'runs/e2-progressive-final-verification-2026-11-21'
sys.path.insert(0, str(OLD))
import common

common.OUT = OUT
spec = importlib.util.spec_from_file_location('saved_checks', OLD / 'verify.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def recovery_receipt():
    summary = checks.SUMMARY
    continuation = summary['continuation']
    checks.refs(continuation['manifest'])
    previous = checks.read(continuation['previous_failed_summary'])
    checks.eq(previous['status'], 'failed', 'Previous failure was rewritten')
    checks.eq(continuation['original_exit_code'], None, 'Original exit must remain unknown')
    receipt = common.load(RESUME / 'growth-recovered.json')
    tracking = common.load(RESUME / 'tracking-recovered.json')
    checks.refs(receipt)
    checks.refs(tracking)
    checks.eq(receipt['exit_code'], None, 'Recovered receipt must retain unknown exit')
    checks.eq(tracking['status'], 'partial', 'Original tracking outcome must remain partial')
    checks.eq(tracking['tracking_finished'], True, 'Recovery tracking did not finish')
    checks.eq(summary['stages']['tier1-growth']['job'], receipt, 'Recovered growth stage receipt differs')
    ledger = common.load(OUT / 'production-resources.json')
    growth = next(entry for entry in ledger['entries'] if entry['run_id'] == 'tier1-growth')
    checks.eq(growth['outcome'], 'partial', 'Original process attempt must remain partial')
    return {'original_exit_code': None, 'growth_receipt': ref(RESUME / 'growth-recovered.json'),
            'previous_failure': continuation['previous_failed_summary'], 'original_ledger_outcome': growth['outcome']}


def page():
    from kenggo.progressive_review import export
    target = OUT / 'review.html'
    export(ref(OUT / 'production-summary.json'), target)
    text = target.read_text()
    match = re.search(r'<script id="learning-data" type="application/json">(.*?)</script>', text, re.S)
    if not match:
        raise ValueError('Shared learning component absent')
    data = json.loads(match.group(1))
    if len(data['runs']) < 2:
        raise ValueError('Earlier run comparisons absent')
    for label in ('training-monitor', 'stopping-validation', 'selected checkpoint', 'masked', 'premature'):
        if label not in text.lower():
            raise ValueError('Page explanation missing: ' + label)
    return {'owner_page': ref(target), 'production_summary': ref(OUT / 'production-summary.json'),
            'current_training_models': list(data['runs'][-1]['training']),
            'history_runs': [{'id': r['id'], 'description': r.get('description'), 'notes': r.get('notes')}
                             for r in data['runs']],
            'verification_costs': 'Separate final-resources.json includes the later verification charge.'}


def main():
    validate_pins()
    checks.PLAN = common.load(ROOT / 'configs/e2_progressive_training_v1.json')
    checks.BASE = checks.PLAN['parameters']['base_inputs']
    checks.RANKS = [f'{n}k' for n in range(12, 0, -1)] + ['1d']
    checks.SUMMARY = common.load(OUT / 'production-summary.json')
    checks.persist()
    checks.check('Frozen source, recovery and verification files', lambda: {'source': common.source(), 'pins': validate_pins()})
    checks.check('Stages, failures and original unknown exit', checks.stage_checks)
    if 'tier1-growth' in checks.SUMMARY['stages']:
        checks.check('Saved growth recovery and retained failure', recovery_receipt)
    else:
        checks.unavailable('Saved growth recovery', 'Resume did not publish a recovered growth stage')
    checks.check('Fixed comparison coverage', checks.fixed_checks)
    datasets = {checks.PLAN['inputs']['parent_summary']['sha256']: checks.PLAN['inputs']['parent_summary']}
    for stage in checks.SUMMARY['stages'].values():
        saved = checks.read(stage['summary'])
        if 'dataset_summary' in saved:
            datasets[saved['dataset_summary']['sha256']] = saved['dataset_summary']
    for pin in datasets.values():
        checks.check('Dataset ' + pin['sha256'][:12], lambda pin=pin: checks.dataset(pin))
    if any('preparation' in tier for tier in checks.SUMMARY['tiers']):
        checks.check('Stopping-validation separation and fixed labels', checks.preparation_checks)
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
    checks.check('New review page with curves and previous runs', page)
    checks.check('Final unchanged source and pinned evidence', lambda: {'source': common.source(), 'pins': validate_pins()})
    common.write('verified-hashes.json', checks.HASHES)
    failed = any(c['status'] == 'failed' for c in checks.RESULT['checks'])
    checks.RESULT.update(status='failed_checks' if failed else 'saved_evidence_checks_finished',
                         completed_utc=common.now().isoformat(), production_status=checks.SUMMARY['status'],
                         production_reason=checks.SUMMARY.get('reason'), verified_file_count=len(checks.HASHES),
                         browser='Separate browser-evidence.json records actual page checks',
                         root_acceptance='Pending fresh independent review; E2/.16/.17 remain open')
    checks.persist()
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
