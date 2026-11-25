"""Finish the skipped composite check without repeating training or the 4M audit."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/home/bojji/code/kenggo')
OUT = ROOT / 'runs/e2-progressive-completion-2026-11-25'
OLD = ROOT / 'runs/e2-progressive-validation-verification-2026-11-24'
sys.path.insert(0, str(OLD))
import checks as adapter


def main():
    target = OUT / 'verification.json'
    if target.exists():
        raise ValueError('This completion attempt already exists; preserve it')
    adapter.validate_pins('bef826e4c665e10554ea34b712769a18c73a65c02bc90d1ed97c44bd93f21df8')
    spec = importlib.util.spec_from_file_location('corrected_composite', OUT / 'composite_checks_v2.py')
    composite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(composite)
    adapter.composite = composite
    adapter.common.OUT = OUT
    c = adapter.checks
    c.OUT = OUT
    c.PLAN = adapter.common.load(ROOT / 'configs/e2_progressive_training_v1.json')
    c.BASE = c.PLAN['parameters']['base_inputs']
    c.RANKS = [f'{n}k' for n in range(12, 0, -1)] + ['1d']
    c.SUMMARY = adapter.common.load(OLD / 'production-summary.json')
    c.diagnostic = adapter.diagnostic
    prior = c.read({'path': str(OLD / 'verification.json'),
                    'sha256': 'fd62bfe7e46f5fc72d3274f4b989c27f0b38df162adf13e615f10c6a12a51429'})
    failed = [item['name'] for item in prior['checks'] if item['status'] != 'passed']
    c.eq(failed, ['Composite stopping sample, independent replay and frozen fixed comparison'], 'Unexpected previous verification gaps')
    c.RESULT.update(schema='e2_composite_verification_correction_v1',
        previous_verification=adapter.ref(OLD / 'verification.json'),
        correction='Compare parent paths after resolution and verify both file hashes. Original failed evidence stays unchanged.',
        limits=['No training, inference, acquisition, complete-game panel or protected-target decoding.',
                'No repeat of the saved 4M fitting audit; read membership metadata to check disjointness.',
                'Check original component against saved arrays; fully replay and encode selected supplemental games.',
                'Other ten checks retain their pinned original evidence; this result does not erase the failed attempt.'])
    for pin in [c.PLAN['inputs']['parent_summary'], *[tier['dataset'] for tier in c.SUMMARY['tiers']]]:
        data = c.read(pin)
        members = c.membership(data['artifacts']['split_manifest'])
        c.DATASETS[pin['sha256']] = {row['source_url'] for row in members['train']}
    c.check('Fixed comparison prerequisites for the skipped check', c.fixed_checks)
    c.check(failed[0], adapter.preparation_checks)
    c.check('All original verifier inputs and production files remain pinned',
            lambda: adapter.validate_pins('bef826e4c665e10554ea34b712769a18c73a65c02bc90d1ed97c44bd93f21df8'))
    good = all(item['status'] == 'passed' for item in c.RESULT['checks'])
    c.RESULT.update(status='passed' if good else 'failed_checks', completed_utc=adapter.common.now().isoformat(),
                    verified_file_count=len(c.HASHES), root_acceptance='Pending separate fresh completion review')
    c.persist()
    adapter.common.write('verified-hashes.json', c.HASHES)
    print(json.dumps({'status':c.RESULT['status'],'checks':[(i['name'],i['status'],i.get('error')) for i in c.RESULT['checks']]}))
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
