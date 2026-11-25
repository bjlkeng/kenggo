"""Exercise adapter routing and fitting barriers with in-memory saved evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('verification_adapter', HERE / 'checks.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AdapterTests(unittest.TestCase):
    def evidence(self):
        source_ref = {'path': 'source.json', 'sha256': 'source'}
        dataset_ref = {'path': 'dataset.json', 'sha256': 'dataset'}
        index_ref = {'path': 'index.json', 'sha256': 'index'}
        split_ref = {'path': 'split.json', 'sha256': 'split'}
        unit = {'source_url': 'heldout', 'sgf_sha256': 'a' * 64,
                'audit_sequence_sha256': 'b' * 64, 'trajectory_sha256': 'c' * 64}
        fitted = {'source_url': 'fitting', 'sgf_sha256': 'd' * 64,
                  'audit_sequence_sha256': 'e' * 64, 'trajectory_sha256': 'f' * 64}
        values = {'source': {'accepted': [unit]}, 'dataset': {'artifacts': {'split_manifest': split_ref, 'base_index': index_ref}},
                  'index': {'records': [fitted]}}
        members = {'train': [{'source_url': 'fitting'}]}
        return dataset_ref, values, members, {'continuation': {'source_summary': source_ref}}

    def test_all_accepted_supplement_sources_and_fingerprints_stay_outside_fitting(self):
        dataset, values, members, summary = self.evidence()
        with patch.object(module.checks, 'SUMMARY', summary, create=True), \
             patch.object(module.checks, 'read', side_effect=lambda ref: values[ref['sha256']]), \
             patch.object(module.checks, 'membership', return_value=members):
            result = module.growth_barrier(dataset)
            self.assertEqual(result['accepted_validation_sources_excluded'], 1)
            for field in ('source_url', *module.composite.FINGERPRINTS):
                old = values['index']['records'][0][field]
                values['index']['records'][0][field] = values['source']['accepted'][0][field]
                if field == 'source_url':
                    members['train'][0]['source_url'] = 'heldout'
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'entered fitting'):
                    module.growth_barrier(dataset)
                values['index']['records'][0][field] = old
                members['train'][0]['source_url'] = 'fitting'

    def test_missing_fitting_member_or_fingerprint_fails_closed(self):
        dataset, values, members, summary = self.evidence()
        with patch.object(module.checks, 'SUMMARY', summary, create=True), \
             patch.object(module.checks, 'read', side_effect=lambda ref: values[ref['sha256']]), \
             patch.object(module.checks, 'membership', return_value=members):
            values['index']['records'][0].pop('trajectory_sha256')
            with self.assertRaisesRegex(ValueError, 'lacks a duplicate fingerprint'):
                module.growth_barrier(dataset)
            values['index']['records'] = []
            with self.assertRaisesRegex(ValueError, 'missing member sources'):
                module.growth_barrier(dataset)

    def test_original_single_split_checker_reproduces_composite_failure(self):
        diagnostic = {'purpose': 'epoch_validation', 'aggregate_ranks': ['11k'], 'normalizer_sha256': 'normalizer',
                      'role': 'validation_development', 'data': {'original': {}, 'supplemental': {}}}
        with patch.object(module.checks, 'read', return_value=diagnostic), \
             patch.object(module.checks, 'BASE', {'normalizer': {'sha256': 'normalizer'}}, create=True), \
             patch.object(module.checks, 'RANKS', ['11k'], create=True), \
             patch.object(module.checks, 'DIAGNOSTICS', {}):
            with self.assertRaisesRegex(KeyError, 'split'):
                module.ORIGINAL_DIAGNOSTIC({'sha256': 'composite'}, 'epoch_validation')

    def test_composite_adapter_routes_to_independent_checker(self):
        diagnostic = {'purpose': 'epoch_validation', 'composite': {'tier': 1}, 'rows': []}
        with patch.object(module.checks, 'read', return_value=diagnostic), \
             patch.object(module.checks, 'SUMMARY', {'tiers': []}, create=True), \
             patch.object(module.checks, 'DIAGNOSTICS', {}), \
             patch.object(module.composite, 'verify', return_value=({}, {'verified': True})) as verify:
            self.assertEqual(module.diagnostic({'sha256': 'composite-route'}, 'epoch_validation'), (diagnostic, {}))
            verify.assert_called_once_with(diagnostic, module.checks, prior_sources=set())


if __name__ == '__main__':
    unittest.main(verbosity=2)
