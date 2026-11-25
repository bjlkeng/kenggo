"""Independent verifier fixtures; never read production arrays or run a model."""
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import composite_checks as verifier
from kenggo.data_splits import read_membership
from kenggo.continuation_data import read, reference
from kenggo.encoding import INPUT_KEYS, apply_normalizer

spec = importlib.util.spec_from_file_location('builder_fixture', ROOT / 'runs/e2-progressive-validation-resume-2026-11-24/test_composite_validation.py')
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def pin(ref):
    filename = Path(ref['path'])
    if hashlib.sha256(filename.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('Fixture pin differs')
    return filename


def eq(actual, expected, label):
    if actual != expected:
        raise ValueError(label)


def refs(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get('path'), str) and isinstance(obj.get('sha256'), str):
            pin(obj)
        for value in obj.values():
            refs(value)
    elif isinstance(obj, list):
        for value in obj:
            refs(value)


class CompositeVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE, prefix='fixture-composite-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = fixture_module.Fixture(self.root)
        original = read(self.fixture.request['inputs']['original_config'])
        original['inputs']['parent_summary'] = self.fixture.parent
        self.fixture.request['inputs']['original_config'] = fixture_module.write(self.root / 'config.json', original)
        self.fixture.request_ref = fixture_module.write(self.root / 'request.json', self.fixture.request)
        request = self.fixture.request_ref
        built = fixture_module.composite.build(request['path'], request['sha256'], self.root / 'job', fixture=True)
        self.diagnostic = read(built['epoch_validation'])
        self.check = types.SimpleNamespace(
            BASE=self.fixture.base, PLAN=original, RANKS=['11k', '1d'],
            HEAD='59d29abf57d8c03eec783a30222fa288403c8e7b',
            DEADLINE=datetime.fromisoformat('2026-11-28T14:13:03.148730+00:00'),
            read=lambda ref: json.loads(pin(ref).read_text()), pin=pin, refs=refs, eq=eq,
            membership=lambda ref: read_membership(pin(ref).parent, ref['sha256'])[1])

    def verify(self, data=None, **kwargs):
        return verifier.verify(data or self.diagnostic, self.check, fixture_only=True, **kwargs)

    def test_complete_components_verify_independently_and_preserve_value_masks(self):
        arrays, evidence = self.verify()
        self.assertEqual(evidence['selected_original_games'], 2)
        self.assertEqual(evidence['selected_supplemental_games'], 2)
        self.assertEqual(len(self.diagnostic['rows']), 12)
        self.assertEqual(int(arrays['policy_valid'].sum()), 12)
        self.assertEqual(int(arrays['value_valid'].sum()), 9)
        self.assertEqual(arrays['value'][-6:-3].tolist(), [1, -1, 1])
        self.assertEqual(arrays['value'][-3:].tolist(), [0, 0, 0])
        self.assertFalse(arrays['duration_valid'].any())
        self.assertNotIn('split', self.diagnostic['data'])

    def test_deterministic_selection_reconstructs_all_cells_and_unique_sources(self):
        inventory = read(self.diagnostic['composite']['inventory'])
        saved = read(self.diagnostic['composite']['selection'])
        for order in (inventory, list(reversed(inventory))):
            selected = verifier.selection(order, self.check.RANKS, 1, set(saved['excluded_source_ids']),
                                          saved['original_source_ids'], saved['supplemental_source_ids'])
            self.assertEqual(selected, saved)
        self.assertEqual(saved['cell_selections'], 8)
        self.assertEqual(saved['unique_games'], 4)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            verifier.selection(inventory, self.check.RANKS, 1, set(saved['source_ids']), [], [])

    def test_missing_original_membership_and_wrong_component_source_are_rejected(self):
        changed = deepcopy(self.diagnostic)
        changed['composite']['components'][0]['memberships'][0]['role'] = 'train'
        with self.assertRaisesRegex(ValueError, 'membership evidence'):
            self.verify(changed)
        changed = deepcopy(self.diagnostic)
        changed['composite']['components'][1]['source_ids'] = changed['composite']['components'][0]['source_ids']
        with self.assertRaisesRegex(ValueError, 'Supplemental component selection'):
            self.verify(changed)

    def test_supplement_training_or_confirmation_membership_is_rejected(self):
        unit = deepcopy(self.fixture.accepted[0])
        record = read(unit['canonical'])
        for role in ('train', 'validation_confirmation', 'final'):
            unit['membership']['role'] = role
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'membership'):
                verifier.validate_unit(unit, record, self.fixture.split['plan'])

    def test_member_mask_and_canonical_pins_are_independent(self):
        unit = deepcopy(self.fixture.accepted[0])
        record = read(unit['canonical'])
        unit['membership']['value_actions'] = 0
        with self.assertRaisesRegex(ValueError, 'membership'):
            verifier.validate_unit(unit, record, self.fixture.split['plan'])
        unit = deepcopy(self.fixture.accepted[0])
        unit['canonical']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'pin'):
            self.check.read(unit['canonical'])
        unit = deepcopy(self.fixture.accepted[0])
        unit['trajectory_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            verifier.validate_unit(unit, record, self.fixture.split['plan'])

    def test_changed_selection_pin_is_rejected_before_rows(self):
        changed = deepcopy(self.diagnostic)
        changed['composite']['selection']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'pin'):
            self.verify(changed)

    def test_missing_previous_stopping_exclusion_and_false_normalizer_fit_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Previous stopping'):
            self.verify(prior_sources={'previous-stopping-game'})
        changed = deepcopy(self.diagnostic)
        changed['composite']['normalization']['fitting_examples'] = self.fixture.grown['examples']
        with self.assertRaisesRegex(ValueError, 'normalization fitting'):
            self.verify(changed)

    def test_changed_masks_signs_legality_or_double_normalization_are_rejected(self):
        arrays = verifier.read_arrays(self.diagnostic['arrays'], self.check)
        for field, index, value in (('policy_valid', -1, False), ('value_valid', -1, True),
                                    ('duration_valid', -1, True), ('value', -6, -1), ('action', -1, 361),
                                    ('legal', (-1, 0), False)):
            with self.subTest(field=field):
                changed = {key: data.copy() for key, data in arrays.items()}
                changed[field][index] = value
                path = self.root / f'changed-{field}.npz'
                np.savez_compressed(path, **changed)
                diagnostic = deepcopy(self.diagnostic)
                diagnostic['arrays'] = reference(path)
                with self.assertRaisesRegex(ValueError, 'Composite arrays'):
                    self.verify(diagnostic)
        doubled = {key: data.copy() for key, data in arrays.items()}
        doubled.update(apply_normalizer({key: arrays[key] for key in INPUT_KEYS}, read(self.fixture.base['normalizer'])))
        path = self.root / 'double-normalized.npz'
        np.savez_compressed(path, **doubled)
        diagnostic = deepcopy(self.diagnostic)
        diagnostic['arrays'] = reference(path)
        with self.assertRaisesRegex(ValueError, 'Composite arrays'):
            self.verify(diagnostic)

    def test_incorrect_rank_exclusion_or_duplicate_row_metadata_is_rejected(self):
        changed = deepcopy(self.diagnostic)
        changed['excluded_rank_rows'] += 1
        with self.assertRaisesRegex(ValueError, 'excluded-rank'):
            self.verify(changed)
        changed = deepcopy(self.diagnostic)
        changed['rows'][-1] = changed['rows'][-2]
        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.verify(changed)

    def test_full_selected_supplement_is_legally_replayed_without_model_loading(self):
        with patch.object(verifier, 'examples_for_game', wraps=verifier.examples_for_game) as replay, \
             patch('kenggo.learner.PopulationModel', side_effect=AssertionError('Model execution forbidden')):
            self.verify()
        selected = self.diagnostic['composite']['components'][1]['source_ids']
        self.assertEqual([call.args[0]['source']['source_url'] for call in replay.call_args_list], selected)
        self.assertTrue(all(call.kwargs['komi_study'] for call in replay.call_args_list))
        units = {unit['source_url']: deepcopy(unit) for unit in self.fixture.accepted}
        records = {source: read(unit['canonical']) for source, unit in units.items()}
        record = records[selected[0]]
        record['actions'][-1]['action'] = record['actions'][0]['action']
        from kenggo.canonical_data import digest
        units[selected[0]]['membership']['canonical_sha256'] = digest(record)
        with self.assertRaises(ValueError):
            verifier.normalized_examples(units, records, selected, self.fixture.split['plan'],
                read(self.fixture.base['examples'])['config'], read(self.fixture.base['normalizer']))

    def test_original_array_component_is_checked_against_original_shards(self):
        component = deepcopy(self.diagnostic['composite']['components'][0])
        original = read(component['diagnostic'])
        arrays = verifier.read_arrays(original['arrays'], self.check)
        arrays['value'][0] *= -1
        changed_arrays = self.root / 'original-changed.npz'
        np.savez_compressed(changed_arrays, **arrays)
        original['arrays'] = reference(changed_arrays)
        component['diagnostic'] = fixture_module.write(self.root / 'original-changed.json', original)
        with self.assertRaisesRegex(ValueError, 'Original component normalized'):
            verifier.verify_original_component(component, self.check)


if __name__ == '__main__':
    unittest.main(verbosity=2)
