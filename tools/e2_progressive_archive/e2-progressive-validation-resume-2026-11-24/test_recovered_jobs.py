"""Behavior checks for preserving the stopping games across later data growth."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_data_growth import parent_fixture, sources_for, request_for
from test_population_audit import write_json
from kenggo import data_growth, progressive_jobs
from kenggo.data_splits import read_membership

spec = importlib.util.spec_from_file_location('recovered_jobs_tested', HERE / 'recovered_jobs.py')
j = importlib.util.module_from_spec(spec)
spec.loader.exec_module(j)


def source_fixture(root, record):
    games = [{**record, 'membership': {'role': 'validation_development', 'exclusions': []}},
             {'source_url': 'unrelated-validation', **{k: 'f' * 64 for k in j.FIELDS},
              'membership': {'role': 'validation_development', 'exclusions': []}}]
    return write_json(root / 'source.json', {'status': 'completed', 'yield_status': 'sufficient_yield',
                                            'usable_games': 2, 'accepted': games})


class RecoveryJobsTests(unittest.TestCase):
    def test_reviewed_helper_change_is_rejected_before_job_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'runs/recovery'
            folder.mkdir(parents=True)
            files = {}
            for name in ('recovered_jobs.py', 'composite-validation.py', 'resume.py'):
                p = folder / name
                p.write_text('original')
                files[str(p.relative_to(root))] = j.file_hash(p)
            with patch.object(j, 'HERE', folder), patch.object(j, 'read', return_value={'files': files}):
                j.verify_runtime({'inputs': {'resume_manifest': {}}})
                (folder / 'composite-validation.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'helper changed'):
                    j.verify_runtime({'inputs': {'resume_manifest': {}}})

    def test_reserved_cached_promotion_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = parent_fixture(root)
            parent_index = j.read(j.read(parent)['artifacts']['base_index'])
            record = next(r for r in parent_index['records'] if r['source_url'].endswith('/capacity.sgf'))
            source = source_fixture(root, {**record, 'source_url': 'different-validation-source'})
            _, sources = sources_for(root, parent, [('allowed', 'oo', 'pp', 6.5)])
            request = request_for(parent, sources, 6)
            request_ref = write_json(root / 'request.json', request)
            config = write_json(root / 'config.json', {'purpose': 'progressive-growth',
                'inputs': {'request': request_ref, 'validation_supplement': source}})
            output = root / 'worker'
            output.mkdir()
            def original_run(*args):
                data_growth.build_growth_round(request, output / 'dataset')
                return {'status': 'completed', 'dataset_summary': j.reference(output / 'dataset/summary.json')}
            with patch.object(j, 'verify_runtime'), patch.object(progressive_jobs, 'run', side_effect=original_run):
                result = j.run(config['path'], config['sha256'], 'unused', 'unused', output)
            self.assertEqual(j.read(result['dataset_summary'])['promoted_source_ids'], [])
            self.assertEqual(j.read(result['stopping_exclusion_check'])['fitting_overlap'], 0)

    def test_actual_growth_rejects_supplement_copy_without_changing_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = parent_fixture(root)
            _, sources = sources_for(root, parent, [('reserved-copy', 'oo', 'pp', 6.5), ('allowed', 'qq', 'rr', 6.5)])
            request = request_for(parent, sources, 8)
            reference_run = data_growth.build_growth_round(request, root / 'reference')
            index = j.read(reference_run['artifacts']['base_index'])
            reserved = next(r for r in index['records'] if r['source_url'].endswith('/reserved-copy.sgf'))
            source = source_fixture(root, {**reserved, 'source_url': 'different-validation-source'})
            request_ref = write_json(root / 'request.json', request)
            config = write_json(root / 'config.json', {'purpose': 'progressive-growth',
                'inputs': {'request': request_ref, 'validation_supplement': source}})
            output = root / 'worker'
            output.mkdir()
            def original_run(*args):
                data_growth.build_growth_round(request, output / 'dataset')
                return {'status': 'completed', 'dataset_summary': j.reference(output / 'dataset/summary.json')}
            with patch.object(j, 'verify_runtime'), patch.object(progressive_jobs, 'run', side_effect=original_run):
                result = j.run(config['path'], config['sha256'], 'unused', 'unused', output)
            summary = j.read(result['dataset_summary'])
            _, rows = read_membership(j.path(summary['artifacts']['split_manifest']).parent,
                                      summary['artifacts']['split_manifest']['sha256'])
            self.assertFalse(any(r['source_url'].endswith('/reserved-copy.sgf') for r in rows['train']))
            self.assertTrue(any(r['source_url'].endswith('/allowed.sgf') for r in rows['train']))
            rejected = next(r for r in rows['excluded'] if r['source_url'].endswith('/reserved-copy.sgf'))
            self.assertIn('duplicate_of_frozen_corpus', rejected['exclusions'])
            self.assertEqual(j.file_hash(parent['path']), parent['sha256'])
            self.assertEqual(j.read(result['stopping_exclusion_check'])['fitting_overlap'], 0)

    def test_source_id_and_each_fingerprint_block_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = parent_fixture(root)
            refs = j.read(parent)['artifacts']
            _, rows = read_membership(j.path(refs['split_manifest']).parent, refs['split_manifest']['sha256'])
            fitting = {r['source_url'] for r in rows['train']}
            record = next(r for r in j.read(refs['base_index'])['records'] if r['source_url'] in fitting)
            for field in ('source_url', *j.FIELDS):
                with self.subTest(field=field):
                    changed = {name: 'x' * 64 for name in ('source_url', *j.FIELDS)}
                    changed[field] = record[field]
                    ref = source_fixture(root, changed)
                    with self.assertRaisesRegex(ValueError, 'entered fitting'):
                        j.assert_fitting_disjoint(parent, ref)

    def test_source_failure_cannot_authorize_growth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = source_fixture(root, {'source_url': 'one', **{k: 'a' * 64 for k in j.FIELDS}})
            value = j.read(ref)
            value['status'] = 'partial'
            ref = write_json(root / 'partial.json', value)
            with self.assertRaisesRegex(ValueError, 'distinct admitted'):
                j.supplement(ref)

    def test_job_exception_restores_all_original_functions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = write_json(root / 'config.json', {'purpose': 'progressive-preparation'})
            original = (progressive_jobs.prepare, j.data_expansion._new_memberships, data_growth._promotion_ids)
            with patch.object(j, 'verify_runtime'), patch.object(progressive_jobs, 'run', side_effect=RuntimeError('fixture')):
                with self.assertRaisesRegex(RuntimeError, 'fixture'):
                    j.run(ref['path'], ref['sha256'], 'unused', 'unused', root)
            self.assertEqual(original, (progressive_jobs.prepare, j.data_expansion._new_memberships, data_growth._promotion_ids))


if __name__ == '__main__':
    unittest.main(verbosity=2)
