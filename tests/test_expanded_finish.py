import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kenggo.base_pilot import _config, _ref
from kenggo.checkpoints import file_hash, save_json
from kenggo.expanded_finish import JOBS, OLD_STAGES, finish_jobs, run, saved_work, validate_finish
from kenggo.expanded_pilot import ARMS, ROOT
from kenggo.resource_ledger import BudgetLedger


class ExpandedFinishTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / '.cache').mkdir()
        self.output = self.root / 'runs/fixture'
        self.output.mkdir(parents=True)
        self.review = self.root / 'runs/review'
        self.review.mkdir()
        self.plan = json.loads((ROOT / 'configs/e2_expanded_plan_v1.json').read_text())
        self.plan['parameters']['output'] = 'runs/fixture'
        save_json(self.root / 'plan.json', self.plan)
        self.plan_ref = _ref(self.root / 'plan.json')
        save_json(self.output / 'frozen-training.json', {'updates': 31250})
        decision = _ref(self.output / 'frozen-training.json')
        sets = []
        for role in ('population_development', 'validation_development'):
            for komi in (13, 1):
                name = f'{role}_komi{komi}'
                save_json(self.output / (name + '.json'), {})
                sets.append({'spec': {'id': name, 'role': role, 'source_ids': [name],
                    'aggregate_ranks': ['1d'], 'komi_halfpoints': komi},
                    'manifest': _ref(self.output / (name + '.json'))})
        stages = {}
        for name in OLD_STAGES:
            directory = self.output / name
            worker = directory / 'worker'
            worker.mkdir(parents=True)
            inputs = {'training_decision': decision}
            config = {'inputs': inputs}
            summary = {'status': 'completed'}
            if name == 'preparation':
                summary['common_sets'] = sets
            elif name in ARMS:
                save_json(worker / 'checkpoint.json', {})
                summary.update(completed_updates=31250, presented_examples=4_000_000,
                    initial_tensor_sha256='a' * 64, row_keys_sha256=name.split('-')[0],
                    update_metrics=[{'order_sha256': name.split('-')[0]}],
                    final_checkpoint={'path': 'checkpoint.json', 'sha256': file_hash(worker / 'checkpoint.json')})
            elif name.startswith('evaluation-'):
                reports = []
                for item in sets:
                    filename = item['spec']['id'] + '.npz'
                    (worker / filename).write_bytes(b'synthetic predictions')
                    reports.append({'set': item['spec']['id'], 'variant': 'ordinary',
                        'artifacts': {'predictions.npz': {'path': filename, 'sha256': file_hash(worker / filename)}}})
                summary['reports'] = reports
            save_json(directory / 'config.json', config)
            save_json(directory / 'provenance.json', {'inputs': inputs})
            save_json(worker / 'summary.json', summary)
            job = {'status': 'completed'}
            save_json(worker / 'job.json', job)
            stages[name] = {'directory': str(directory), 'config': _ref(directory / 'config.json'),
                'provenance': _ref(directory / 'provenance.json'), 'summary': _ref(worker / 'summary.json'), 'job': job}
        self.ledger_path = self.root / self.plan['parameters']['ledger']
        self.ledger_path.parent.mkdir(parents=True)
        self.ledger = BudgetLedger.create(self.ledger_path, budget_id=self.plan['budget_id'], gpu_seconds=86400)
        before = self.ledger.snapshot()
        self.ledger.reserve('old', estimate_gpu_seconds=0, shutdown_gpu_seconds=0, metadata={})
        self.ledger.record('old', allocations=[], outcome='failed', arm_shares={'shared': 1},
                           cpu_seconds=20, wall_seconds=20, reason='reference cap')
        now = datetime.now(timezone.utc) - timedelta(hours=12)
        self.origin = {'schema': 'expanded_e2_session_v1', 'plan': self.plan_ref,
            'launch_utc': now.isoformat(), 'deadline_utc': (now + timedelta(days=1)).isoformat(),
            'limits': self.plan['parameters']['limits'], 'ledger_before': before}
        save_json(self.output / 'session.json', self.origin)
        failed_dir = self.output / 'evaluation-human-reference/worker'
        failed_dir.mkdir(parents=True)
        save_json(failed_dir / 'job.json', {'status': 'failed', 'reason': 'cpu_budget'})
        save_json(failed_dir / 'progress.json', {'status': 'running'})
        self.failure = {'schema': 'expanded_e2_pilot_v1', 'status': 'incomplete', 'reason': 'reference CPU cap',
            'promotion': 'not_promoted', 'plan': self.plan_ref, 'stages': stages,
            'failed_stage': {'name': 'evaluation-human-reference', 'directory': str(failed_dir.parent),
                'artifacts': {'worker/job.json': _ref(failed_dir / 'job.json')}},
            'costs': {'launch_utc': self.origin['launch_utc'], 'deadline_utc': self.origin['deadline_utc'],
                      'cpu_seconds': 20, 'elapsed_seconds': 20, 'ledger': self.ledger.snapshot()}}
        save_json(self.output / 'failure.json', self.failure)
        handoff = {'verification_status': 'passed_available_evidence',
            'production_failure': _ref(self.output / 'failure.json'),
            'verification_costs': {'gpu_seconds': 0, 'charged_cpu_seconds_including_reserve': 6, 'wall_seconds': 10},
            'production_costs': {'wait_cpu_seconds': .5}}
        save_json(self.review / 'handoff.json', handoff)
        refs = {key: self.plan['inputs'][key] for key in ('data', 'split', 'rules')}
        refs.update(original_plan=self.plan_ref, original_session=_ref(self.output / 'session.json'),
            original_failure=_ref(self.output / 'failure.json'), verification_handoff=_ref(self.review / 'handoff.json'),
            partial_reference_progress=_ref(failed_dir / 'progress.json'), training_decision=decision)
        self.config = _config(self.plan, purpose='expanded-finish', arm='shared', inputs=refs,
            parameters={'jobs': list(JOBS), 'output': 'runs/fixture', 'ledger': self.plan['parameters']['ledger'],
                'ledger_sha256': file_hash(self.ledger_path), 'deadline_utc': self.origin['deadline_utc'],
                'verification_directory': 'runs/review', 'preparation_cpu_reserve_seconds': 120})
        save_json(self.root / 'finish.json', self.config)
        self.patcher = patch('kenggo.expanded_finish.ROOT', self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_only_saved_analysis_and_original_match_panels_are_wired(self):
        validate_finish(self.config, self.output, self.ledger.snapshot()['state'])
        summaries, _ = saved_work(self.plan, self.failure, self.output)
        calls = []
        session = SimpleNamespace(stages=self.failure['stages'], stage=lambda *args, **kwargs: calls.append((args, kwargs)))
        finish_jobs(self.plan, self.plan_ref, _ref(self.root / 'finish.json'), summaries, session)
        self.assertEqual([args[0] for args, _ in calls], list(JOBS))
        self.assertEqual(len(calls[0][0][1]['parameters']['common_sets']), 4)
        for args, options in calls[1:]:
            cohort = args[0].removeprefix('matches-')
            config = args[1]
            self.assertEqual(config['inputs']['normalizer'], self.plan['parameters']['datasets'][cohort]['inputs']['normalizer'])
            self.assertEqual(config['parameters']['seed'] if 'seed' in config['parameters'] else config['seed'], self.plan['seed'])
            self.assertNotIn('device', options)
            for treatment in ('conditioned', 'unconditioned'):
                bundle = config['parameters']['learned_artifacts'][treatment]
                self.assertIn(cohort + '-' + treatment, bundle['checkpoint_manifest'])

    def test_changed_clock_ledger_or_already_attempted_work_is_rejected(self):
        state = self.ledger.snapshot()['state']
        changed = copy.deepcopy(state)
        changed['entries'] = []
        with self.assertRaisesRegex(ValueError, 'cumulative ledger'):
            validate_finish(self.config, self.output, changed)
        changed = copy.deepcopy(self.config)
        changed['parameters']['deadline_utc'] = datetime.now(timezone.utc).isoformat()
        with self.assertRaisesRegex(ValueError, 'unchanged capped-reference'):
            validate_finish(changed, self.output, state)
        (self.output / 'analysis').mkdir()
        with self.assertRaisesRegex(ValueError, 'one use only'):
            validate_finish(self.config, self.output, state)

    def test_missing_or_changed_saved_work_cannot_start_analysis(self):
        path = self.output / ARMS[0] / 'worker/summary.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'expected hash'):
            saved_work(self.plan, self.failure, self.output)

    def test_later_failure_keeps_original_evidence_and_cumulative_costs(self):
        original = (self.output / 'failure.json').read_bytes()
        def fail(*args):
            session = args[-1]
            session.failed_stage = {'name': 'matches-baseline', 'directory': str(self.output / 'matches-baseline'), 'artifacts': {}}
            raise RuntimeError('game CPU cap')
        with patch('kenggo.expanded_finish.capture_provenance', return_value=({'source': {}}, b'')), \
                patch('kenggo.expanded_finish.finish_jobs', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'game CPU cap'):
                run(self.root / 'finish.json', file_hash(self.root / 'finish.json'), self.output)
        result = json.loads((self.output / 'completion.json').read_text())
        self.assertEqual((self.output / 'failure.json').read_bytes(), original)
        self.assertEqual(result['failed_stage'], self.failure['failed_stage'])
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['completion_status'], 'failed')
        self.assertEqual(result['completion_failed_stage']['name'], 'matches-baseline')
        self.assertEqual(result['costs']['deadline_utc'], self.origin['deadline_utc'])
        self.assertGreaterEqual(result['costs']['elapsed_seconds'], 12 * 3600)
        self.assertGreaterEqual(result['costs']['cpu_seconds'], 20 + 6 + .5 + 120)
        self.assertFalse(result['costs']['ledger']['state']['reservations'])
        self.assertEqual(result['costs']['ledger']['charged_gpu_seconds'], 0)
        with self.assertRaisesRegex(ValueError, 'ledger changed'):
            run(self.root / 'finish.json', file_hash(self.root / 'finish.json'), self.output)


if __name__ == '__main__':
    unittest.main()
