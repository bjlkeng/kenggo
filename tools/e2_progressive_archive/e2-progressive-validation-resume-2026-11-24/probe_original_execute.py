"""Read-only integration probe: reach the first training call without running it."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('resume_probe', HERE / 'resume.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ReachedTraining(BaseException):
    pass


def probe():
    study = object.__new__(r.ResumedStudy)
    study.plan = r.load(r.ROOT / 'configs/e2_progressive_training_v1.json')
    study.output = r.PROD
    study.state = deepcopy(r.load(r.PREVIOUS / 'summary.json'))
    study.state.update(tiers=[], status='running')
    study.manifest_ref = {'path': 'not-needed-in-read-only-probe', 'sha256': 'not-needed'}
    study.manifest = {'composite_request': r.reference(HERE / 'composite-request.json')}
    study.require = lambda *args, **kwargs: None
    study.check = lambda: None
    study.persist = lambda: None
    report = {'reused_stages': [], 'production_jobs_started': 0, 'model_queries': 0}
    before = (r.PROD / 'tier1-growth-request.json').read_bytes()
    reused = r.ResumedStudy.stage.__get__(study)
    def stage(name, config, **options):
        if name in study.state['stages']:
            value = reused(name, config, **options)
            report['reused_stages'].append(name)
            return value
        if config['purpose'] == 'progressive-preparation':
            selected = study.coverage(1)
            request = r.read(study.manifest['composite_request'])
            report['selection'] = {'games': len(selected['selection']['source_ids']),
                                   'rows': selected['selection']['expected_rows']}
            return ({'status': 'completed', 'summary': {'path': 'synthetic-unopened-preparation', 'sha256': 'synthetic'}},
                    {'train_monitor': request['inputs']['train_monitor'],
                     'epoch_validation': {'path': 'synthetic-unopened-validation', 'sha256': 'synthetic'}})
        if config['purpose'] == 'continuation-training':
            report.update(first_training_stage=name, parameters=config['parameters'],
                          normalizer=config['inputs']['normalizer'],
                          fitting_examples=config['inputs']['examples'])
            raise ReachedTraining()
        raise AssertionError('Unexpected production step: ' + name)
    study.stage = stage
    with patch.object(r, 'verify_manifest'), patch.object(r.original, 'save_json', side_effect=r.immutable_request):
        try:
            r.original.execute(study)
        except ReachedTraining:
            pass
    if report.get('first_training_stage') != 'tier1-control-epoch00':
        raise AssertionError('Original loop did not reach its unchanged first training stage')
    assert report['parameters']['round_updates'] == 31250
    assert report['parameters']['evaluate_only'] is True
    assert before == (r.PROD / 'tier1-growth-request.json').read_bytes()
    assert report['normalizer'] == study.plan['parameters']['base_inputs']['normalizer']
    report['original_growth_request_unchanged'] = True
    return report


if __name__ == '__main__':
    print(json.dumps(probe(), indent=2))
