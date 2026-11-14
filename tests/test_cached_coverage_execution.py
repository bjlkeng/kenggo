import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('audit_execution', Path(__file__).resolve().parents[1] / 'tools/run_cached_coverage.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExecutionAccountingTests(unittest.TestCase):
    def test_full_wrapper_and_preparation_are_charged(self):
        result = module.total_costs({'cpu_seconds': 10, 'wall_seconds': 12},
            {'cpu_seconds': 11.5, 'wall_seconds': 15},
            [{'cpu_seconds': .5, 'wall_seconds': 2}])
        self.assertEqual(result['total']['cpu_seconds'], 12)
        self.assertEqual(result['total']['wall_seconds'], 17)
        self.assertEqual(result['outside_job']['cpu_seconds'], 2)
        self.assertEqual(result['outside_job']['wall_seconds'], 5)
        self.assertEqual(result['budget_status'], 'within_limits')

    def test_outside_job_cost_can_breach_the_total_limit(self):
        result = module.total_costs({'cpu_seconds': 294, 'wall_seconds': 585},
            {'cpu_seconds': 299, 'wall_seconds': 598},
            [{'cpu_seconds': 2, 'wall_seconds': 6}])
        self.assertEqual(result['budget_status'], 'exceeded')
        self.assertEqual(result['total']['cpu_seconds'], 301)
        self.assertEqual(result['total']['wall_seconds'], 604)

    def test_conservative_job_charge_is_preserved(self):
        result = module.total_costs({'cpu_seconds': 20, 'wall_seconds': 30},
            {'cpu_seconds': 19.9, 'wall_seconds': 30.1}, [])
        self.assertEqual(result['total']['cpu_seconds'], 20)
        self.assertEqual(result['total']['wall_seconds'], 30.1)

    def test_measured_launch_charges_wrapper_and_job_once(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "runs") as name:
            root = Path(name).resolve()
            fake = root / 'fake_worker.py'
            fake.write_text("from pathlib import Path\nimport json,sys,time\ntime.sleep(.2)\n(Path(sys.argv[3])/'summary.json').write_text(json.dumps({'status':'completed'}))\n")
            frozen = root / 'frozen.json'; frozen.write_text('{}')
            prep = root / 'preparation.json'; prep.write_text(json.dumps({'costs': {'cpu_seconds': .01, 'wall_seconds': .02}}))
            def ref(path): return {'path': str(path), 'sha256': module.pin(path)}
            config = {'schema': 'e2_cached_coverage_execution_v1', 'worker': ref(fake), 'frozen': ref(frozen),
                'supervisor': ref(Path(module.__file__).resolve()), 'time_binary': ref(Path('/usr/bin/time')),
                'preparation': [ref(prep)], 'candidate_ids_sha256': 'fixture', 'output': str(root / 'output'),
                'limits': {'cpu_seconds': 20, 'wall_seconds': 20, 'shutdown_seconds': 1, 'output_bytes': 1024**2}}
            path = root / 'execution.json'; path.write_text(json.dumps(config))
            result = subprocess.run([sys.executable, module.__file__, str(path), '--sha256', module.pin(path)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root / 'output/summary.json').read_text())
            self.assertEqual(summary['execution_status'], 'completed')
            self.assertEqual(summary['audit_status'], 'completed')
            accounting = summary['full_execution_accounting']
            self.assertGreater(accounting['outside_job']['cpu_seconds'], .01)
            self.assertGreater(accounting['outside_job']['wall_seconds'], .02)
            entries = summary['ledger']['state']['entries']
            self.assertEqual(len(entries), 2)
            for key in ('cpu_seconds', 'wall_seconds'):
                self.assertAlmostEqual(sum(entry[key] for entry in entries), accounting['total'][key])
            self.assertEqual(summary['ledger']['charged_gpu_seconds'], 0)



if __name__ == '__main__':
    unittest.main()
