"""Execute the one frozen cached-coverage audit inside its finite CPU boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time



def pin(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def total_costs(job, wrapper, preparation):
    total = {key: max(job[key], wrapper[key]) + sum(item[key] for item in preparation)
             for key in ('cpu_seconds', 'wall_seconds')}
    return {'total': total, 'outside_job': {key: total[key] - job[key] for key in total},
            'budget_status': 'within_limits' if total['cpu_seconds'] <= 300 and total['wall_seconds'] <= 600 else 'exceeded'}


def process_wall():
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    ticks = os.sysconf('SC_CLK_TCK')
    return time.clock_gettime(time.CLOCK_BOOTTIME) - int(fields[19]) / ticks + 1 / ticks


def execute(config, args):
    from kenggo.checkpoints import save_json
    from kenggo.pilot_jobs import JobLimits, run_job
    from kenggo.resource_ledger import BudgetLedger
    limits = JobLimits(**config['limits'])
    if (limits.cpu_seconds > 295 or limits.wall_seconds > 580 or limits.shutdown_seconds > 10
            or limits.output_bytes > 126 * 1024**2):
        raise ValueError('Limits exceed the declared execution reserve')
    output = Path(config['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'config.json', config)
    ledger = BudgetLedger.create(output / 'resources.json', budget_id='e2-cached-coverage-2026-11-14', gpu_seconds=0)
    job = run_job([sys.executable, config['worker']['path'], 'audit', config['frozen']['path'],
                   str(output / 'job'), '--inputs-sha256', config['frozen']['sha256']],
        output / 'job', cwd=Path.cwd(), limits=limits, ledger=ledger,
        run_id='e2-cached-coverage-2026-11-14', arm_shares={'coverage_audit': 1}, device=None,
        environment={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
        metadata={'frozen_sha256': config['frozen']['sha256'], 'candidate_ids_sha256': config['candidate_ids_sha256']})
    summary_path = output / 'job' / 'summary.json'
    worker_summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    save_json(output / 'worker_execution.json', {'schema': config['schema'], 'job_status': job['status'],
        'audit_status': None if worker_summary is None else worker_summary['status'],
        'audit_summary': None if worker_summary is None else {'path': str(summary_path), 'sha256': pin(summary_path)},
        'job_costs': job['costs'], 'ledger': ledger.snapshot(), 'config_sha256': args.sha256,
        'gpu_seconds': 0, 'network_requests': 0})
    print(json.dumps({'job_status': job['status'], 'audit_status': None if worker_summary is None else worker_summary['status'], 'costs': job['costs']}))
    if job['status'] != 'completed':
        raise SystemExit(1)


def measured_execution(config, args):
    from kenggo.checkpoints import save_json
    from kenggo.resource_ledger import BudgetLedger
    output = Path(config['output']).resolve()
    if output.exists():
        raise ValueError('Execution output already exists')
    measured = args.config.with_suffix('.execution-time.json')
    measured.touch(exist_ok=False)
    time_format = '{"wall_seconds":%e,"user_cpu_seconds":%U,"system_cpu_seconds":%S,"exit_code":%x}'
    result = subprocess.run([config['time_binary']['path'], '--quiet', '--format', time_format,
        '--output', str(measured), sys.executable, str(Path(__file__)), str(args.config),
        '--sha256', args.sha256, '--worker'], check=False)
    timing = json.loads(measured.read_text())
    worker = json.loads((output / 'worker_execution.json').read_text())
    preparation = [json.loads(Path(ref['path']).read_text())['costs'] for ref in config['preparation']]
    # GNU time prints hundredths; round both CPU components and elapsed time upward.
    wrapper = {'cpu_seconds': timing['user_cpu_seconds'] + timing['system_cpu_seconds'] + .02 + time.process_time(),
               'wall_seconds': max(timing['wall_seconds'] + .01, process_wall())}
    accounting = total_costs(worker['job_costs'], wrapper, preparation)
    ledger = BudgetLedger.open(output / 'resources.json', budget_id='e2-cached-coverage-2026-11-14', gpu_seconds=0)
    accounting['output_bytes'] = sum(p.stat().st_size for folder in (output, args.config.parent)
                                      for p in folder.rglob('*') if p.is_file())
    if accounting['output_bytes'] > 128 * 1024**2:
        accounting['budget_status'] = 'exceeded'
    outcome = 'completed' if result.returncode == 0 and accounting['budget_status'] == 'within_limits' else 'failed'
    overhead_id = 'e2-cached-coverage-2026-11-14-control'
    ledger.reserve(overhead_id, estimate_gpu_seconds=0, shutdown_gpu_seconds=0,
                   metadata={'purpose': 'full_wrapper_and_metadata_preparation', 'config_sha256': args.sha256})
    ledger.record(overhead_id, allocations=[], outcome=outcome, arm_shares={'coverage_audit': 1},
                  reason=None if outcome == 'completed' else 'execution_or_total_budget_failure', **accounting['outside_job'])
    save_json(output / 'summary.json', {**worker, 'execution_status': outcome,
        'full_execution_accounting': accounting, 'wrapper_measurement': {'path': str(measured), 'sha256': pin(measured)},
        'wrapper_costs': wrapper, 'preparation': config['preparation'], 'ledger': ledger.snapshot(),
        'cost_scope': 'Full supervised wrapper process including imports and final writes, outer controller through the accounting boundary, and recorded metadata preparation. Measurement rounding is charged upward; final accounting-file writes are outside the measurement.'})
    print(json.dumps({'execution_status': outcome, 'audit_status': worker['audit_status'], **accounting}))
    if outcome != 'completed':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if pin(args.config) != args.sha256:
        raise ValueError('Execution configuration changed')
    config = json.loads(args.config.read_text())
    if config['schema'] != 'e2_cached_coverage_execution_v1':
        raise ValueError('Unsupported execution configuration')
    for ref in [config[key] for key in ('worker', 'frozen', 'supervisor', 'time_binary')] + config['preparation']:
        if pin(ref['path']) != ref['sha256']:
            raise ValueError('Pinned execution input changed: ' + ref['path'])
    if Path(config['supervisor']['path']).resolve() != Path(__file__).resolve():
        raise ValueError('Wrong execution supervisor')
    (execute if args.worker else measured_execution)(config, args)


if __name__ == '__main__':
    main()
