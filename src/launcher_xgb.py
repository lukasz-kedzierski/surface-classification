#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import yaml


def run_training_instance(script_path, config, experiment_id, output_dir, progress_dir):
    """Run a single training instance with the given configuration."""

    # Build the command
    cmd = [
        sys.executable,
        script_path,
        '--config',
        config,
        '--experiment-id',
        experiment_id,
        '--output-dir',
        output_dir,
        ]

    print(f"Starting training with config: {experiment_id}")
    process = subprocess.run(cmd, check=False, text=True)

    return {
        'config': config,
        'experiment_id': experiment_id,
        'returncode': process.returncode,
        'success': process.returncode == 0
    }


def load_configurations(config_file):
    """Load all configurations from YAML file."""
    with open(config_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return data


def main():
    parser = argparse.ArgumentParser(description='Run multiple training instances')
    parser.add_argument('--config-file', required=True,
                        help='YAML file containing all configurations')
    parser.add_argument('--script-path', required=True,
                        help='Path to your training script')
    parser.add_argument('--output-dir', default='./results_new/cv',
                        help='Base directory for experiment outputs')
    parser.add_argument('--max-parallel', type=int, default=4,
                        help='Maximum number of parallel processes')

    args = parser.parse_args()

    # Load configurations
    try:
        config_params = load_configurations(args.config_file)
        experiments_configs = config_params['experiments']
        print(f"Loaded {len(experiments_configs)} configurations")
    except Exception as e:
        print(f"Error loading config file: {e}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir).joinpath(time.strftime("%Y-%m-%d-%H-%M-%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create progress directory
    progress_dir = output_dir / "progress"

    # Parallel execution
    print(f"Running {len(experiments_configs)} experiments with max {args.max_parallel} parallel processes")

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        # Submit all jobs
        experiments, _, _ = config_params.values()
        ids = (params['experiment_name'] for params in experiments)
        future_to_config = {
            executor.submit(run_training_instance, args.script_path, args.config_file, experiment_id, output_dir, progress_dir): experiment_id
            for experiment_id in ids
        }

        results = []
        for future in as_completed(future_to_config):
            result = future.result()
            results.append(result)

            # Update progress file to mark as completed/failed
            exp_name = result['experiment_id']
            progress_file = progress_dir / f"{exp_name}_progress.json"

            if progress_file.exists():
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    data['status'] = 'completed' if result['success'] else 'failed'
                    data['current_step'] = data['total_steps']

                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f)
                except:
                    pass  # If we can't update, it's not critical

    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print("\n=== Summary ===")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed experiments:")
        for result in results:
            if not result['success']:
                exp_name = result['config'].get('experiment_name', 'Unknown')
                print(f"  - {exp_name}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
