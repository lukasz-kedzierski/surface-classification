#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import yaml
from tqdm import tqdm


class ProgressTracker:
    """Track progress of multiple experiments using file-based communication."""

    def __init__(self, progress_dir, experiment_names):
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_names = experiment_names
        self.progress_bars = {}
        self.stop_monitoring = False

        # Create progress files for each experiment
        for exp_name in experiment_names:
            progress_file = self.progress_dir / f"{exp_name}_progress.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_steps': 0,
                    'current_step': 0,
                    'status': 'initializing',
                    'current_fold': 0,
                    'total_folds': 0,
                    'current_epoch': 0,
                    'total_epochs': 0,
                    'train_loss': 0.0,
                    'val_loss': 0.0
                }, f)

    def start_monitoring(self):
        """Start monitoring progress in a separate thread."""
        self.monitor_thread = threading.Thread(target=self._monitor_progress, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring_func(self):
        """Stop monitoring progress."""
        self.stop_monitoring = True
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)

        # Close all progress bars
        for pbar in self.progress_bars.values():
            pbar.close()

    def _monitor_progress(self):
        """Monitor progress files and update progress bars."""
        while not self.stop_monitoring:
            for exp_name in self.experiment_names:
                progress_file = self.progress_dir / f"{exp_name}_progress.json"

                try:
                    if progress_file.exists():
                        with open(progress_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Create progress bar if it doesn't exist
                        if exp_name not in self.progress_bars:
                            if data['total_steps'] > 0:
                                self.progress_bars[exp_name] = tqdm(
                                    total=data['total_steps'],
                                    desc=f"{exp_name}",
                                    position=len(self.progress_bars),
                                    leave=True
                                )

                        # Update existing progress bar
                        elif exp_name in self.progress_bars:
                            pbar = self.progress_bars[exp_name]

                            # Update total if it changed
                            if pbar.total != data['total_steps'] and data['total_steps'] > 0:
                                pbar.total = data['total_steps']
                                pbar.refresh()

                            # Update progress
                            if data['current_step'] > pbar.n:
                                pbar.update(data['current_step'] - pbar.n)

                            # Update description with current status
                            status_info = []
                            if data['status'] == 'training':
                                status_info.append(f"Fold {data['current_fold']}/{data['total_folds']}")
                                status_info.append(f"Epoch {data['current_epoch']}/{data['total_epochs']}")
                                if data['train_loss'] > 0:
                                    status_info.append(f"Loss: {data['train_loss']:.2E}")
                            elif data['status'] == 'completed':
                                status_info.append("✅ COMPLETED")
                            elif data['status'] == 'failed':
                                status_info.append("❌ FAILED")

                            desc = f"{exp_name}"
                            if status_info:
                                desc += f" - {' | '.join(status_info)}"

                            pbar.set_description(desc)

                except (json.JSONDecodeError, FileNotFoundError, KeyError):
                    # Skip if file is being written or doesn't exist yet
                    continue

            time.sleep(0.5)  # Update every 500ms


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
        '--progress-dir',
        progress_dir
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

    # Get experiment names
    experiment_names = [exp['experiment_name'] for exp in experiments_configs]

    # Initialize progress tracker
    progress_tracker = ProgressTracker(progress_dir, experiment_names)
    progress_tracker.start_monitoring()

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

    # Stop progress monitoring
    progress_tracker.stop_monitoring_func()

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
