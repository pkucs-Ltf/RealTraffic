#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point - task selection and parameter configuration.
Supports three task types:
1. Simulation task (simulation)
2. Traffic light control optimization (traffic_light_optimization)
3. Lane change strategy optimization (lane_change_optimization)
"""

import argparse
import sys
import os
import yaml
import logging
from typing import Dict, Any

from task.simu_task import SimuTask
from task.opti_task_TLC import OptiTaskTLC
from task.lane_change_task import LaneChangeTask  # noqa: F401 – used via available_tasks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MainRunner")


class TaskRunner:
    """Main task runner class."""

    def __init__(self):
        """Initialize the task runner."""
        self.available_tasks = {
            'simulation': {
                'class': SimuTask,
                'config': 'configs/simu_Manha.yml',
                'description': 'Simulation task - urban region traffic simulation workflow'
            },
            'traffic_light_optimization': {
                'class': OptiTaskTLC,
                'config': 'configs/opti_task_TLC_Manha_dqn.yml',
                'description': 'Traffic light control optimization task'
            },
            'lane_change_optimization': {
                'class': LaneChangeTask,
                'config': 'configs/lane_change_task_DC.yml',
                'description': 'Lane change strategy optimization task - road resource coordination'
            },
        }

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            config_file (str): Path to the config file.

        Returns:
            Dict[str, Any]: Parsed configuration dictionary.
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Config loaded successfully: {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config {config_file}: {e}")
            return {}

    def validate_task_type(self, task_type: str) -> bool:
        """
        Validate whether a task type is supported.

        Args:
            task_type (str): Task type string.

        Returns:
            bool: True if valid, False otherwise.
        """
        return task_type in self.available_tasks

    def list_available_tasks(self):
        """Print all available task types."""
        print("\nAvailable task types:")
        print("=" * 60)
        for task_type, info in self.available_tasks.items():
            print(f"Task type: {task_type}")
            print(f"Description: {info['description']}")
            print(f"Default config: {info['config']}")
            print("-" * 60)

    def run_task(self, task_type: str, config_file: str = None,
                 config_override: Dict = None):
        """
        Run the specified task.

        Args:
            task_type (str): Task type.
            config_file (str): Path to config file (optional).
            config_override (Dict): Config overrides (optional).

        Returns:
            Any: Task execution result.
        """
        if not self.validate_task_type(task_type):
            logger.error(f"Invalid task type: {task_type}")
            self.list_available_tasks()
            return None

        task_info = self.available_tasks[task_type]

        if config_file is None:
            config_file = task_info['config']

        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            return None

        logger.info(f"Starting task: {task_type}")
        logger.info(f"Using config: {config_file}")

        try:
            config = self.load_config(config_file)

            if config_override:
                config.update(config_override)
                logger.info("Config overrides applied.")

            task_class = task_info['class']
            task_instance = task_class(config_dict=config)

            logger.info(f"Running: {task_info['description']}...")
            result = task_instance.run()

            logger.info(f"Task '{task_type}' completed.")
            return result

        except Exception as e:
            logger.error(f"Error running task '{task_type}': {e}", exc_info=True)
            return None

    def interactive_mode(self):
        """Launch interactive task selection mode."""
        print("\n=== Interactive Task Runner ===")
        self.list_available_tasks()

        while True:
            try:
                task_type = input("\nSelect a task type (type 'quit' to exit): ").strip()

                if task_type.lower() == 'quit':
                    print("Exiting.")
                    break

                if not self.validate_task_type(task_type):
                    print(f"Invalid task type: {task_type}")
                    continue

                use_custom_config = input("Use a custom config file? (y/N): ").strip().lower()
                config_file = None

                if use_custom_config == 'y':
                    config_file = input("Enter config file path: ").strip()
                    if not os.path.exists(config_file):
                        print(f"Config file not found: {config_file}")
                        continue

                result = self.run_task(task_type, config_file)

                if result:
                    print(f"\nTask result:")
                    print("-" * 40)
                    for key, value in result.items():
                        if isinstance(value, (str, int, float)):
                            print(f"{key}: {value}")
                    print("-" * 40)

                continue_choice = input("\nRun another task? (Y/n): ").strip().lower()
                if continue_choice == 'n':
                    break

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting.")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")


def create_argument_parser():
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Urban traffic simulation and optimization task runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available task types:
  simulation                 - Regional simulation task
  traffic_light_optimization - Traffic light control optimization
  lane_change_optimization   - Lane change strategy optimization

Examples:
  python run.py -t simulation
  python run.py -t simulation -c configs/simu_Manha.yml
  python run.py -t traffic_light_optimization -c configs/opti_task_TLC_Manha_dqn.yml
  python run.py -t lane_change_optimization -c configs/lane_change_task_DC.yml
  python run.py --interactive
  python run.py --list-tasks
        """
    )

    parser.add_argument(
        '-t', '--task-type',
        type=str,
        choices=['simulation', 'traffic_light_optimization', 'lane_change_optimization'],
        default='simulation',
        help='Task type (default: simulation)'
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to config file (optional; uses task default if not specified)'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Launch interactive mode'
    )

    parser.add_argument(
        '--list-tasks',
        action='store_true',
        help='List all available task types'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    return parser


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    runner = TaskRunner()

    if args.list_tasks:
        runner.list_available_tasks()
        return

    if args.interactive:
        runner.interactive_mode()
        return

    if args.task_type:
        result = runner.run_task(args.task_type, args.config)
        if result is None:
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
