"""
Traffic light control (TLC) optimization task class.
Implements the traffic signal optimization workflow.
"""

import logging
import os
import json
from datetime import datetime
import shutil
import pickle
import uuid
import xml.etree.ElementTree as ET
import yaml
from typing import Dict, Any

from simulation.runner_opti import OptimizationRunner, get_all_tls_ids_from_netxml
from utils.lane_change_utils import get_average_reward

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class OptiTaskTLC:
    """Traffic light control optimization task class."""

    def __init__(self, config_file: str = None, config_dict: Dict = None):
        """
        Initialize the TLC optimization task.

        Args:
            config_file (str): Path to the config file.
            config_dict (Dict): Config dictionary; takes priority if provided.
        """
        if config_dict:
            self.config = config_dict
        elif config_file:
            self.config = self._load_config(config_file)
        else:
            raise ValueError("Either config_file or config_dict must be provided.")

        self.city = self.config.get('city', 'Manha')
        self.data_paths = self.config.get('data_paths', {})
        self.rl_config = self.config.get('rl_config', {})

        self.optimization_runner = OptimizationRunner(self.config)
        run_prefix = self.config.get('results_dir_prefix', 'optimization_run')
        self.run_id = run_prefix + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = os.path.join(os.getcwd(), self.run_id)
        self.results_file = os.path.join(self.results_dir, "optimization_summary.json")

        self.logger = logging.getLogger("OptiTaskTLC")

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            return {}

    def generate_final_report(self, all_results):
        """Generate a final report from all successful results."""
        if not all_results:
            return {}

        report = {
            "best_by_modification_count": {},
            "overall_best": None
        }

        best_by_mods = {}
        for result in all_results.values():
            mod_count = result['mod_count']
            if mod_count not in best_by_mods or result['reward'] > best_by_mods[mod_count]['reward']:
                best_by_mods[mod_count] = result

        report['best_by_modification_count'] = best_by_mods

        overall_best = max(all_results.values(), key=lambda x: x['reward'])
        report['overall_best'] = overall_best

        return report

    def get_tl_ids_from_net(self, net_file_path: str) -> list:
        """Lightly parse a net.xml file to extract all traffic-light junction IDs."""
        ids = []
        try:
            tree = ET.parse(net_file_path)
            root = tree.getroot()
            for junction in root.findall('junction'):
                if 'traffic_light' in junction.get('type', ''):
                    ids.append(junction.get('id'))
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse XML file {net_file_path}: {e}")
        return ids

    def load_or_generate_initial_data(self):
        """Load existing initial decision data, or generate it by running a baseline simulation."""
        decisions_data_pkl = self.data_paths.get('decisions_data_pkl')
        lane_averages_pkl = self.data_paths.get('lane_averages_pkl')

        if os.path.exists(decisions_data_pkl) and os.path.exists(lane_averages_pkl):
            with open(decisions_data_pkl, 'rb') as f:
                initial_tl_phase = pickle.load(f)
            with open(lane_averages_pkl, 'rb') as f:
                initial_lane_avg = pickle.load(f)
            self.logger.info("Loaded existing initial decision data.")
        else:
            self.logger.info("No initial data files found; running a baseline simulation to generate data.")

            if self.config.get('rl_tls_ids_all'):
                rl_tls_ids = get_all_tls_ids_from_netxml(self.data_paths.get('original_net_file'))
            else:
                rl_tls_ids = self.config.get('rl_tls_ids', [])

            if not rl_tls_ids:
                rl_tls_ids = []

            initial_tl_phase, initial_lane_avg = self.optimization_runner.run_optimization_simulation(
                net_file_path=self.data_paths.get('original_net_file'),
                route_file=self.data_paths.get('route_file'),
                rl_tls_ids=rl_tls_ids
            )
            self.logger.info("Initial data generation complete.")

        return initial_tl_phase, initial_lane_avg

    def run(self):
        """Main TLC optimization task execution flow."""
        os.makedirs(self.results_dir, exist_ok=True)
        self.logger.info(f"Optimization run ID: {self.run_id}")

        self.logger.info("--- Checking initial decision data ---")
        initial_tl_phase, initial_lane_avg = self.load_or_generate_initial_data()

        simulation_count = 0

        reward_static_pkl = self.data_paths.get('reward_static_pkl')
        initial_reward = get_average_reward(reward_static_pkl)
        initial_reward = initial_reward[0]
        print(initial_reward)
        simulation_count += 1
        self.logger.info(f"Baseline reward on original network: {initial_reward:.4f}")

        result = {
            'run_id': self.run_id,
            'initial_reward': initial_reward,
            'simulation_count': simulation_count,
            'initial_tl_phase': initial_tl_phase,
            'initial_lane_avg': initial_lane_avg,
            'results_dir': self.results_dir,
            'config': self.config
        }

        self.logger.info("TLC optimization task initialization complete.")
        return result

    def run_extended_optimization(self):
        """Run an extended optimization pipeline (placeholder for additional logic)."""
        pass


def main():
    """Standalone entry point for testing."""
    task = OptiTaskTLC("configs/opti_task_TLC.yml")
    result = task.run()
    print(f"TLC optimization task completed. Result: {result}")


if __name__ == "__main__":
    main()
