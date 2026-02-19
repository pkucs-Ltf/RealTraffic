"""
Lane change strategy optimization task class.
Implements the road resource coordination optimization workflow.
"""

import logging
import os
import json
import re
from datetime import datetime
import shutil
import pickle
import uuid
import xml.etree.ElementTree as ET
import yaml
from typing import Dict, Any, Optional

from simulation.runner_opti import OptimizationRunner, get_all_tls_ids_from_netxml
from utils.lane_change_utils import GreedyLaneChanger, get_average_reward

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class LaneChangeTask:
    """Lane change strategy optimization task class - road resource coordination workflow."""

    def __init__(self, config_file: str = None, config_dict: Dict = None):
        """
        Initialize the lane change optimization task.

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

        if self.config.get('rl_tls_ids_all'):
            self.rl_tls_ids = get_all_tls_ids_from_netxml(self.config.get('data_paths', {}).get('original_net_file'))
        else:
            self.rl_tls_ids = self.config.get('rl_tls_ids', [])

        if not self.rl_tls_ids:
            self.rl_tls_ids = []

        self.city = self.config.get('city', 'Manha')
        self.data_paths = self.config.get('data_paths', {})
        self.simulation_config = self.config.get('simulation', {})
        self.optimization_config = self.config.get('optimization', {})
        self.lane_change_config = self.config.get('lane_change', {})
        self.output_config = self.config.get('output', {})

        self.optimization_runner = OptimizationRunner(self.config)

        run_prefix = self.output_config.get('results_dir_prefix', 'optimization_run')
        self.run_id = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = os.path.join(os.getcwd(), self.run_id)
        self.results_file = os.path.join(self.results_dir, "optimization_summary.json")

        self.max_simulations = self.simulation_config.get('max_simulations', 100)
        self.end_time = self.simulation_config.get('end_time', 600)
        self.top_n = self.optimization_config.get('top_n', 12)

        self.logger = logging.getLogger("LaneChangeTask")

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            return {}

    def _find_latest_model_by_name(self, model_dir: str, model_name: str) -> Optional[str]:
        """Find the latest episode checkpoint file for a given model name in a directory."""
        if not os.path.isdir(model_dir):
            return None

        checkpoint_files = [f for f in os.listdir(model_dir)
                            if f.startswith(f'{model_name}_episode_') and f.endswith('.pth')]

        if checkpoint_files:
            latest_file = sorted(checkpoint_files,
                                 key=lambda x: int(x.split('_episode_')[1].split('.')[0]))[-1]
            return os.path.join(model_dir, latest_file)

        return None

    def generate_final_report(self, all_results):
        """Generate a final report from all successful optimization results."""
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
        """Load existing initial decision data, or generate it via a baseline simulation."""
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

            initial_tl_phase, initial_lane_avg = self.optimization_runner.run_optimization_simulation(
                net_file_path=self.data_paths.get('original_net_file'),
                route_file=self.data_paths.get('route_file'),
                reward_static_pkl=self.data_paths.get('reward_static_pkl'),
                rl_tls_ids=self.rl_tls_ids,
                tls_to_train=[]
            )
            self.logger.info("Initial data generation complete.")

        return initial_tl_phase, initial_lane_avg

    def run_optimization_cycle(self, initial_state, max_simulations):
        """Run the main BFS-style optimization loop."""
        processing_queue = [initial_state]
        all_successful_results = {"original": initial_state}
        simulation_count = 1

        while processing_queue and simulation_count < max_simulations:
            next_processing_queue = []

            for current_state in processing_queue:
                self.logger.info(f"\n{'='*20} Processing network (Mod Cnt: {current_state['mod_count']}, Reward: {current_state['reward']:.2f}) {'='*20}")

                decisions_data = {
                    tl_id: {"preferred_phase": res[0], "connection_pressures": res[1]}
                    for tl_id, res in current_state['tl_phase_result'].items()
                }
                processor = GreedyLaneChanger(
                    lane_obeservation=current_state['lane_averages'],
                    top_n=self.top_n
                )

                temp_parse_dir = None
                try:
                    temp_parse_dir = processor.prepare_for_analysis(current_state['net_file_path'])
                    tl_ids = processor.get_traffic_light_ids()

                    all_proposals = []
                    for intersection_id in tl_ids:
                        if intersection_id not in decisions_data:
                            continue

                        proposals = processor.process_intersection(
                            intersection_id,
                            decisions_data[intersection_id]['preferred_phase'],
                            decisions_data[intersection_id]['connection_pressures']
                        )
                        if proposals:
                            all_proposals.extend(proposals)

                    if not all_proposals:
                        self.logger.info("No feasible proposals found for the current state.")
                        continue

                    all_proposals.sort(key=lambda p: p.get('probability', 0), reverse=True)
                    max_proposals = self.optimization_config.get('max_proposals_per_round', 10)
                    top_proposals = all_proposals[:max_proposals]
                    self.logger.info(f"Selected top {len(top_proposals)} proposals from {len(all_proposals)} candidates.")

                    for best_proposal in top_proposals:
                        if simulation_count >= max_simulations:
                            self.logger.warning("Maximum simulation count reached; terminating all loops.")
                            break

                        intersection_id_for_log = best_proposal['intersection_id']
                        self.logger.info(f"  -> Testing intersection [{intersection_id_for_log}] (total simulations: {simulation_count + 1}/{max_simulations})")

                        temp_net_file, temp_dir = processor.process_single_proposal(
                            current_state['net_file_path'], best_proposal
                        )

                        if not temp_net_file:
                            if temp_dir and os.path.exists(temp_dir):
                                shutil.rmtree(temp_dir)
                            continue

                        try:
                            base_checkpoint_dir = self.optimization_runner.rl_config.get('checkpoint_dir')
                            temp_checkpoint_dir = os.path.join(temp_dir, 'models')
                            os.makedirs(temp_checkpoint_dir, exist_ok=True)

                            tld_dict_path = os.path.join(base_checkpoint_dir, 'tls_dict.pkl')

                            if os.path.exists(tld_dict_path):
                                self.logger.info(f"    -> tld_dict.pkl detected; entering multi-model loading mode")
                                shutil.copy(tld_dict_path, temp_checkpoint_dir)
                                self.logger.info(f"    -> Copied tld_dict.pkl to temp directory")

                                with open(tld_dict_path, 'rb') as f:
                                    tld_dict = pickle.load(f)

                                models_found_count = 0
                                for tld_id, model_name_in_dict in tld_dict.items():
                                    latest_model_for_tld = self._find_latest_model_by_name(base_checkpoint_dir, model_name_in_dict)

                                    if latest_model_for_tld:
                                        shutil.copy(latest_model_for_tld, temp_checkpoint_dir)
                                        self.logger.info(f"    -> Copied latest model for {tld_id}: {os.path.basename(latest_model_for_tld)}")
                                        models_found_count += 1
                                    else:
                                        self.logger.warning(f"    -> No model file found for {tld_id} in {base_checkpoint_dir}")

                                if models_found_count == 0:
                                    self.logger.info(f"    -> No models found in any directory; will train from scratch")

                            simulation_count += 1
                            new_tl_phase, new_lane_avg = self.optimization_runner.run_optimization_simulation(
                                net_file_path=temp_net_file,
                                route_file=self.data_paths.get('route_file'),
                                reward_static_pkl=self.data_paths.get('reward_static_pkl'),
                                rl_tls_ids=self.rl_tls_ids,
                                tls_to_train=[intersection_id_for_log],
                                checkpoint_dir=temp_checkpoint_dir
                            )
                            new_reward = get_average_reward(self.data_paths.get('reward_static_pkl'))
                            print('Simulation plan', new_reward)
                            new_reward = new_reward[0]
                            print(new_reward, simulation_count)
                            if new_reward > current_state['reward'] + 30:
                                self.logger.info(f"    -> !!! Performance improved !!!  (parent reward: {current_state['reward']:.4f}, new reward: {new_reward:.4f})")

                                mod_count = current_state['mod_count'] + 1
                                state_id = str(uuid.uuid4())

                                round_dir = os.path.join(self.results_dir, f"{mod_count}_modifications")
                                os.makedirs(round_dir, exist_ok=True)
                                final_net_dir = os.path.join(round_dir, f"reward_{new_reward:.2f}_{state_id[:8]}".replace('.', '_'))
                                shutil.move(temp_dir, final_net_dir)
                                final_net_path = os.path.join(final_net_dir, os.path.basename(temp_net_file))

                                new_state = {
                                    "id": state_id,
                                    "net_file_path": os.path.abspath(final_net_path),
                                    "reward": new_reward,
                                    "mod_count": mod_count,
                                    "tl_phase_result": new_tl_phase,
                                    "lane_averages": new_lane_avg,
                                    "proposal_history": current_state['proposal_history'] + [best_proposal]
                                }

                                next_processing_queue.append(new_state)
                                all_successful_results[state_id] = new_state
                            else:
                                self.logger.info(f"    -> No improvement (parent reward: {current_state['reward']:.4f}, new reward: {new_reward:.4f}); discarding branch.")
                                shutil.rmtree(temp_dir)
                        except Exception as e:
                            self.logger.error(f"Error during simulation or proposal evaluation: {e}", exc_info=True)
                            if temp_dir and os.path.exists(temp_dir):
                                shutil.rmtree(temp_dir)

                finally:
                    if temp_parse_dir and os.path.exists(temp_parse_dir):
                        shutil.rmtree(temp_parse_dir)

                if simulation_count >= max_simulations:
                    break

            processing_queue = next_processing_queue

            if self.output_config.get('save_intermediate_results', True):
                final_report = self.generate_final_report(all_successful_results)
                with open(self.results_file, 'w', encoding='utf-8') as f:
                    json.dump(final_report, f, indent=4, ensure_ascii=False)

        return all_successful_results, simulation_count

    def run_optimization_cycle_dfs(self, initial_state, max_simulations):
        """Run the DFS-based optimization loop - depth-first exploration of improved networks."""
        all_successful_results = {"original": initial_state}
        simulation_count = 1

        def dfs_explore(current_state, depth=0):
            """Recursive DFS exploration function."""
            nonlocal simulation_count, all_successful_results

            if simulation_count >= max_simulations:
                return

            indent = "  " * depth
            self.logger.info(f"\n{indent}{'='*15} DFS explore (depth: {depth}, Mod Cnt: {current_state['mod_count']}, Reward: {current_state['reward']:.2f}) {'='*15}")

            decisions_data = {
                tl_id: {"preferred_phase": res[0], "connection_pressures": res[1]}
                for tl_id, res in current_state['tl_phase_result'].items()
            }
            processor = GreedyLaneChanger(
                lane_obeservation=current_state['lane_averages'],
                top_n=self.top_n
            )

            temp_parse_dir = None
            try:
                temp_parse_dir = processor.prepare_for_analysis(current_state['net_file_path'])
                tl_ids = processor.get_traffic_light_ids()

                all_proposals = []
                for intersection_id in tl_ids:
                    if intersection_id not in decisions_data:
                        continue

                    proposals = processor.process_intersection(
                        intersection_id,
                        decisions_data[intersection_id]['preferred_phase'],
                        decisions_data[intersection_id]['connection_pressures']
                    )
                    if proposals:
                        all_proposals.extend(proposals)

                if not all_proposals:
                    self.logger.info(f"{indent}No feasible proposals in current state; backtracking.")
                    return

                all_proposals.sort(key=lambda p: p.get('probability', 0), reverse=True)
                max_proposals = self.optimization_config.get('max_proposals_per_round', 10)
                top_proposals = all_proposals[:max_proposals]
                self.logger.info(f"{indent}Selected top {len(top_proposals)} proposals from {len(all_proposals)} for DFS testing.")

                for i, best_proposal in enumerate(top_proposals):
                    if simulation_count >= max_simulations:
                        self.logger.warning(f"{indent}Maximum simulation count reached; terminating DFS.")
                        break

                    intersection_id_for_log = best_proposal['intersection_id']
                    self.logger.info(f"{indent}-> Testing intersection [{intersection_id_for_log}] proposal {i+1}/{len(top_proposals)} (total simulations: {simulation_count + 1}/{max_simulations})")

                    temp_net_file, temp_dir = processor.process_single_proposal(
                        current_state['net_file_path'], best_proposal
                    )

                    if not temp_net_file:
                        if temp_dir and os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                        continue

                    try:
                        base_checkpoint_dir = self.optimization_runner.rl_config.get('checkpoint_dir')
                        temp_checkpoint_dir = os.path.join(temp_dir, 'models')
                        os.makedirs(temp_checkpoint_dir, exist_ok=True)

                        tld_dict_path = os.path.join(base_checkpoint_dir, 'tls_dict.pkl')

                        if os.path.exists(tld_dict_path):
                            self.logger.info(f"{indent}  -> tld_dict.pkl detected; entering multi-model loading mode")
                            shutil.copy(tld_dict_path, temp_checkpoint_dir)
                            self.logger.info(f"{indent}  -> Copied tld_dict.pkl to temp directory")

                            with open(tld_dict_path, 'rb') as f:
                                tld_dict = pickle.load(f)

                            models_found_count = 0
                            for tld_id, model_name_in_dict in tld_dict.items():
                                latest_model_for_tld = self._find_latest_model_by_name(base_checkpoint_dir, model_name_in_dict)

                                if latest_model_for_tld:
                                    shutil.copy(latest_model_for_tld, temp_checkpoint_dir)
                                    self.logger.info(f"{indent}  -> Copied latest model for {tld_id}: {os.path.basename(latest_model_for_tld)}")
                                    models_found_count += 1
                                else:
                                    self.logger.warning(f"{indent}  -> No model found for {tld_id} in {base_checkpoint_dir}")

                            if models_found_count == 0:
                                self.logger.info(f"{indent}  -> No models found; will train from scratch")

                        simulation_count += 1
                        new_tl_phase, new_lane_avg = self.optimization_runner.run_optimization_simulation(
                            net_file_path=temp_net_file,
                            route_file=self.data_paths.get('route_file'),
                            reward_static_pkl=self.data_paths.get('reward_static_pkl'),
                            rl_tls_ids=self.rl_tls_ids,
                            tls_to_train=[intersection_id_for_log],
                            checkpoint_dir=temp_checkpoint_dir
                        )
                        new_reward = get_average_reward(self.data_paths.get('reward_static_pkl'))
                        print('Simulation plan', new_reward)
                        new_reward = new_reward[0]
                        print(new_reward, simulation_count)

                        if new_reward > current_state['reward'] + 30:
                            self.logger.info(f"{indent}  -> !!! Performance improved !!!  (parent reward: {current_state['reward']:.4f}, new reward: {new_reward:.4f})")
                            print('Performance improved!!!', new_reward, current_state['reward'])
                            mod_count = current_state['mod_count'] + 1
                            state_id = str(uuid.uuid4())

                            round_dir = os.path.join(self.results_dir, f"{mod_count}_modifications")
                            os.makedirs(round_dir, exist_ok=True)
                            final_net_dir = os.path.join(round_dir, f"reward_{new_reward:.2f}_{state_id[:8]}".replace('.', '_'))
                            shutil.move(temp_dir, final_net_dir)
                            final_net_path = os.path.join(final_net_dir, os.path.basename(temp_net_file))

                            new_state = {
                                "id": state_id,
                                "net_file_path": os.path.abspath(final_net_path),
                                "reward": new_reward,
                                "mod_count": mod_count,
                                "tl_phase_result": new_tl_phase,
                                "lane_averages": new_lane_avg,
                                "proposal_history": current_state['proposal_history'] + [best_proposal]
                            }

                            all_successful_results[state_id] = new_state

                            if self.output_config.get('save_intermediate_results', True):
                                final_report = self.generate_final_report(all_successful_results)
                                with open(self.results_file, 'w', encoding='utf-8') as f:
                                    json.dump(final_report, f, indent=4, ensure_ascii=False)

                            self.logger.info(f"{indent}  -> Immediately diving deeper into this improved state...")
                            dfs_explore(new_state, depth + 1)

                            if simulation_count >= max_simulations:
                                break

                        else:
                            self.logger.info(f"{indent}  -> No improvement (parent reward: {current_state['reward']:.4f}, new reward: {new_reward:.4f}); trying next proposal.")
                            shutil.rmtree(temp_dir)

                    except Exception as e:
                        self.logger.error(f"{indent}Error during simulation or proposal evaluation: {e}", exc_info=True)
                        if temp_dir and os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)

            finally:
                if temp_parse_dir and os.path.exists(temp_parse_dir):
                    shutil.rmtree(temp_parse_dir)

        self.logger.info("=== Starting DFS-based depth-first optimization ===")
        dfs_explore(initial_state)

        return all_successful_results, simulation_count

    def run(self):
        if self.config.get('use_dfs', False):
            return self.run_with_dfs()
        """Main lane change optimization task execution flow."""
        os.makedirs(self.results_dir, exist_ok=True)
        self.logger.info(f"Optimization run ID: {self.run_id}")

        self.logger.info("--- Checking initial decision data ---")
        initial_tl_phase, initial_lane_avg = self.load_or_generate_initial_data()

        reward_static_pkl = self.data_paths.get('reward_static_pkl')
        initial_reward = get_average_reward(reward_static_pkl)
        initial_reward = initial_reward
        self.logger.info(f"Baseline reward on original network: {initial_reward:.4f}")

        initial_state = {
            "id": "original",
            "net_file_path": os.path.abspath(self.data_paths.get('original_net_file')),
            "reward": initial_reward,
            "mod_count": 0,
            "tl_phase_result": initial_tl_phase,
            "lane_averages": initial_lane_avg,
            "proposal_history": []
        }

        all_successful_results, simulation_count = self.run_optimization_cycle(
            initial_state, self.max_simulations
        )

        self.logger.info(f"\n{'='*20} All optimization rounds complete {'='*20}")
        self.logger.info(f"Total simulations run: {simulation_count}")
        self.logger.info(f"Effective optimized versions found: {len(all_successful_results) - 1}")

        if self.output_config.get('generate_summary_report', True):
            final_report = self.generate_final_report(all_successful_results)
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Final report saved to: {self.results_file}")

        return {
            'run_id': self.run_id,
            'initial_reward': initial_reward,
            'simulation_count': simulation_count,
            'successful_results_count': len(all_successful_results) - 1,
            'results_file': self.results_file,
            'results_dir': self.results_dir,
            'config': self.config
        }

    def run_with_dfs(self):
        """Main lane change optimization flow using DFS depth-first exploration."""
        os.makedirs(self.results_dir, exist_ok=True)
        self.logger.info(f"Optimization run ID (DFS mode): {self.run_id}")

        self.logger.info("--- Checking initial decision data ---")
        initial_tl_phase, initial_lane_avg = self.load_or_generate_initial_data()

        reward_static_pkl = self.data_paths.get('reward_static_pkl')
        initial_reward = get_average_reward(reward_static_pkl)
        initial_reward = initial_reward
        self.logger.info(f"Baseline reward on original network: {initial_reward:.4f}")
        print('Baseline reward on original network:', initial_reward)

        initial_state = {
            "id": "original",
            "net_file_path": os.path.abspath(self.data_paths.get('original_net_file')),
            "reward": initial_reward,
            "mod_count": 0,
            "tl_phase_result": initial_tl_phase,
            "lane_averages": initial_lane_avg,
            "proposal_history": []
        }

        all_successful_results, simulation_count = self.run_optimization_cycle_dfs(
            initial_state, self.max_simulations
        )

        self.logger.info(f"\n{'='*20} DFS optimization complete {'='*20}")
        self.logger.info(f"Total simulations run: {simulation_count}")
        self.logger.info(f"Effective optimized versions found: {len(all_successful_results) - 1}")

        if self.output_config.get('generate_summary_report', True):
            final_report = self.generate_final_report(all_successful_results)
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Final report saved to: {self.results_file}")

        return {
            'run_id': self.run_id,
            'initial_reward': initial_reward,
            'simulation_count': simulation_count,
            'successful_results_count': len(all_successful_results) - 1,
            'results_file': self.results_file,
            'results_dir': self.results_dir,
            'config': self.config,
            'search_method': 'DFS'
        }


def main():
    """Standalone entry point for testing."""
    task = LaneChangeTask("configs/lane_change_task.yml")
    result = task.run()
    print(f"Lane change optimization task completed. Result: {result}")


if __name__ == "__main__":
    main()
