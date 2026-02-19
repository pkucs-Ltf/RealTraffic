"""
Simulation task class.
Implements the urban region traffic simulation workflow.
"""

import yaml
import logging
import os
import shutil
from typing import Dict, Any, List

from simulation.runner_simu import SimulationRunner
from evaluation.comparator import Comparator
from adjustment.scale import ScaleAdjuster
from utils.car_simulate_simu import Generate_NewOD, dumpfile, loadfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimuTask:
    """Simulation task class - executes the urban region traffic simulation workflow."""

    def __init__(self, config_file: str = None, config_dict: Dict = None):
        """
        Initialize the simulation task.

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

        self.simulation_runner = SimulationRunner(self.config.get('simulation', {}))
        self.comparator = Comparator(self.config.get('optimizer', {}))
        self.scale_adjuster = ScaleAdjuster(self.config.get('optimizer', {}))

        self.state = self._init_state()

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _init_state(self) -> Dict[str, Any]:
        """Initialize the task state."""
        data_paths = self.config.get('data_paths', {})
        simulation_config = self.config.get('simulation', {})

        state = {
            "iteration": 0,
            "current_score": float('inf'),
            "best_score": float('inf'),
            "best_params": None,
            "history": [],
            "last_action": None,
            "convergence_count": 0,
            "exploration_counter": 0,
            "scale_test_results": [],
        }

        state.update(data_paths)

        state.update({
            'need_mergeRegion': simulation_config.get('need_mergeRegion', False),
            'max_od_iterations': simulation_config.get('max_od_iterations', 50),
        })

        return state

    def copy_file(self, source_path, destination_path):
        """
        Copy a file from source to destination.

        If the source file does not exist, an empty file is created first.
        If the destination directory does not exist, it is created automatically.

        Args:
            source_path (str): Path to the source file.
            destination_path (str): Path to the destination file.
        """
        try:
            if not os.path.exists(source_path):
                print(f"Source file {source_path} not found; creating an empty file.")
                with open(source_path, 'w') as f:
                    pass

            dest_dir = os.path.dirname(destination_path)
            if dest_dir and not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
                print(f"Created destination directory: {dest_dir}")

            shutil.copy(source_path, destination_path)
            print(f"File copied from {source_path} to {destination_path}")
            return True

        except Exception as e:
            print(f"Error during file copy: {e}")
            return False

    def iterative_optimization(self, max_od_iterations: int = None):
        """
        Iterative OD matrix optimization loop.

        Args:
            max_od_iterations: Maximum number of OD optimization iterations.
        """
        if max_od_iterations is None:
            max_od_iterations = self.state.get('max_od_iterations', 50)

        logger.info("Starting OD matrix optimization phase...")

        generator = Generate_NewOD(
            rou_file=self.state['rou_file'],
            grid_shp=self.state["grid_shp"],
            third_adjust_firstSTEP=False,
            od_pkl=self.state["od_file"]
        )

        new_od_dict, new_od_matrix, num_regions = generator.generate(
            needmerge=self.state["need_mergeRegion"]
        )

        if 'rou_file' in self.state and 'iop_file' in self.state:
            shutil.copy(self.state['rou_file'], self.state['iop_file'])

        self.simulation_runner.reset(
            rou_file=self.state['rou_file'],
            od_dict=new_od_dict,
            real_data_file=self.state['real_data_file'],
            edge_limit_file=self.state['edge_limit_file'],
            net_file=self.state['net_file'],
            iop_file=self.state.get('iop_file', None),
            taz_file=self.state.get('taz_file', None)
        )

        reward_max = -9999
        best_sumodata = None
        history_average_speeds_best = None

        for step in range(max_od_iterations):
            logger.info(f"OD fine-tuning step {step+1}/{max_od_iterations}")

            state_info, reward, done, info, av_speed, sumodata, history_average_speeds = \
                self.simulation_runner.step(self.comparator)
            logger.info(f'best: {history_average_speeds}')

            percentage = ((self.simulation_runner.numcars - 19794) / 19794) * 100 if hasattr(self.simulation_runner, 'numcars') else None
            logger.info(f"Current vehicle count: {self.simulation_runner.numcars}, percentage of 19794: {percentage:.2f}%")

            if reward > 0.6 and abs(percentage) < 40:
                experiment_dir = self.config.get('output', {}).get('experiment_dir', 'Experiment/output')
                self.copy_file(self.state['best_rou_file'], os.path.join(experiment_dir, "best_{}_{}.rou.xml".format(reward, percentage)))

            if reward > 0.80 and abs(percentage) < 10:
                break

            if reward > reward_max:
                reward_max = reward
                best_sumodata = sumodata
                history_average_speeds_best = history_average_speeds
                shutil.copy(self.simulation_runner.iop_file, self.state['best_rou_file'])
                logger.info(f"New best found - score: {info['score']:.4f}, reward: {reward:.4f}")

            if done:
                logger.info(f"Convergence reached at step {step+1}")
                break

            logger.info(f"Step {step+1} complete - score: {info['score']:.4f}, reward: {reward:.4f}")

        if history_average_speeds_best:
            logger.info(f'{reward_max} best score')
            logger.info(f'{history_average_speeds_best}')
        if best_sumodata:
            output_dir = self.config.get('output', {}).get('pkl_dir', 'pkl')
            os.makedirs(output_dir, exist_ok=True)
            dumpfile(history_average_speeds_best, os.path.join(output_dir, 'history_average_speeds_best.pkl'))
            dumpfile(best_sumodata, os.path.join(output_dir, 'best_sumodata.pkl'))

    def run_single_simulation(self):
        """Run a single simulation test."""
        logger.info("Running a single simulation test.")

        real_data = loadfile(self.state.get("real_data_file", ""))
        sim_metrics_file = self.simulation_runner.run_simulation_with_traci(
            self.state.get("net_file", ""),
            self.state.get("rou_file", ""),
            real_data
        )

        score = self.comparator.compare(
            self.state.get("real_data_file", ""),
            self.state.get("real_data_type", "pkl"),
            sim_metrics_file['average_speeds'],
            sim_metrics_file['average_waiting_vehicles'],
            self.state.get("edge_limit_file", "")
        )

        self.state["current_score"] = score
        return score

    def run_multi_scale_test(self):
        """Run multi-scale tests to find the best route scale factor."""
        optimizer_config = self.config.get('optimizer', {})
        test_scales = optimizer_config.get('test_scales', [1.0])

        logger.info(f"Running multi-scale test (exploration_counter: {self.state['exploration_counter']})")

        scale_results = []

        for scale in test_scales:
            logger.info(f"Testing scale: {scale}")

            temp_rou_file = self.scale_adjuster.apply_adjustment(
                self.state.get("rou_file", ""),
                {'scale': scale},
                'testscale'
            )

            real_data = loadfile(self.state.get("real_data_file", ""))
            sim_metrics_file = self.simulation_runner.run_simulation_with_traci(
                self.state.get("net_file", ""),
                temp_rou_file,
                real_data
            )

            score = self.comparator.compare(
                self.state.get("real_data_file", ""),
                self.state.get("real_data_type", "pkl"),
                sim_metrics_file['average_speeds'],
                sim_metrics_file['average_waiting_vehicles'],
                self.state.get("edge_limit_file", "")
            )

            scale_result = {
                'scale': scale,
                'score': score,
                'rou_file': temp_rou_file,
                'sim_metrics': sim_metrics_file
            }
            scale_results.append(scale_result)
            logger.info(f"Scale {scale} score: {score:.4f}")
            self.state["iteration"] += 1

        best_result = min(scale_results, key=lambda x: x['score'])
        self.state["scale_test_results"] = scale_results

        self.state["current_score"] = best_result['score']
        self.state["rou_file"] = best_result['rou_file']

        logger.info(f"Best scale: {best_result['scale']}, score: {best_result['score']:.4f}")

        return best_result

    def run(self):
        """Main simulation task execution flow."""
        try:
            self.state["exploration_counter"] = self.state.get("exploration_counter", 0) + 1

            optimizer_config = self.config.get('optimizer', {})
            exploration_limit = optimizer_config.get('exploration_counter_limit', 10)

            if self.state["exploration_counter"] < exploration_limit:
                best_result = self.run_multi_scale_test()
                self.iterative_optimization()
            else:
                score = self.run_single_simulation()

            self.state["iteration"] = self.state.get("iteration", 0) + 1

            current_score = self.state.get("current_score", float('inf'))
            best_score = self.state.get("best_score", float('inf'))

            if current_score > best_score:
                self.state["best_score"] = current_score
                self.state["best_params"] = {
                    'scale': getattr(self.scale_adjuster, 'current_scale', 1.0),
                    'net_file': self.state.get("net_file", ""),
                    'rou_file': self.state.get("rou_file", "")
                }
                self.state["convergence_count"] = 0
            else:
                self.state["convergence_count"] = self.state.get("convergence_count", 0) + 1

            logger.info(f"Simulation score: {current_score:.4f}, best score: {self.state['best_score']:.4f}")

            return {
                'current_score': current_score,
                'best_score': self.state['best_score'],
                'iteration': self.state['iteration'],
                'state': self.state
            }

        except Exception as e:
            logger.error(f"Error during simulation task: {e}", exc_info=True)
            raise


def main():
    """Standalone entry point for testing."""
    task = SimuTask("configs/simu_task.yml")
    result = task.run()
    logger.info(f"Simulation task completed. Result: {result}")


if __name__ == "__main__":
    main()
