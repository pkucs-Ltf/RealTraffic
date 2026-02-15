"""
变道策略优化任务类
基于opti_Manha.py重构，实现城市道路资源的协同优化工作流
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class LaneChangeTask:
    """变道策略优化任务类 - 完成城市道路资源的协同优化工作流"""
    
    def __init__(self, config_file: str = None, config_dict: Dict = None):
        """
        初始化变道策略优化任务
        
        Args:
            config_file (str): 配置文件路径
            config_dict (Dict): 配置字典，如果提供则优先使用
        """
        if config_dict:
            self.config = config_dict
        elif config_file:
            self.config = self._load_config(config_file)
        else:
            raise ValueError("必须提供config_file或config_dict之一")
        
        if self.config.get('rl_tls_ids_all'):
            self.rl_tls_ids = get_all_tls_ids_from_netxml(self.config.get('data_paths', {}).get('original_net_file'))
        else:
            self.rl_tls_ids = self.config.get('rl_tls_ids', [])
        
        if not self.rl_tls_ids:
            # 可以添加默认的交通灯IDs或者自动检测
            self.rl_tls_ids = []
    
        # 获取配置
        self.city = self.config.get('city', 'Manha')
        self.data_paths = self.config.get('data_paths', {})
        self.simulation_config = self.config.get('simulation', {})
        self.optimization_config = self.config.get('optimization', {})
        self.lane_change_config = self.config.get('lane_change', {})
        self.output_config = self.config.get('output', {})
        
        # 初始化优化运行器
        self.optimization_runner = OptimizationRunner(self.config)
        
        # 动态生成的路径
        run_prefix = self.output_config.get('results_dir_prefix', 'optimization_run')
        self.run_id = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = os.path.join(os.getcwd(), self.run_id)
        self.results_file = os.path.join(self.results_dir, "optimization_summary.json")
        
        # 仿真参数
        self.max_simulations = self.simulation_config.get('max_simulations', 100)
        self.end_time = self.simulation_config.get('end_time', 600)
        self.top_n = self.optimization_config.get('top_n', 12)
        
        self.logger = logging.getLogger("LaneChangeTask")
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}

    def _find_latest_model_by_name(self, model_dir: str, model_name: str) -> Optional[str]:
        """在指定目录中查找特定模型名称的最新episode文件"""
        if not os.path.isdir(model_dir):
            return None
        
        # 查找以{model_name}_episode_开头且以.pth结尾的文件
        checkpoint_files = [f for f in os.listdir(model_dir) 
                          if f.startswith(f'{model_name}_episode_') and f.endswith('.pth')]
        
        if checkpoint_files:
            # 按回合数排序，取最新的
            latest_file = sorted(checkpoint_files, 
                               key=lambda x: int(x.split('_episode_')[1].split('.')[0]))[-1]
            return os.path.join(model_dir, latest_file)
        
        return None

    def generate_final_report(self, all_results):
        """根据所有成功的结果生成最终报告"""
        if not all_results:
            return {}

        report = {
            "best_by_modification_count": {},
            "overall_best": None
        }
        
        # 找到每种修改次数下的最优解
        best_by_mods = {}
        for result in all_results.values():
            mod_count = result['mod_count']
            if mod_count not in best_by_mods or result['reward'] > best_by_mods[mod_count]['reward']:
                best_by_mods[mod_count] = result
        
        report['best_by_modification_count'] = best_by_mods

        # 找到全局最优解
        overall_best = max(all_results.values(), key=lambda x: x['reward'])
        report['overall_best'] = overall_best
        
        return report

    def get_tl_ids_from_net(self, net_file_path: str) -> list:
        """从net.xml文件中轻量化地读取所有交通灯控交叉口的ID"""
        ids = []
        try:
            tree = ET.parse(net_file_path)
            root = tree.getroot()
            for junction in root.findall('junction'):
                if 'traffic_light' in junction.get('type', ''):
                    ids.append(junction.get('id'))
        except ET.ParseError as e:
            self.logger.error(f"无法解析XML文件 {net_file_path}: {e}")
        return ids

    def load_or_generate_initial_data(self):
        """加载或生成初始决策数据"""
        decisions_data_pkl = self.data_paths.get('decisions_data_pkl')
        lane_averages_pkl = self.data_paths.get('lane_averages_pkl')
        
        if os.path.exists(decisions_data_pkl) and os.path.exists(lane_averages_pkl):
            # 加载现有数据
            with open(decisions_data_pkl, 'rb') as f: 
                initial_tl_phase = pickle.load(f)
            with open(lane_averages_pkl, 'rb') as f: 
                initial_lane_avg = pickle.load(f)
            self.logger.info("已加载现有的初始决策数据")
        else:
            # 生成新数据
            self.logger.info("未发现初始数据文件，将在原始路网上运行一次仿真以生成数据。")


            # 运行仿真生成初始数据
            initial_tl_phase, initial_lane_avg = self.optimization_runner.run_optimization_simulation(
                net_file_path=self.data_paths.get('original_net_file'),
                route_file=self.data_paths.get('route_file'),
                reward_static_pkl=self.data_paths.get('reward_static_pkl'),
                rl_tls_ids=self.rl_tls_ids,
                tls_to_train=[]
            )
            self.logger.info("初始数据生成完毕。")
        
        return initial_tl_phase, initial_lane_avg

    def run_optimization_cycle(self, initial_state, max_simulations):
        """运行主优化循环"""
        processing_queue = [initial_state]
        all_successful_results = {"original": initial_state}
        simulation_count = 1  # 已经运行了一次初始仿真
        
        while processing_queue and simulation_count < max_simulations:
            next_processing_queue = []
            
            for current_state in processing_queue:
                self.logger.info(f"\n{'='*20} 正在处理路网 (Mod Cnt: {current_state['mod_count']}, Reward: {current_state['reward']:.2f}) {'='*20}")
                
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
                    # 1. 预处理当前路网
                    temp_parse_dir = processor.prepare_for_analysis(current_state['net_file_path'])
                    tl_ids = processor.get_traffic_light_ids()

                    # 2. 从所有路口搜集所有可行的优化方案
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
                        self.logger.info("在当前状态下未发现任何可行的优化方案。")
                        continue

                    # 3. 全局排序并选择最优的方案
                    all_proposals.sort(key=lambda p: p.get('probability', 0), reverse=True)
                    max_proposals = self.optimization_config.get('max_proposals_per_round', 10)
                    top_proposals = all_proposals[:max_proposals]
                    self.logger.info(f"从 {len(all_proposals)} 个可行方案中选出最优的 {len(top_proposals)} 个进行测试。")

                    # 4. 逐一测试每个顶级方案
                    for best_proposal in top_proposals:
                        if simulation_count >= max_simulations:
                            self.logger.warning("已达到最大仿真次数上限，终止所有循环。")
                            break
                        
                        intersection_id_for_log = best_proposal['intersection_id']
                        self.logger.info(f"  -> 测试路口 [{intersection_id_for_log}] 的一个全局最优方案 (总仿真次数: {simulation_count + 1}/{max_simulations})")
                        
                        temp_net_file, temp_dir = processor.process_single_proposal(
                            current_state['net_file_path'], best_proposal
                        )
                        
                        if not temp_net_file:
                            if temp_dir and os.path.exists(temp_dir): 
                                shutil.rmtree(temp_dir)
                            continue

                        try:
                            # --- 模型管理：复制最新模型到临时目录 ---
                            base_checkpoint_dir = self.optimization_runner.rl_config.get('checkpoint_dir')
                            temp_checkpoint_dir = os.path.join(temp_dir, 'models')
                            os.makedirs(temp_checkpoint_dir, exist_ok=True)
                            
                            tld_dict_path = os.path.join(base_checkpoint_dir, 'tls_dict.pkl')
                            
                            if os.path.exists(tld_dict_path):
                                # 多模型(tld)处理逻辑
                                self.logger.info(f"    -> 检测到tld_dict.pkl，进入多模型加载模式")
                                shutil.copy(tld_dict_path, temp_checkpoint_dir)
                                self.logger.info(f"    -> 已将 tld_dict.pkl 复制到临时目录")

                                with open(tld_dict_path, 'rb') as f:
                                    tld_dict = pickle.load(f)
                                
                                models_found_count = 0
                                for tld_id, model_name_in_dict in tld_dict.items():
                                    latest_model_for_tld = self._find_latest_model_by_name(base_checkpoint_dir, model_name_in_dict)
                                    
                                    if latest_model_for_tld:
                                        shutil.copy(latest_model_for_tld, temp_checkpoint_dir)
                                        self.logger.info(f"    -> 已复制 {tld_id} 的最新模型: {os.path.basename(latest_model_for_tld)}")
                                        models_found_count += 1
                                    else:
                                        self.logger.warning(f"    -> 未在 {base_checkpoint_dir} 找到 {tld_id} 的模型文件")
                                
                                if models_found_count == 0:
                                    self.logger.info(f"    -> 在所有指定目录中均未找到模型，将从头开始训练")


                            # 运行仿真测试
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
                            print('仿真计划',new_reward)
                            new_reward = new_reward[0]
                            print(new_reward,simulation_count)
                            if new_reward > current_state['reward']+30:
                                self.logger.info(f"    -> !!! 性能提升 !!!  (父节点奖励: {current_state['reward']:.4f}, 新奖励: {new_reward:.4f})")
                                 
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
                                self.logger.info(f"    -> 性能未提升或下降 (父节点奖励: {current_state['reward']:.4f}, 新奖励: {new_reward:.4f})，丢弃此分支。")
                                shutil.rmtree(temp_dir)
                        except Exception as e:
                            self.logger.error(f"仿真或评估方案时出错: {e}", exc_info=True)
                            if temp_dir and os.path.exists(temp_dir): 
                                shutil.rmtree(temp_dir)

                finally:
                    # 清理预处理时产生的临时目录
                    if temp_parse_dir and os.path.exists(temp_parse_dir):
                        shutil.rmtree(temp_parse_dir)

                if simulation_count >= max_simulations: 
                    break
            
            processing_queue = next_processing_queue
            
            # 实时更新最终报告
            if self.output_config.get('save_intermediate_results', True):
                final_report = self.generate_final_report(all_successful_results)
                with open(self.results_file, 'w', encoding='utf-8') as f:
                    json.dump(final_report, f, indent=4, ensure_ascii=False)

        return all_successful_results, simulation_count

    def run_optimization_cycle_dfs(self, initial_state, max_simulations):
        """运行基于DFS的主优化循环 - 深度优先探索有效路网"""
        all_successful_results = {"original": initial_state}
        simulation_count = 1  # 已经运行了一次初始仿真
        
        def dfs_explore(current_state, depth=0):
            """DFS递归探索函数"""
            nonlocal simulation_count, all_successful_results
            
            if simulation_count >= max_simulations:
                return
                
            indent = "  " * depth
            self.logger.info(f"\n{indent}{'='*15} DFS探索 (深度: {depth}, Mod Cnt: {current_state['mod_count']}, Reward: {current_state['reward']:.2f}) {'='*15}")
            
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
                # 1. 预处理当前路网
                temp_parse_dir = processor.prepare_for_analysis(current_state['net_file_path'])
                tl_ids = processor.get_traffic_light_ids()

                # 2. 从所有路口搜集所有可行的优化方案
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
                    self.logger.info(f"{indent}在当前状态下未发现任何可行的优化方案，回溯到上层。")
                    return

                # 3. 全局排序并选择最优的方案
                all_proposals.sort(key=lambda p: p.get('probability', 0), reverse=True)
                max_proposals = self.optimization_config.get('max_proposals_per_round', 10)
                top_proposals = all_proposals[:max_proposals]
                self.logger.info(f"{indent}从 {len(all_proposals)} 个可行方案中选出最优的 {len(top_proposals)} 个进行DFS测试。")

                # 4. DFS策略：对每个方案依次进行深度探索
                for i, best_proposal in enumerate(top_proposals):
                    if simulation_count >= max_simulations:
                        self.logger.warning(f"{indent}已达到最大仿真次数上限，终止DFS探索。")
                        break
                    
                    intersection_id_for_log = best_proposal['intersection_id']
                    self.logger.info(f"{indent}-> 测试并深入探索路口 [{intersection_id_for_log}] 的方案 {i+1}/{len(top_proposals)} (总仿真次数: {simulation_count + 1}/{max_simulations})")
                    
                    temp_net_file, temp_dir = processor.process_single_proposal(
                        current_state['net_file_path'], best_proposal
                    )
                    
                    if not temp_net_file:
                        if temp_dir and os.path.exists(temp_dir): 
                            shutil.rmtree(temp_dir)
                        continue

                    try:
                        # --- 模型管理：复制最新模型到临时目录 ---
                        base_checkpoint_dir = self.optimization_runner.rl_config.get('checkpoint_dir')
                        temp_checkpoint_dir = os.path.join(temp_dir, 'models')
                        os.makedirs(temp_checkpoint_dir, exist_ok=True)
                        
                        tld_dict_path = os.path.join(base_checkpoint_dir, 'tls_dict.pkl')
                        
                        if os.path.exists(tld_dict_path):
                            # 多模型(tld)处理逻辑
                            self.logger.info(f"{indent}  -> 检测到tld_dict.pkl，进入多模型加载模式")
                            shutil.copy(tld_dict_path, temp_checkpoint_dir)
                            self.logger.info(f"{indent}  -> 已将 tld_dict.pkl 复制到临时目录")

                            with open(tld_dict_path, 'rb') as f:
                                tld_dict = pickle.load(f)
                            
                            models_found_count = 0
                            for tld_id, model_name_in_dict in tld_dict.items():
                                latest_model_for_tld = self._find_latest_model_by_name(base_checkpoint_dir, model_name_in_dict)
                                
                                if latest_model_for_tld:
                                    shutil.copy(latest_model_for_tld, temp_checkpoint_dir)
                                    self.logger.info(f"{indent}  -> 已复制 {tld_id} 的最新模型: {os.path.basename(latest_model_for_tld)}")
                                    models_found_count += 1
                                else:
                                    self.logger.warning(f"{indent}  -> 未在 {base_checkpoint_dir} 找到 {tld_id} 的模型文件")
                            
                            if models_found_count == 0:
                                self.logger.info(f"{indent}  -> 在所有指定目录中均未找到模型，将从头开始训练")

                        # 运行仿真测试
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
                        print('仿真计划',new_reward)
                        new_reward = new_reward[0]
                        print(new_reward,simulation_count)

                        
                        if new_reward > current_state['reward']+30:
                            self.logger.info(f"{indent}  -> !!! 性能提升 !!!  (父节点奖励: {current_state['reward']:.4f}, 新奖励: {new_reward:.4f})")
                            print('性能提升!!!',new_reward,current_state['reward'])
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
                            
                            # 实时更新最终报告
                            if self.output_config.get('save_intermediate_results', True):
                                final_report = self.generate_final_report(all_successful_results)
                                with open(self.results_file, 'w', encoding='utf-8') as f:
                                    json.dump(final_report, f, indent=4, ensure_ascii=False)
                            
                            # DFS核心：立即深入探索这个有效的新状态
                            self.logger.info(f"{indent}  -> 立即深入探索这个改进的状态...")
                            dfs_explore(new_state, depth + 1)
                            
                            # 如果达到仿真次数限制，提前结束
                            if simulation_count >= max_simulations:
                                break
                                
                        else:
                            self.logger.info(f"{indent}  -> 性能未提升或下降 (父节点奖励: {current_state['reward']:.4f}, 新奖励: {new_reward:.4f})，尝试下一个方案。")
                            shutil.rmtree(temp_dir)
                            
                    except Exception as e:
                        self.logger.error(f"{indent}仿真或评估方案时出错: {e}", exc_info=True)
                        if temp_dir and os.path.exists(temp_dir): 
                            shutil.rmtree(temp_dir)

            finally:
                # 清理预处理时产生的临时目录
                if temp_parse_dir and os.path.exists(temp_parse_dir):
                    shutil.rmtree(temp_parse_dir)
        
        # 开始DFS探索
        self.logger.info("=== 开始基于DFS的深度优先优化探索 ===")
        dfs_explore(initial_state)
        
        return all_successful_results, simulation_count

    def run(self):
        if self.config.get('use_dfs', False):
            return self.run_with_dfs()
        """运行变道策略优化任务主流程"""
        os.makedirs(self.results_dir, exist_ok=True)
        self.logger.info(f"本次优化运行ID: {self.run_id}")
        
        # 1. 初始数据准备
        self.logger.info("--- 检查初始决策数据 ---")
        initial_tl_phase, initial_lane_avg = self.load_or_generate_initial_data()
        
        # 2. 初始化迭代参数
        # a. 计算原始路网的基准奖励
        reward_static_pkl = self.data_paths.get('reward_static_pkl')
        initial_reward = get_average_reward(reward_static_pkl)
        initial_reward = initial_reward
        self.logger.info(f"原始路网基准奖励为: {initial_reward:.4f}")

        # c. 初始化处理队列和结果存储
        initial_state = {
            "id": "original",
            "net_file_path": os.path.abspath(self.data_paths.get('original_net_file')),
            "reward": initial_reward,
            "mod_count": 0,
            "tl_phase_result": initial_tl_phase,
            "lane_averages": initial_lane_avg,
            "proposal_history": []
        }
        
        # 3. 主优化循环
        all_successful_results, simulation_count = self.run_optimization_cycle(
            initial_state, self.max_simulations
        )

        # 4. 结束并生成最终报告
        self.logger.info(f"\n{'='*20} 所有优化轮次结束 {'='*20}")
        self.logger.info(f"共进行了 {simulation_count} 次仿真。")
        self.logger.info(f"共发现 {len(all_successful_results) - 1} 个有效的优化版本。")
        
        if self.output_config.get('generate_summary_report', True):
            final_report = self.generate_final_report(all_successful_results)
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=4, ensure_ascii=False)
            self.logger.info(f"最终结果摘要已保存至: {self.results_file}")

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
        """运行变道策略优化任务主流程 - 使用DFS深度优先探索"""
        os.makedirs(self.results_dir, exist_ok=True)
        self.logger.info(f"本次优化运行ID (DFS模式): {self.run_id}")
        
        # 1. 初始数据准备
        self.logger.info("--- 检查初始决策数据 ---")
        initial_tl_phase, initial_lane_avg = self.load_or_generate_initial_data()
        
        # 2. 初始化迭代参数
        # a. 计算原始路网的基准奖励
        reward_static_pkl = self.data_paths.get('reward_static_pkl')
        initial_reward = get_average_reward(reward_static_pkl)
        initial_reward = initial_reward
        self.logger.info(f"原始路网基准奖励为: {initial_reward:.4f}")
        print('原始路网基准奖励为:',initial_reward)
        # c. 初始化处理队列和结果存储
        initial_state = {
            "id": "original",
            "net_file_path": os.path.abspath(self.data_paths.get('original_net_file')),
            "reward": initial_reward,
            "mod_count": 0,
            "tl_phase_result": initial_tl_phase,
            "lane_averages": initial_lane_avg,
            "proposal_history": []
        }
        
        # 3. 主优化循环 - 使用DFS版本
        all_successful_results, simulation_count = self.run_optimization_cycle_dfs(
            initial_state, self.max_simulations
        )

        # 4. 结束并生成最终报告
        self.logger.info(f"\n{'='*20} DFS优化轮次结束 {'='*20}")
        self.logger.info(f"共进行了 {simulation_count} 次仿真。")
        self.logger.info(f"共发现 {len(all_successful_results) - 1} 个有效的优化版本。")
        
        if self.output_config.get('generate_summary_report', True):
            final_report = self.generate_final_report(all_successful_results)
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=4, ensure_ascii=False)
            self.logger.info(f"最终结果摘要已保存至: {self.results_file}")

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
    """主执行函数 - 用于独立测试"""
    task = LaneChangeTask("configs/lane_change_task.yml")
    result = task.run()
    print(f"变道策略优化任务完成，结果: {result}")


if __name__ == "__main__":
    main()
