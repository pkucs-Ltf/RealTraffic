"""
强化学习交通信号控制器模块
实现策略模式的控制器抽象层，支持多种控制策略的无缝切换
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import pickle
import os
import time


class TrafficLightController(ABC):
    """交通信号控制器抽象基类"""
    
    @abstractmethod
    def decide_action(self, tl_id: str, observation: Dict, timestep: int) -> int:
        """
        决策下一个相位
        
        Args:
            tl_id: 交通灯ID
            observation: 当前观测状态
            timestep: 当前时间步
            
        Returns:
            int: 选择的相位ID
        """
        pass
    
    @abstractmethod
    def on_step_end(self, tl_id: str, transition: Dict):
        """
        步骤结束回调，用于训练
        
        Args:
            tl_id: 交通灯ID
            transition: 状态转移信息 {state, action, reward, next_state, done}
        """
        pass
    
    @abstractmethod
    def reset(self, tl_ids: List[str]):
        """
        重置控制器状态
        
        Args:
            tl_ids: 需要重置的交通灯ID列表
        """
        pass


class GreedyController(TrafficLightController):
    """贪心控制器 - 包装现有的MaxPressure逻辑"""
    
    def __init__(self, ltf_instance):
        """
        初始化贪心控制器
        
        Args:
            ltf_instance: LTF_Traci实例，用于调用原有的get_max_pressure_action方法
        """
        self.ltf = ltf_instance
        
    def decide_action(self, tl_id: str, observation: Dict, timestep: int, is_training_agent: bool = True) -> int:
        """使用MaxPressure算法决策"""
        return self.ltf.get_max_pressure_action(tl_id)
    
    def on_step_end(self, tl_id: str, transition: Dict):
        """贪心控制器不需要学习"""
        pass
    
    def reset(self, tl_ids: List[str]):
        """贪心控制器无状态，无需重置"""
        pass


class StaticController(TrafficLightController):
    """固定配时控制器"""
    
    def __init__(self, ltf_instance):
        """
        初始化固定配时控制器
        
        Args:
            ltf_instance: LTF_Traci实例，包含raw_phase_data
        """
        self.ltf_instance = ltf_instance
        self.current_program_index = {}  # tl_id -> 当前程序索引
        self.current_program_time = {}   # tl_id -> 当前程序运行时间
        
    def decide_action(self, tl_id: str, observation: Dict, timestep: int, is_training_agent: bool = True) -> int:
        """根据预设配时方案决策"""
        # 如果没有该交通灯的原始相位数据，返回当前相位
        if tl_id not in self.ltf_instance.raw_phase_data:
            return observation.get('current_phase', 0)
            
        # 初始化该交通灯的配时状态
        if tl_id not in self.current_program_index:
            self.current_program_index[tl_id] = 0
            self.current_program_time[tl_id] = 0
            
        # 获取当前相位程序
        current_index = self.current_program_index[tl_id]
        raw_phases = self.ltf_instance.raw_phase_data[tl_id]
        
        # 如果没有相位数据，返回当前相位
        if not raw_phases:
            return observation.get('current_phase', 0)
            
        # 获取当前相位和持续时间
        current_phase_info = raw_phases[current_index]
        phase_obj, phase_duration = current_phase_info
        
        # 增加当前相位运行时间
        self.current_program_time[tl_id] += 1
        
        # 检查是否需要切换到下一个相位
        if self.current_program_time[tl_id] >= phase_duration:
            # 切换到下一个相位
            self.current_program_index[tl_id] = (current_index + 1) % len(raw_phases)
            self.current_program_time[tl_id] = 0
            
        # 返回当前应该执行的相位索引
        return self.current_program_index[tl_id]
        
    
    def on_step_end(self, tl_id: str, transition: Dict):
        """固定配时控制器不需要学习"""
        pass
    
    def reset(self, tl_ids: List[str]):
        """重置配时程序状态"""
        for tl_id in tl_ids:
            # 重置所有交通灯的配时状态
            self.current_program_index[tl_id] = 0
            self.current_program_time[tl_id] = 0


class StateExtractor:
    """状态提取器 - 负责从SUMO环境中提取RL所需的状态特征"""
    
    def __init__(self, intersections: Dict, conn, rl_tls_ids: List[str]):
        """
        初始化状态提取器
        
        Args:
            intersections: 路口信息字典
            conn: SUMO连接对象
            rl_tls_ids: 参与RL控制的路口ID列表
        """
        self.intersections = intersections
        self.conn = conn
        self.rl_tls_ids = rl_tls_ids
        self.state_history = defaultdict(deque)  # 历史状态缓存
        
    def extract_observation(self, tl_id: str, current_phases: Dict, current_phase_times: Dict) -> np.ndarray:
        """
        提取单个路口的完整状态向量
        
        Args:
            tl_id: 交通灯ID
            current_phases: 当前相位字典
            current_phase_times: 当前相位时间字典
            
        Returns:
            np.ndarray: 动态维度状态向量 [道路状态(N维) + 相位状态(M维)]
                       N = 该路口实际进入车道数量
                       M = 该路口实际相位数量
        """
        features = []
        
        # 第一部分：道路状态 - 每个进入车道的车辆数
        road_features = self._extract_local_traffic_features(tl_id)
        features.extend(road_features)
        
        # # 第二部分：相位状态 - 当前相位的one-hot编码
        phase_features = self._extract_signal_state_features(tl_id, current_phases, current_phase_times)
        features.extend(phase_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_local_traffic_features(self, tl_id: str) -> List[float]:
        """提取局部交通流量特征 - 动态提取每个进入车道的车辆数"""
        features = []
        
        if tl_id not in self.intersections:
            return []
        
        # 获取所有进入车道（start_lanes）并按车道ID排序以保持一致性
        start_lanes = set()
        for phase_links in self.intersections[tl_id]['phase_available_lanelinks']:
            for start_lane, end_lane in phase_links[1]:
                start_lanes.add(start_lane)
        
        # 按车道ID排序，确保每次提取的顺序一致
        sorted_lanes = sorted(list(start_lanes))
        
        # 为每个进入车道提取车辆数
        for lane_id in sorted_lanes:
            try:
                vehicle_count = self.conn.lane.getLastStepVehicleNumber(lane_id)
                # 归一化车辆数（假设每个车道最大容量100辆）
                normalized_count = min(vehicle_count / 100.0, 1.0)
                features.append(normalized_count)
            except Exception:
                features.append(0.0)
        
        return features
    
    
    def _extract_signal_state_features(self, tl_id: str, current_phases: Dict, current_phase_times: Dict) -> List[float]:
        """提取信号状态特征 - 动态相位one-hot编码"""
        features = []
        
        if tl_id not in self.intersections or tl_id not in current_phases:
            return []
        
        # 获取该路口的实际相位数量
        actual_phases = len(self.intersections[tl_id]['phase_available_lanelinks'])
        current_phase = current_phases[tl_id]
        
        # 创建动态长度的one-hot编码
        phase_onehot = [0.0] * actual_phases
        
        # 确保相位索引在有效范围内
        if 0 <= current_phase < actual_phases:
            phase_onehot[current_phase] = 1.0
        
        features.extend(phase_onehot)
        
        return features
    


class RewardCalculator:
    """三层奖励函数计算器"""
    
    def __init__(self, intersections: Dict, conn, rl_tls_ids: List[str], reward_config: Dict):
        """
        初始化奖励计算器
        
        Args:
            intersections: 路口信息字典
            conn: SUMO连接对象
            rl_tls_ids: 参与RL控制的路口ID列表
            reward_config: 奖励配置参数
        """
        self.intersections = intersections
        self.conn = conn
        self.rl_tls_ids = rl_tls_ids
        self.reward_config = reward_config
        
        # 权重配置
        self.local_weight = reward_config.get('local_weight', 0.75)
        self.rl_group_weight = reward_config.get('rl_group_weight', 0.15)
        self.global_weight = reward_config.get('global_weight', 0.10)
    
    def compute_reward(self, tl_id: str) -> float:
        """计算三层组合奖励"""
        # 第一层：局部路口奖励（主要）
        local_reward = 0.0
        if self.reward_config.get('use_local_queue', True):
            local_reward += self.local_queue_penalty(tl_id)
        # if self.reward_config.get('use_local_waiting', True):
        #     local_reward += self.local_waiting_time_penalty(tl_id)
        if self.reward_config.get('use_local_throughput', True):
            local_reward += self.local_throughput_reward(tl_id)
        
        # # 第二层：RL路口群体效率
        # rl_group_reward = self.rl_group_efficiency_reward()
        
        # # 第三层：全局效率
        # global_reward = self.global_efficiency_reward()
        
        # 组合奖励
        total_reward = (self.local_weight * local_reward )
        
        # 奖励裁剪
        reward_clip = self.reward_config.get('reward_clip', [-100, 100])
        total_reward = np.clip(total_reward, reward_clip[0], reward_clip[1])
        
        return total_reward
    
    def local_queue_penalty(self, tl_id: str) -> float:
        """计算该路口的排队长度惩罚"""
        if tl_id not in self.intersections:
            return 0.0
        
        penalty = 0
        try:
            for phase_links in self.intersections[tl_id]['phase_available_lanelinks']:
                for start_lane, _ in phase_links[1]:
                    penalty += self.conn.lane.getLastStepHaltingNumber(start_lane)
        except Exception:
            pass
        
        return -penalty
    
    def local_waiting_time_penalty(self, tl_id: str) -> float:
        """计算该路口相关车辆的等待时间惩罚"""
        if tl_id not in self.intersections:
            return 0.0
        
        penalty = 0
        try:
            relevant_vehicles = self._get_relevant_vehicles(tl_id)
            for vehicle_id in relevant_vehicles:
                penalty += self.conn.vehicle.getWaitingTime(vehicle_id)
        except Exception:
            pass
        
        return -penalty / 100.0  # 归一化
    
    def local_throughput_reward(self, tl_id: str) -> float:
        """计算该路口的通行效率奖励"""
        if tl_id not in self.intersections:
            return 0.0
        
        throughput = 0
        try:
            for phase_links in self.intersections[tl_id]['phase_available_lanelinks']:
                for start_lane, end_lane in phase_links[1]:
                    # 通过该连接的车辆数
                    throughput += self.conn.lane.getLastStepVehicleNumber(end_lane)
        except Exception:
            pass
        
        return throughput
    
    def rl_group_efficiency_reward(self) -> float:
        """计算所有RL控制路口的平均通行效率"""
        if not self.rl_tls_ids:
            return 0.0
        
        total_efficiency = 0
        valid_count = 0
        
        for rl_tl_id in self.rl_tls_ids:
            if rl_tl_id in self.intersections:
                try:
                    # 计算该RL路口的效率指标
                    local_throughput = 0
                    local_waiting = 0
                    
                    for phase_links in self.intersections[rl_tl_id]['phase_available_lanelinks']:
                        for start_lane, end_lane in phase_links[1]:
                            local_throughput += self.conn.lane.getLastStepVehicleNumber(end_lane)
                            local_waiting += self.conn.lane.getLastStepHaltingNumber(start_lane)
                    
                    # 效率 = 通行量 - 排队惩罚
                    efficiency = local_throughput - local_waiting * 0.5
                    total_efficiency += efficiency
                    valid_count += 1
                    
                except Exception:
                    pass
        
        return total_efficiency / valid_count if valid_count > 0 else 0.0
    
    def global_efficiency_reward(self) -> float:
        """计算全网所有路口的平均通行效率"""
        total_efficiency = 0
        valid_count = 0
        
        for tl_id in self.intersections.keys():
            try:
                # 计算该路口的效率指标
                local_throughput = 0
                local_waiting = 0
                
                for phase_links in self.intersections[tl_id]['phase_available_lanelinks']:
                    for start_lane, end_lane in phase_links[1]:
                        local_throughput += self.conn.lane.getLastStepVehicleNumber(end_lane)
                        local_waiting += self.conn.lane.getLastStepHaltingNumber(start_lane)
                
                # 效率 = 通行量 - 排队惩罚
                efficiency = local_throughput - local_waiting * 0.5
                total_efficiency += efficiency
                valid_count += 1
                
            except Exception:
                pass
        
        return total_efficiency / valid_count if valid_count > 0 else 0.0
    
    def _get_relevant_vehicles(self, tl_id: str) -> List[str]:
        """获取与该路口相关的车辆ID列表"""
        relevant_vehicles = []
        
        if tl_id not in self.intersections:
            return relevant_vehicles
        
        try:
            # 获取该路口控制的所有车道上的车辆
            for phase_links in self.intersections[tl_id]['phase_available_lanelinks']:
                for start_lane, end_lane in phase_links[1]:
                    # 起始车道的车辆
                    vehicles_on_start = self.conn.lane.getLastStepVehicleIDs(start_lane)
                    relevant_vehicles.extend(vehicles_on_start)
                    
                    # 结束车道的车辆
                    vehicles_on_end = self.conn.lane.getLastStepVehicleIDs(end_lane)
                    relevant_vehicles.extend(vehicles_on_end)
        except Exception:
            pass
        
        return list(set(relevant_vehicles))  # 去重


class TrafficLightControllerManager:
    """交通信号控制器管理器 - 统一管理不同类型的控制器"""
    
    def __init__(self, tls_ids: List[str], intersections: Dict, 
                 rl_mode: str = 'none', rl_tls_ids: List[str] = None,
                 non_rl_policy: str = 'greedy', training: bool = False,
                 checkpoint_dir: str = None, rl_config: Dict = None,
                 ltf_instance=None, need_dump_tlc: bool = False,given_tls_dict: dict = None):
        """
        初始化控制器管理器
        
        Args:
            tls_ids: 所有交通灯ID列表
            intersections: 路口信息字典
            rl_mode: RL模式 ('none', 'dqn', 'ppo')
            rl_tls_ids: 参与RL控制的路口ID列表
            non_rl_policy: 非RL路口的控制策略 ('greedy', 'static')
            training: 是否为训练模式
            checkpoint_dir: 模型保存/加载路径
            rl_config: RL配置参数
            ltf_instance: LTF_Traci实例
        """
        self.tls_ids = tls_ids
        self.intersections = intersections
        self.rl_mode = rl_mode
        self.rl_tls_ids = rl_tls_ids or []
        self.non_rl_policy = non_rl_policy
        self.training = training
        self.checkpoint_dir = checkpoint_dir
        self.rl_config = rl_config or {}
        self.ltf_instance = ltf_instance
        # self.static_list=[]
        # self
        # 控制器字典
        self.controllers = {}
        self.need_dump_tlc = need_dump_tlc
        self.given_tls_dict = given_tls_dict
        # tl_id到短名的映射，避免文件名超长
        if self.given_tls_dict:
            self.tls_dict = self.given_tls_dict
        else:
            self.tls_dict = {tl_id: f'junction_{i+1}' for i, tl_id in enumerate(self.tls_ids)}
        
        # INSERT_YOUR_CODE
        # 反查tls_dict，即已知junction_x名也能查回原始tl_id，提供新字典tls_dict_reverse
        self.tls_dict_reverse = {v: k for k, v in self.tls_dict.items()}
        
        # 初始化控制器
        self._init_controllers()
        
        # 初始化状态提取器和奖励计算器（如果使用RL）
        if self.rl_mode != 'none' and self.ltf_instance:
            self.state_extractor = StateExtractor(
                intersections, self.ltf_instance.conn, self.rl_tls_ids
            )
            
            self.reward_calculator = RewardCalculator(
                intersections, self.ltf_instance.conn, self.rl_tls_ids,
                self.rl_config.get('reward_config', {})
            )
        else:
            self.state_extractor = None
            self.reward_calculator = None
        
        # 存储上一步的状态和动作（用于训练）
        self.prev_states = {}
        self.prev_actions = {}
    
    def _init_controllers(self):
        """初始化各种控制器"""
        # 为每个路口分配控制器
        for tl_id in self.tls_ids:
            if tl_id in self.rl_tls_ids and self.rl_mode != 'none':
                # RL控制器
                self.controllers[tl_id] = self._create_rl_controller(tl_id)
            elif self.non_rl_policy == 'greedy':
                # 贪心控制器
                self.controllers[tl_id] = GreedyController(self.ltf_instance)
            elif self.non_rl_policy == 'static':
                # 静态控制器
                self.controllers[tl_id] = StaticController(self.ltf_instance)
            else:
                # 默认使用贪心控制器
                self.controllers[tl_id] = StaticController(self.ltf_instance)
    
    def _create_rl_controller(self, tl_id: str):
        """为指定路口创建RL控制器"""
        # 计算状态和动作维度
        state_dim = self._calculate_state_dim(tl_id)
        action_dim = len(self.intersections[tl_id]['phase_available_lanelinks'])
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.need_dump_tlc:
            with open(os.path.join(self.checkpoint_dir, 'tls_dict.pkl'), 'wb') as f:
                    pickle.dump(self.tls_dict, f)
        # 创建RL控制器
        if self.rl_mode == 'dqn':
            return DQNController(tl_id, state_dim, action_dim, 
                               self.rl_config.get('dqn_params', {}),
                               self.training, self.checkpoint_dir)
        elif self.rl_mode == 'ppo':
            return PPOController(tl_id, state_dim, action_dim,
                               self.rl_config.get('ppo_params', {}),
                               self.training, self.checkpoint_dir)

    
    def _calculate_state_dim(self, tl_id: str) -> int:
        """动态计算单个路口的状态维度"""
        
        # 计算道路状态维度：实际进入车道数量
        start_lanes = set()
        for phase_links in self.intersections[tl_id]['phase_available_lanelinks']:
            for start_lane, end_lane in phase_links[1]:
                start_lanes.add(start_lane)
        road_state_dim = len(start_lanes)
        
        # 计算相位状态维度：实际相位数量
        phase_state_dim = len(self.intersections[tl_id]['phase_available_lanelinks'])
        
        total_dim = road_state_dim + phase_state_dim
        print(f"路口 {tl_id}: {road_state_dim}个进入车道 + {phase_state_dim}个相位 = {total_dim}维状态空间")
        
        return total_dim
    
    def _load_static_programs(self) -> Dict:
        """加载静态配时方案"""
        try:
            if self.checkpoint_dir and os.path.exists(os.path.join(self.checkpoint_dir, 'programs.pkl')):
                with open(os.path.join(self.checkpoint_dir, 'programs.pkl'), 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"加载静态配时方案失败: {e}")
        
        return {}
    
    def decide_action(self, tl_id: str, observation: Dict, timestep: int, is_training_agent: bool = True) -> int:
        """统一的动作决策接口"""
        if tl_id not in self.controllers:
            # 如果没有对应的控制器，使用默认相位
            return observation.get('current_phase', 0)
        
        # 如果是RL控制器且需要状态提取
        if (tl_id in self.rl_tls_ids and self.rl_mode != 'none' and 
            self.state_extractor and self.ltf_instance):
            
            # 提取RL状态
            rl_observation = self.state_extractor.extract_observation(
                tl_id, self.ltf_instance.current_phases, self.ltf_instance.current_phase_times
            )
            observation = rl_observation

            # ===== 动态动作掩码 =====
            # 规则：
            # - 若未达最小绿灯时长(t_min)，仅允许保持当前相位（避免PPO on-policy偏差）
            # - 否则，仅允许有可通行链接(绿/直行)的相位；若极端为全False，则放开全部
            try:
                phase_links_list = self.intersections[tl_id]['phase_available_lanelinks']
                base_mask = np.array([len(links) > 0 for _, links in phase_links_list], dtype=bool)
                current_phase = self.ltf_instance.current_phases.get(tl_id, 0)
                if self.ltf_instance.current_phase_times.get(tl_id, 0) < self.ltf_instance.t_min:
                    mask = np.zeros(len(phase_links_list), dtype=bool)
                    if 0 <= current_phase < len(phase_links_list):
                        mask[current_phase] = True
                    else:
                        mask[:] = True  # 兜底
                else:
                    mask = base_mask if base_mask.any() else np.ones(len(phase_links_list), dtype=bool)
            except Exception:
                # 兜底：全部可用
                mask = np.ones(len(self.intersections[tl_id]['phase_available_lanelinks']), dtype=bool)

            controller = self.controllers[tl_id]
            if hasattr(controller, 'agent'):
                return controller.agent.predict(
                    observation,
                    mask,
                    deterministic=not is_training_agent
                )

        return self.controllers[tl_id].decide_action(tl_id, observation, timestep, is_training_agent)
    
    def on_step_end(self, timestep: int):
        """步骤结束处理，主要用于RL训练"""
        if not self.training or self.rl_mode == 'none' or not self.reward_calculator:
            return
        
        # 为每个RL路口计算奖励并存储经验
        for tl_id in self.rl_tls_ids:
            if tl_id in self.controllers and tl_id in self.prev_states:
                # 计算奖励
                reward = self.reward_calculator.compute_reward(tl_id)
                
                # 获取当前状态
                current_state = self.state_extractor.extract_observation(
                    tl_id, self.ltf_instance.current_phases, self.ltf_instance.current_phase_times
                )
                
                # 构建转移信息
                transition = {
                    'state': self.prev_states[tl_id],
                    'action': self.prev_actions.get(tl_id, 0),
                    'reward': reward,
                    'next_state': current_state,
                    'done': False  # 在仿真结束时会设置为True
                }
                
                # 传递给控制器进行学习
                self.controllers[tl_id].on_step_end(tl_id, transition)
    
    def reset(self):
        """重置所有控制器"""
        for controller in self.controllers.values():
            controller.reset(self.tls_ids)
        
        # 清空历史状态
        self.prev_states.clear()
        self.prev_actions.clear()
    
    def set_eval_mode(self):
        """将所有RL控制器切换为评测模式（关闭Dropout/BN随机）。"""
        for tl_id in self.rl_tls_ids:
            controller = self.controllers.get(tl_id)
            if controller and hasattr(controller, 'set_eval_mode'):
                controller.set_eval_mode()

    def set_train_mode(self):
        """将所有RL控制器切换回训练模式。"""
        for tl_id in self.rl_tls_ids:
            controller = self.controllers.get(tl_id)
            if controller and hasattr(controller, 'set_train_mode'):
                controller.set_train_mode()

    def save_checkpoint(self, episode: int):
        """保存检查点"""
        if not self.checkpoint_dir or self.rl_mode == 'none':
            return
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 保存每个RL控制器的模型
        for tl_id in self.rl_tls_ids:
            if tl_id in self.controllers:
                controller = self.controllers[tl_id]
                if hasattr(controller, 'save'):
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'{self.tls_dict[tl_id]}_episode_{episode}.pth')
                    controller.save(checkpoint_path)
        
        print(f"已保存第 {episode} 回合的检查点到 {self.checkpoint_dir}")
    
    def load_checkpoint(self):
        """加载检查点"""
        if not self.checkpoint_dir or self.rl_mode == 'none':
            return
        
        # 为每个RL控制器加载模型
        for tl_id in self.rl_tls_ids:
            if tl_id in self.controllers:
                controller = self.controllers[tl_id]
                if hasattr(controller, 'load'):
                    # 查找最新的检查点文件
                    checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                                      if f.startswith(f'{self.tls_dict[tl_id]}_episode_') and f.endswith('.pth')]
                    if checkpoint_files:
                        # 按回合数排序，取最新的
                        latest_file = sorted(checkpoint_files, 
                                           key=lambda x: int(x.split('_episode_')[1].split('.')[0]))[-1]
                        checkpoint_path = os.path.join(self.checkpoint_dir, latest_file)
                        controller.load(checkpoint_path)
                        print(f"已为路口 {tl_id} 加载检查点: {latest_file}")

    def save_checkpoint_partial(self, episode: int, tls_to_train: List[str]):
        """保存部分训练的检查点，自动查找最大的episode编号+1进行保存"""
        if not self.checkpoint_dir or self.rl_mode == 'none':
            return
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        new_episode_num = episode
        for tl_id in tls_to_train:
            if tl_id in self.controllers:
                # 查找当前tl_id所有已存在部分训练文件，获取最大episode编号
                checkpoint_files = [f for f in os.listdir(self.checkpoint_dir)
                                   if f.startswith(f'{self.tls_dict[tl_id]}_episode_') and f.endswith('.pth')]
                max_episode = -1
                for fname in checkpoint_files:
                    try:
                        idx = fname.split('_episode_')[1].split('.')[0]
                        idx_int = int(idx)
                        if idx_int > max_episode:
                            max_episode = idx_int
                    except Exception:
                        continue
                # 最大编号+1作为新episode编号
                save_episode = max_episode + 2
                controller = self.controllers[tl_id]
                if hasattr(controller, 'save'):
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'{self.tls_dict[tl_id]}_episode_{save_episode}.pth')
                    controller.save(checkpoint_path)
                    new_episode_num = save_episode  # 只记录最后一个的编号（如果有多个tl_id会复写）

        print(f"已为 {tls_to_train} 保存第 {new_episode_num} 回合的部分训练检查点到 {self.checkpoint_dir}")

    def load_checkpoint_partial(self, tls_to_train: List[str]):
        """加载部分训练的检查点"""
        if not self.checkpoint_dir or self.rl_mode == 'none':
            return
        tls_to_load=[]
        for k in self.rl_tls_ids:
            if k not in tls_to_train:
                tls_to_load.append(k)
        # 为指定的RL控制器加载模型
        for tl_id in tls_to_load:
            if tl_id in self.controllers:
                controller = self.controllers[tl_id]
                if hasattr(controller, 'load'):
                    # 查找最新的部分训练检查点文件
                    checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                                      if f.startswith(f'{self.tls_dict[tl_id]}_episode_') and f.endswith('.pth')]
                    if checkpoint_files:
                        # 按回合数排序，取最新的
                        latest_file = sorted(checkpoint_files, 
                                           key=lambda x: int(x.split('_episode_')[1].split('.')[0]))[-1]
                        checkpoint_path = os.path.join(self.checkpoint_dir, latest_file)
                        controller.load(checkpoint_path)
                        print(f"已为路口 {tl_id} 加载部分训练检查点: {latest_file}")


from .rl_algorithms import DQNAgent, PPOAgent


class DQNController(TrafficLightController):
    """DQN强化学习控制器"""
    
    def __init__(self, tl_id: str, state_dim: int, action_dim: int, 
                 dqn_params: Dict, training: bool, checkpoint_dir: str):
        self.tl_id = tl_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training = training
        self.checkpoint_dir = checkpoint_dir
        
        # 创建DQN智能体
        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **dqn_params
        )
        
        # 如果不是训练模式，尝试加载模型
        if not training and checkpoint_dir:
            self.load_latest_model()
    
    def decide_action(self, tl_id: str, observation: Dict, timestep: int, is_training_agent: bool = True) -> int:
        """DQN动作决策"""
        
        state = observation
        
        # 生成动作掩码（确保动作在有效范围内）
        mask = np.ones(self.action_dim, dtype=bool)
        
        # 预测动作
        action = self.agent.predict(state, mask, deterministic=not is_training_agent)
        
        return action
    
    def on_step_end(self, tl_id: str, transition: Dict):
        """存储经验并学习"""
        if self.training:
            # 存储经验
            self.agent.store_transition(
                state=transition['state'],
                action=transition['action'],
                reward=transition['reward'],
                next_state=transition['next_state'],
                done=transition['done']
            )
            
            # 学习更新
            metrics = self.agent.learn()
            
            # 可以在这里记录训练指标
            if metrics:
                pass  # 可以添加日志记录
    
    def reset(self, tl_ids: List[str]):
        """重置DQN状态"""
        # DQN通常不需要重置，因为它使用经验回放
        pass
    
    def save(self, path: str):
        """保存DQN模型"""
        self.agent.save(path)
    
    def load(self, path: str):
        """加载DQN模型"""
        self.agent.load(path)
    
    def set_eval_mode(self):
        """评测模式（关闭Dropout等随机性）。"""
        try:
            self.agent.q_network.eval()
            self.agent.target_network.eval()
        except Exception:
            pass

    def set_train_mode(self):
        """训练模式。"""
        try:
            self.agent.q_network.train()
            self.agent.target_network.train()
        except Exception:
            pass

    def load_latest_model(self):
        """加载最新的模型"""
        if not self.checkpoint_dir or not os.path.exists(self.checkpoint_dir):
            return
        
        # 查找最新的检查点文件
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith(f'{self.tls_dict[self.tl_id]}_episode_') and f.endswith('.pth')]
        if checkpoint_files:
            # 按回合数排序，取最新的
            latest_file = sorted(checkpoint_files, 
                               key=lambda x: int(x.split('_episode_')[1].split('.')[0]))[-1]
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_file)
            self.load(checkpoint_path)


class PPOController(TrafficLightController):
    """PPO强化学习控制器"""
    
    def __init__(self, tl_id: str, state_dim: int, action_dim: int,
                 ppo_params: Dict, training: bool, checkpoint_dir: str):
        self.tl_id = tl_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.training = training
        self.checkpoint_dir = checkpoint_dir
        
        # 创建PPO智能体
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **ppo_params
        )
        
        # 如果不是训练模式，尝试加载模型
        if not training and checkpoint_dir:
            self.load_latest_model()
    
    def decide_action(self, tl_id: str, observation: Dict, timestep: int, is_training_agent: bool = True) -> int:
        """PPO动作决策"""
        
        state = observation
        
        # 生成动作掩码（确保动作在有效范围内）
        mask = np.ones(self.action_dim, dtype=bool)
        
        # 预测动作
        action = self.agent.predict(state, mask, deterministic=not is_training_agent)
        
        return action
    
    def on_step_end(self, tl_id: str, transition: Dict):
        """存储轨迹并学习"""
        if self.training:
            # 存储经验
            self.agent.store_transition(
                state=transition['state'],
                action=transition['action'],
                reward=transition['reward'],
                next_state=transition['next_state'],
                done=transition['done']
            )
            
            # 学习更新（PPO会在收集足够经验后自动学习）
            metrics = self.agent.learn()
            
            # 可以在这里记录训练指标
            if metrics:
                pass  # 可以添加日志记录
    
    def reset(self, tl_ids: List[str]):
        """重置PPO状态"""
        self.agent.reset_buffer()
    
    def save(self, path: str):
        """保存PPO模型"""
        self.agent.save(path)
    
    def load(self, path: str):
        """加载PPO模型"""
        self.agent.load(path)
    
    def load_latest_model(self):
        """加载最新的模型"""
        if not self.checkpoint_dir or not os.path.exists(self.checkpoint_dir):
            return
        
        # 查找最新的检查点文件
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith(f'{self.tls_dict[self.tl_id]}_episode_') and f.endswith('.pth')]
        if checkpoint_files:
            # 按回合数排序，取最新的
            latest_file = sorted(checkpoint_files, 
                               key=lambda x: int(x.split('_episode_')[1].split('.')[0]))[-1]
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_file)
            self.load(checkpoint_path)

    def set_eval_mode(self):
        """评测模式（关闭Dropout等随机性）。"""
        try:
            self.agent.network.eval()
        except Exception:
            pass

    def set_train_mode(self):
        """训练模式。"""
        try:
            self.agent.network.train()
        except Exception:
            pass
