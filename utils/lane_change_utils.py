"""
变道优化相关工具函数
包含GreedyLaneChanger类和get_average_reward函数
"""

import pickle
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import logging
import subprocess
import xml.etree.ElementTree as ET
import json
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import math
import uuid
import shutil

def load_metrics(file_path):
    """加载所有迭代的指标数据"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_average_reward(file_='pkl/reward_static_all.pkl'):
    """
    提取每个指标在每个迭代、每个指定时间点的值。
    返回: metrics_over_time[metric][iteration][time_index]
    """
    all_metrics=load_metrics(file_)
    # 动态提取 all_metrics[0] 中所有的 time 字段，按升序排序作为 time_points
    time_points = sorted([item['time'] for item in all_metrics[0] if 'time' in item])
    metric_names = ['vehicles', 'waiting_time', 'queue_length', 'mean_speed', 'mean_travel_time','arrival_carnum']

    metrics_over_time = {metric: [] for metric in metric_names}
    value_list=[]
    for iteration_data in all_metrics:
        # iteration_data: list of dicts, each dict有'time'和各指标
        values_at_times = {metric: [] for metric in metric_names}
        for t in time_points:
            data_at_t = next((item for item in iteration_data if item.get('time') == t), None)
            for metric in metric_names:
                if data_at_t is not None:
                    values_at_times[metric].append(data_at_t.get(metric))
        value_list.append([
            values_at_times['arrival_carnum'][-1],
            values_at_times['mean_speed'][-1],
            values_at_times['mean_travel_time'][-1],
            values_at_times['queue_length'][-1],
            values_at_times['waiting_time'][-1],
            values_at_times['vehicles'][-1]
        ])
    
    # 找到value_list中第一个元素（arrival_carnum）最大的那个元素，并返回该元素
    if not value_list:
        return None
    value_list_new = [i[0] for i in value_list if i[0] > 0]
    max_item = np.max(value_list_new)
    print(value_list)
    return max_item 
    # max_item = max(value_list, key=lambda x: x[0])
    # return max_item

# def get_average_reward(pickle_path: str) -> float:
#     """
#     读取 reward pickle 文件，计算并返回第一代的平均奖励值。

#     Args:
#         pickle_path (str): pkl/reward_static_all.pkl 文件的路径。

#     Returns:
#         float: 第一代所有交叉口的平均奖励值。如果文件不存在或为空，返回一个极小值。
#     """
#     if not os.path.exists(pickle_path):
#         print(f"错误: Pickle 文件不存在 at {pickle_path}")
#         return -float('inf')

#     with open(pickle_path, 'rb') as f:
#         data = pickle.load(f)
    
#     if not data or not isinstance(data, list) or not data[0]:
#         print(f"错误: Pickle 文件 {pickle_path} 中没有有效数据。")
#         return -float('inf')

#     # 计算第一代的所有路口奖励平均值
#     first_gen_rewards = data[0]
#     average_value = np.mean([np.mean(rewards) for rewards in first_gen_rewards.values()])
    
#     return average_value


def get_average_reward1(pickle_path: str) -> float:
    """
    读取 reward pickle 文件，计算并返回第一代的平均奖励值。（备用版本）

    Args:
        pickle_path (str): pkl/reward_static_all.pkl 文件的路径。

    Returns:
        float: 第一代所有交叉口的平均奖励值。如果文件不存在或为空，返回一个极小值。
    """
    if not os.path.exists(pickle_path):
        print(f"错误: Pickle 文件不存在 at {pickle_path}")
        return -float('inf' )

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    if not data or not isinstance(data, list) or not data[0]:
        print(f"错误: Pickle 文件 {pickle_path} 中没有有效数据。")
        return -float('inf')

    # 计算第一代的所有路口奖励平均值
    first_gen_rewards = data[0]
    average_value = np.mean([np.mean(rewards) for rewards in first_gen_rewards.values()])
    
    return average_value

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交叉口贪心潮汐变道策略（不依赖 LLM）
基于 changelane.md 中描述的算法实现
"""
import warnings
warnings.filterwarnings("ignore")
import logging
import os
import subprocess
import xml.etree.ElementTree as ET
import json
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import math
import uuid
import shutil


class GreedyLaneChanger:
    """
    贪心潮汐变道策略实现类
    
    主要功能：
    1. 解析 SUMO 路网文件（.net.xml 及其拆分文件）
    2. 根据相位压力和偏好相位进行贪心决策
    3. 生成新增连接建议并合并为新路网文件
    """
    
    def __init__(self, logger=None, lane_obeservation=None,top_n=5):
        self.logger = logger or logging.getLogger(__name__)
        
        # 存储解析后的路网数据
        self.nodes = {}  # node_id -> node_info
        self.edges = {}  # edge_id -> edge_info  
        self.connections = {}  # (from, to, fromLane, toLane) -> connection_info
        self.traffic_lights = {}  # tl_id -> tl_info
        
        self.raw_netfile=""
        # 存储分析结果
        self.connection_scores = {}  # connection_key -> score
        self.edge_pressures = {}  # edge_id -> pressure
        self.lane_utilizations = {}  # (edge_id, lane_idx) -> utilization
        self.lane_obeservation = lane_obeservation  # lane_id -> utilization
        self.tl_to_edges = {}  # tl_id -> edge_id
        self.top_n=top_n
        
    def get_netconvert_files(self, net_file: str) -> Dict[str, str]:
        """
        根据网络文件路径生成对应的基础文件路径
        """
        if net_file.endswith(".net.xml"):
            base_name = net_file[:-8]
        else:
            base_name, _ = os.path.splitext(net_file)
            
        return {
            "net": net_file,
            "node": f"{base_name}.nod.xml",
            "edge": f"{base_name}.edg.xml", 
            "connection": f"{base_name}.con.xml",
            "tll": f"{base_name}.tll.xml"
        }
    
    def split_network_file(self, net_file_path: str, output_prefix: str) -> None:
        """
        使用 netconvert 拆分网络文件
        """
        split_cmd = f"netconvert --sumo-net-file {net_file_path} --plain-output-prefix {output_prefix}"
        
        ret = subprocess.run(split_cmd, shell=True)
        if ret.returncode != 0:
            raise Exception(f"netconvert 拆分命令执行失败: {split_cmd}")
    
    def parse_network_files(self, base_prefix: str) -> None:
        """
        解析所有基础网络文件
        """
        files = self.get_netconvert_files(base_prefix)
        
        # 解析节点文件
        self._parse_nodes(files["node"])
        
        # 解析边文件
        self._parse_edges(files["edge"])
        
        # 解析连接文件
        self._parse_connections(files["connection"])
        
        # 解析信号灯文件
        self._parse_traffic_lights(files["tll"])
        
    
    def _parse_nodes(self, node_file: str) -> None:
        """解析节点文件"""
        try:
            tree = ET.parse(node_file)
            root = tree.getroot()
            
            for node in root.findall('node'):
                node_id = node.get('id')
                self.nodes[node_id] = {
                    'id': node_id,
                    'x': float(node.get('x', 0)),
                    'y': float(node.get('y', 0)),
                    'type': node.get('type', ''),
                    'tl': node.get('tl', ''),  # 信号灯ID
                }
        except Exception as e:
            raise Exception(f"解析节点文件 {node_file} 出错: {e}")
    
    def _parse_edges(self, edge_file: str) -> None:
        """解析边文件"""
        try:
            
            edge_file=self.raw_netfile
            tree = ET.parse(edge_file)
            root = tree.getroot()
            
            for edge in root.findall('edge'):
                edge_id = edge.get('id')
                # 跳过内部边（SUMO内部计算用，以 ':' 开头或 function="internal" 标记）
                if edge_id and edge_id.startswith(':'):
                    continue
                if edge.get('function', '') == 'internal':
                    continue
                
                # 解析车道信息
                lanes = []
                for lane in edge.findall('lane'):
                    lane_info = {
                        'id': lane.get('id'),
                        'index': int(lane.get('index', 0)),
                        'speed': float(lane.get('speed', 13.89)),
                        'length': float(lane.get('length', 0)),
                        'width': float(lane.get('width', 3.2)),
                        'shape': lane.get('shape', ''),
                        'allow': lane.get('allow', ''),
                        'disallow': lane.get('disallow', ''),
                    }
                    lanes.append(lane_info)
                
                self.edges[edge_id] = {
                    'id': edge_id,
                    'from': edge.get('from'),
                    'to': edge.get('to'),
                    'priority': int(edge.get('priority', 1)),
                    'numLanes': int(edge.get('numLanes', len(lanes))),
                    'speed': float(edge.get('speed', 13.89)),
                    'lanes': lanes,
                }
        except Exception as e:
            raise Exception(f"解析边文件 {edge_file} 出错: {e}")
    
    def _parse_connections(self, connection_file: str) -> None:
        """解析连接文件"""
        try:
            self.connections={}
            tree = ET.parse(connection_file)
            root = tree.getroot()
            
            for conn in root.findall('connection'):
                from_edge = conn.get('from')
                to_edge = conn.get('to')
                from_lane = int(conn.get('fromLane', 0))
                to_lane = int(conn.get('toLane', 0))
                
                conn_key = (from_edge, to_edge, from_lane, to_lane)
                
                self.connections[conn_key] = {
                    'from': from_edge,
                    'to': to_edge,
                    'fromLane': from_lane,
                    'toLane': to_lane,
                    'via': conn.get('via', ''),
                    'tl': conn.get('tl', ''),
                    'linkIndex': int(conn.get('linkIndex', -1)),
                    'dir': conn.get('dir', ''),
                    'state': conn.get('state', ''),
                }
        except Exception as e:
            raise Exception(f"解析连接文件 {connection_file} 出错: {e}")
    
    def _parse_traffic_lights(self, tll_file: str) -> None:
        """解析信号灯文件"""
        try:
            tree = ET.parse(tll_file)
            root = tree.getroot()
            
            for tl_logic in root.findall('tlLogic'):
                tl_id = tl_logic.get('id')
                
                # 解析相位信息
                phases = []
                for phase in tl_logic.findall('phase'):
                    phase_info = {
                        'duration': float(phase.get('duration', 0)),
                        'state': phase.get('state', ''),
                        'minDur': float(phase.get('minDur', 0)),
                        'maxDur': float(phase.get('maxDur', 0)),
                    }
                    phases.append(phase_info)
                
                self.traffic_lights[tl_id] = {
                    'id': tl_id,
                    'type': tl_logic.get('type', ''),
                    'programID': tl_logic.get('programID', ''),
                    'offset': float(tl_logic.get('offset', 0)),
                    'phases': phases,
                }
                
            # 解析连接到信号灯的映射
            for conn in root.findall('connection'):
                tl_id = conn.get('tl')
                if tl_id in self.traffic_lights:
                    if 'connections' not in self.traffic_lights[tl_id]:
                        self.traffic_lights[tl_id]['connections'] = []
                    self.traffic_lights[tl_id]['connections'].append({
                        'from': conn.get('from'),
                        'to': conn.get('to'),
                        'fromLane': int(conn.get('fromLane', 0)),
                        'toLane': int(conn.get('toLane', 0)),
                        'linkIndex': int(conn.get('linkIndex', -1)),
                    })
                    
        except Exception as e:
            raise Exception(f"解析信号灯文件 {tll_file} 出错: {e}")
    
    def build_connection_mapping(self, intersection_id: str) -> Dict[int, Tuple[str, str, int, int]]:
        """
        建立连接序号到具体连接的映射
        
        Returns:
            Dict[int, Tuple]: linkIndex -> (from_edge, to_edge, from_lane, to_lane)
        """
        if intersection_id not in self.traffic_lights:
            return {}
        
        tl_info = self.traffic_lights[intersection_id]
        mapping = {}
        
        # 从信号灯连接信息中建立映射
        if 'connections' in tl_info:
            for conn in tl_info['connections']:
                link_index = conn['linkIndex']
                if link_index >= 0:
                    mapping[link_index] = (
                        conn['from'],
                        conn['to'], 
                        conn['fromLane'],
                        conn['toLane']
                    )
        
        return mapping
    
    def calculate_connection_scores_from_preferred_phase(self, intersection_id: str, 
                                                       preferred_phase: str, 
                                                       connection_pressures: Dict[int, float]) -> Dict[Tuple, float]:
        """
        根据偏好相位和每个连接位置的具体压力计算连接评分
        
        Args:
            intersection_id: 路口ID
            preferred_phase: 偏好相位状态字符串
            connection_pressures: 连接位置压力字典 {link_index: pressure}
            
        Returns:
            Dict: connection_key -> score
        """
        if intersection_id not in self.traffic_lights:
            return {}
        
        connection_mapping = self.build_connection_mapping(intersection_id)
        scores = {}
        
        # 遍历偏好相位中的每个字符位置
        for link_idx, char in enumerate(preferred_phase):
            # 只处理绿灯(G)的连接
            if (char.upper() == 'G' or char.upper() == 'S') and link_idx in connection_mapping:
                conn_tuple = connection_mapping[link_idx]
                # 直接使用给定的连接压力
                pressure = connection_pressures.get(link_idx, 0.0)
                scores[conn_tuple] = pressure
        
        return scores
    
    def calculate_edge_pressures(self, connection_scores: Dict[Tuple, float]) -> Dict[str, float]:
        """
        计算边级压力
        
        Args:
            connection_scores: 连接评分字典
            
        Returns:
            Dict: edge_id -> pressure
        """
        edge_pressures = defaultdict(float)
        
        for (from_edge, to_edge, from_lane, to_lane), score in connection_scores.items():
            edge_pressures[from_edge] += score
            
        return dict(edge_pressures)
    
    def calculate_lane_utilizations(self, intersection_id: str) -> Dict[Tuple[str, int], int]:
        """
        计算车道利用度（该车道上的连接数量）
        
        Args:
            intersection_id: 路口ID
            
        Returns:
            Dict: (edge_id, lane_index) -> utilization_count
        """
        utilizations = defaultdict(int)
        
        # 获取与该路口相关的所有连接
        tl_connections = []
        if intersection_id in self.traffic_lights and 'connections' in self.traffic_lights[intersection_id]:
            tl_connections = self.traffic_lights[intersection_id]['connections']
        
        for conn in tl_connections:
            from_edge = conn['from']
            from_lane = conn['fromLane']
            utilizations[(from_edge, from_lane)] += 1
            
        return dict(utilizations)
    
    def get_direction_priority(self, from_edge: str, to_edge: str) -> int:
        """
        获取转向优先级（直行 > 右转 > 左转）
        
        Returns:
            int: 优先级数值，越小优先级越高
        """
        # 这里简化处理，实际应该根据几何位置计算转向类型
        # 暂时返回固定值，实际使用时需要根据边的几何关系判断
        return 1
    
    def get_turn_type(self, from_edge_id: str, to_edge_id: str) -> Optional[str]:
        """
        根据边的几何关系计算转向类型
        
        Args:
            from_edge_id: 入口边ID
            to_edge_id: 出口边ID
            
        Returns:
            Optional[str]: 'straight', 'left', 'right', 'uturn' 或 None
        """
        try:
            from_edge = self.edges[from_edge_id]
            to_edge = self.edges[to_edge_id]

            # 获取三个关键节点
            from_node_id = from_edge['from']
            intersection_node_id = from_edge['to'] 
            to_node_id = to_edge['to']
            
            # 确保是同一个交叉口
            if to_edge['from'] != intersection_node_id:
                return None

            from_node = self.nodes[from_node_id]
            intersection_node = self.nodes[intersection_node_id]
            to_node = self.nodes[to_node_id]

            # 计算入口和出口向量
            vec_in = (intersection_node['x'] - from_node['x'], intersection_node['y'] - from_node['y'])
            vec_out = (to_node['x'] - intersection_node['x'], to_node['y'] - intersection_node['y'])

            # 计算向量角度
            angle_in = math.atan2(vec_in[1], vec_in[0])
            angle_out = math.atan2(vec_out[1], vec_out[0])

            # 计算角度差并转为度数
            angle_diff = math.degrees(angle_out - angle_in)
            
            # 标准化到 [-180, 180]
            if angle_diff > 180:
                angle_diff -= 360
            if angle_diff < -180:
                angle_diff += 360

            # 根据角度判断转向类型 (阈值可调)
            if abs(angle_diff) < 25:
                return 'straight'
            elif angle_diff > 0:
                if angle_diff > 155:
                    return 'uturn'
                return 'left'
            else: # angle_diff < 0
                if angle_diff < -155:
                    return 'uturn'
                return 'right'

        except KeyError as e:
            return None
        except Exception as e:
            return None
            
    def validate_turn_decision(self, from_edge_id: str, to_edge_id: str, 
                               from_lane_idx: int) -> bool:
        """
        验证一个转向决策的合理性
        
        Args:
            from_edge_id: 入口边ID
            to_edge_id: 出口边ID
            from_lane_idx: 选择的入口车道索引
            
        Returns:
            bool: 如果合理则为 True，否则为 False
        """
        # 1. 计算目标转向类型
        target_turn_type = self.get_turn_type(from_edge_id, to_edge_id)
        if not target_turn_type:
            return False
        
        # 2. 约束检查1: 转向几何与车道物理位置匹配
        from_edge = self.edges.get(from_edge_id)
        if not from_edge:
            return False
        
        num_lanes = from_edge['numLanes']
        # 假设车道索引从左到右为 0, 1, 2, ...
        # 定义功能区（最左侧车道 index=0, 最右侧车道 index=num_lanes-1)
        is_left_area = (from_lane_idx == 0)
        is_right_area = (from_lane_idx == num_lanes - 1)
        
        if target_turn_type == 'left' and not is_left_area:
            return False
            
        if target_turn_type == 'right' and not is_right_area:
            return False
            
        # 3. 约束检查2: 新增转向与车道现有功能不冲突
        existing_turn_types = set()
        for conn_from, conn_to, conn_from_lane, _ in self.connections.keys():
            if conn_from == from_edge_id and conn_from_lane == from_lane_idx:
                turn_type = self.get_turn_type(conn_from, conn_to)
                if turn_type:
                    existing_turn_types.add(turn_type)
        
        
        # 检查冲突：一个车道不能同时存在左转和右转
        if 'left' in existing_turn_types and target_turn_type == 'right':
            # self.logger.warning(f"不合理约束2: 不能为已有左转功能的车道增加右转功能")
            return False
        
        if 'right' in existing_turn_types and target_turn_type == 'left':
            # self.logger.warning(f"不合理约束2: 不能为已有右转功能的车道增加左转功能")
            return False

        # 新增约束：直行和左转不能在同一个车道
        if ('straight' in existing_turn_types and target_turn_type == 'left') or \
           ('left' in existing_turn_types and target_turn_type == 'straight'):
            # self.logger.warning(f"不合理约束3: 直行和左转不能在同一个车道 (现有功能: {existing_turn_types}, 尝试新增: {target_turn_type})")
            return False

        # if target_turn_type in existing_turn_types:
        #     self.logger.info(f"注意：为车道 {from_edge_id}_{from_lane_idx} 增加一个已存在的转向类型 '{target_turn_type}'")

        return True
    
    def select_top_connections(self, intersection_id: str, preferred_phase: str, 
                               connection_pressures: Dict[int, float], top_n: int = 5) -> List[Tuple[Tuple, float]]:
        """
        选择最佳的 top_n 个连接
        
        Returns:
            List[Tuple]: 列表，每个元素为 ((from_edge, to_edge, from_lane, to_lane), score)
        """
        if intersection_id not in self.traffic_lights:
            return []
        
        # 计算连接评分
        connection_scores = self.calculate_connection_scores_from_preferred_phase(
            intersection_id, preferred_phase, connection_pressures)
        if not connection_scores:
            return []
        
        # 计算边压力
        edge_pressures = self.calculate_edge_pressures(connection_scores)
        
        # 计算车道利用度
        lane_utilizations = self.calculate_lane_utilizations(intersection_id)
        
        # 计算所有连接的综合评分
        all_scored_connections = []
        for conn_tuple, pressure_score in connection_scores.items():
            from_edge, to_edge, from_lane, to_lane = conn_tuple
            if len(self.edges[from_edge]['lanes'])<2:
                continue
                
            
            score = pressure_score
            edge_pressure = edge_pressures.get(from_edge, 0)
            lane_util = lane_utilizations.get((from_edge, from_lane), 0)
            direction_priority = self.get_direction_priority(from_edge, to_edge)
            
            combined_score = (score * 1000 + edge_pressure * 100 - 
                            lane_util * 10 - direction_priority)
            
            all_scored_connections.append((conn_tuple, combined_score))
        
        # 按评分降序排序
        all_scored_connections.sort(key=lambda x: x[1], reverse=True)
        
        # self.logger.info(f"路口 {intersection_id} 找到 {len(all_scored_connections)} 个候选连接，返回前 {top_n} 个")
        return all_scored_connections[:top_n]
    
    def select_best_from_lane(self, from_edge: str, target_connection: Tuple[str, str, int, int]) -> Optional[Tuple[int, float]]:
        """
        选择最佳出发车道（利用度最低的可用车道）
        
        Args:
            from_edge: 进入边ID
            target_connection: 目标连接
            
        Returns:
            Optional[Tuple[int, float]]: (最佳车道索引, 该车道压力) 或 None
        """
        if from_edge not in self.edges:
            return None
        
        edge_info = self.edges[from_edge]
        lanes = edge_info['lanes']
        
 
        # 找到可用车道
        candidate_lanes = []
        for lane in lanes:
            lane_idx = lane['index']
            if lane_idx == target_connection[2]:
                continue
            # 检查车辆类别限制
            allow = lane.get('allow', '')
            disallow = lane.get('disallow', '')
            
            # 简化处理：如果没有特殊限制，认为可用
            if not disallow or 'passenger' not in disallow:
                pressure = self.lane_obeservation.get(lane.get('id'), 0.0) # 使用 .get 避免 KeyError
                candidate_lanes.append((lane_idx, pressure, lane))
        
        if not candidate_lanes:
            return None
        
        # 按压力排序(升序)，选择压力最低的
        candidate_lanes.sort(key=lambda x: (x[1], -x[2]['speed'], -x[2]['width'], x[0]))
        
        best_lane_idx = candidate_lanes[0][0]
        best_lane_pressure = candidate_lanes[0][1]
        
        # self.logger.info(f"边 {from_edge} 选择最佳出发车道: {best_lane_idx}, "
        #                 f"压力: {best_lane_pressure}")
        
        return best_lane_idx, best_lane_pressure
    
    def generate_fan_out_connections(self, best_connection: Tuple[str, str, int, int], 
                                   best_from_lane: int) -> List[Tuple[str, str, int, int]]:
        """
        生成扇出连接建议
        
        Args:
            best_connection: 最佳连接 (from_edge, to_edge, from_lane, to_lane)
            best_from_lane: 最佳出发车道
            
        Returns:List[Tuple]: 新增连接列表
            List[Tuple]: 新增连接列表
        """
        from_edge, to_edge, _, _ = best_connection
        
        if to_edge not in self.edges:
            return []
        
        to_edge_info = self.edges[to_edge]
        to_lanes = to_edge_info['lanes']
        
        new_connections = []
        
        # 遍历目标边的所有车道
        for to_lane in to_lanes:
            to_lane_idx = to_lane['index']
            
            # 检查车辆类别限制
            allow = to_lane.get('allow', '')
            disallow = to_lane.get('disallow', '')
            
            # 简化处理：如果没有特殊限制，认为可用
            if disallow and 'passenger' in disallow:
                continue
            
            new_conn = (from_edge, to_edge, best_from_lane, to_lane_idx)
            
            # 检查是否已存在
            if new_conn not in self.connections:
                new_connections.append(new_conn)
        
        self.logger.info(f"生成 {len(new_connections)} 个新连接: {new_connections}")
        return new_connections
    
    def process_intersection(self, intersection_id: str, preferred_phase: str, 
                           connection_pressures: Dict[int, float]) -> List[Dict]:
        """
        处理单个路口，找出所有合理的优化方案
        
        Args:
            intersection_id: 路口ID
            preferred_phase: 偏好相位状态字符串
            connection_pressures: 连接位置压力字典 {link_index: pressure}
            
        Returns:
            List[Dict]: 合理的优化方案列表
        """
        # self.logger.info(f"开始处理路口 {intersection_id}")

        # 1. 选择 Top-N 候选连接
        top_connections = self.select_top_connections(intersection_id, preferred_phase, connection_pressures)
        if not top_connections:
            return []
        
        # 2. 筛选所有合理的优化方案
        valid_proposals = []
        for connection, score in top_connections:
            from_edge = connection[0]
            
            # 2a. 为该连接选择最佳出发车道
            result = self.select_best_from_lane(from_edge, connection)
            if result is None:
                self.logger.warning(f"路口 {intersection_id} 无法为连接 {connection} 选择出发车道")
                continue
            
            best_from_lane, lane_pressure = result
            
            # 2b. 对方案进行合理性验证
            from_edge, to_edge, _, _ = connection
            is_turn_reasonable = self.validate_turn_decision(
                from_edge, to_edge, best_from_lane
            )

            # 2c. 如果合理，则加入方案列表
            if is_turn_reasonable:
                proposal = {
                    "intersection_id": intersection_id, # 增加路口ID
                    "connection": connection,
                    "from_lane": best_from_lane,
                    "lane_pressure": lane_pressure,
                    "original_score": score,
                    'probability': score/lane_pressure if lane_pressure > 0 else float('inf')
                }
                valid_proposals.append(proposal)
            else:
                pass
                
        return valid_proposals
    
    def prepare_for_analysis(self, net_file_path: str) -> str:
        """
        准备分析所需的文件和数据。
        执行拆分和解析，并返回用于清理的临时目录路径。
        """
        if self.raw_netfile=="":
            self.raw_netfile=net_file_path
        net_file_path = os.path.abspath(net_file_path)
        base_prefix = os.path.splitext(os.path.basename(net_file_path))[0]
        base_prefix=os.path.basename(net_file_path).split('.')[0]
        a=os.path.basename(net_file_path)
        # Create a unique temp directory for this analysis
        temp_dir = os.path.join(os.path.dirname(net_file_path), f"temp_analysis_{uuid.uuid4().hex[:8]}")
        os.makedirs(temp_dir, exist_ok=True)
        
        split_prefix = os.path.join(temp_dir, base_prefix)
        
        self.logger.info(f"Preparing analysis in temporary directory: {temp_dir}")
        self.split_network_file(net_file_path, split_prefix)
        self.parse_network_files(split_prefix)
        
        return temp_dir
        
    def get_traffic_light_ids(self) -> List[str]:
        """返回已解析路网中所有交通信号灯的ID列表"""
        return list(self.traffic_lights.keys())

    # def process_single_proposal(self, net_file_path: str, proposal_to_apply: Dict,split_prefix: str,iter_dir: str) -> Tuple[str, str]:
    #     """
    #     仅针对一个给定的优化方案执行完整的文件生成流程。
    #     返回 (最终生成的路网文件路径, 临时工作目录路径)。
    #     """
    #     # 1. 准备工作目录
       
      
    #     try:
    #         # # 2. 拆分
    #         # self.split_network_file(net_file_path, split_prefix)

    #         # 3. 应用单个方案
    #         all_new_connections = defaultdict(list)
            
    #         new_connections = self.generate_fan_out_connections(proposal_to_apply['connection'], proposal_to_apply['from_lane'])
            
    #         if new_connections:
    #             intersection_id = proposal_to_apply['intersection_id']
    #             all_new_connections[intersection_id].extend(new_connections)

    #         # 4. 更新与合并
    #         if all_new_connections:
    #             connection_file = f"{split_prefix}.con.xml"
    #             # 需要重新解析原始连接文件来更新
    #             # self._parse_connections(f"{split_prefix}.con.xml")
    #             self.update_connection_file(connection_file, all_new_connections)
            
    #         output_net_file = os.path.join(iter_dir, "modified.net.xml")
    #         self.merge_network_files(split_prefix, output_net_file)
            
    #         return output_net_file, iter_dir

    #     except Exception as e:
    #         self.logger.error(f"处理单个方案时出错: {e}", exc_info=True)
    #         if os.path.exists(iter_dir):
    #             shutil.rmtree(iter_dir)
    #         return None, None

    def update_connection_file(self, connection_file: str, new_connections: Dict[str, List[Tuple]]) -> None:
        """
        更新连接文件，添加新的连接
        
        Args:
            connection_file: 连接文件路径
            new_connections: 新连接字典 {intersection_id: [connections]}
        """
        try:
            tree = ET.parse(connection_file)
            root = tree.getroot()
        except Exception as e:
            raise Exception(f"解析连接文件 {connection_file} 出错: {e}")
        
        # 为每个路口添加新连接
        for intersection_id, connections in new_connections.items():
            if not connections:
                continue
            
            # 添加注释标记
            comment = ET.Comment(f"=== 贪心潮汐变道策略新增连接 - 路口 {intersection_id} ===")
            root.append(comment)
            
            # 添加新连接
            for from_edge, to_edge, from_lane, to_lane in connections:
                conn_elem = ET.Element('connection')
                conn_elem.set('from', from_edge)
                conn_elem.set('to', to_edge)
                conn_elem.set('fromLane', str(from_lane))
                conn_elem.set('toLane', str(to_lane))
                root.append(conn_elem)
            
            # 添加结束注释
            end_comment = ET.Comment(f"=== 路口 {intersection_id} 新增连接结束 ===")
            root.append(end_comment)
        
        # 保存文件
        tree.write(connection_file, encoding="utf-8", xml_declaration=True)
    
    def merge_network_files(self, base_prefix: str, output_file: str) -> None:
        """
        合并网络文件生成新的 .net.xml
        
        Args:
            base_prefix: 基础文件前缀
            output_file: 输出文件路径
        """
        files = self.get_netconvert_files(base_prefix)
        
        merge_cmd = (
            f"netconvert "
            f"--node-files {files['node']} "
            f"--edge-files {files['edge']} " 
            f"--connection-files {files['connection']} "
            f"--output-file {output_file} "
            f"--tls.layout incoming"
        )
        
        self.logger.info(f"执行合并命令: {merge_cmd}")
        ret = subprocess.run(merge_cmd, shell=True)
        if ret.returncode != 0:
            raise Exception(f"netconvert 合并命令执行失败: {merge_cmd}")
        
        self.logger.info(f"成功生成新路网文件: {output_file}")
        
    def match_tl_to_edge(self, net_file: str) -> Dict[str, Set[str]]:
        """
        从 .net.xml 文件中匹配信号灯(junction)到其入口道路(edge)。

        Args:
            net_file: SUMO路网文件 (.net.xml) 的路径。

        Returns:
            一个字典，键是信号灯ID，值是与该信号灯关联的入口edge ID集合。
        """
        tree = ET.parse(net_file)
        root = tree.getroot()

        tl_to_edges = {}

        # 遍历所有的junction
        for junction in root.findall('junction'):
            # 如果type中含有'traffic_light'字样，就确定这个交叉口是信号灯
            if 'traffic_light' in junction.get('type', ''):
                tl_id = junction.get('id')
                # incLanes是他的进入车道
                inc_lanes_str = junction.get('incLanes')
                if not tl_id or not inc_lanes_str:
                    continue

                inc_lanes = inc_lanes_str.split()
                
                # 每个item[:-2]就是他的进入道路，集合去个重就是红绿灯对应的道路集合
                incoming_edges = set()
                for lane in inc_lanes:
                    # SUMO车道ID格式为 'edgeID_laneIndex'。使用rsplit获取edgeID部分
                    if '_' in lane:
                        edge_id = lane.rsplit('_', 1)[0]
                        incoming_edges.add(edge_id)
                
                if incoming_edges:
                    # 用字典存下来每个信号灯的对应道路
                    tl_to_edges[tl_id] = incoming_edges
        
        self.tl_to_edges = tl_to_edges
        
    def _get_connections_from_tree(self, root: ET.Element) -> Set[Tuple[str, str, int, int]]:
        """从XML树中提取所有连接信息"""
        connections = set()
        for conn in root.findall('connection'):
            if conn.get('from') and conn.get('to'):
                connections.add((
                    conn.get('from'),
                    conn.get('to'),
                    int(conn.get('fromLane')),
                    int(conn.get('toLane'))
                ))
        return connections

    def _get_link_maps_for_tl(self, root: ET.Element, tl_id: str) -> Tuple[Dict[int, Tuple], Dict[Tuple, int]]:
        """为单个信号灯创建linkIndex到连接和反向的映射"""
        link_to_conn = {}
        conn_to_link = {}
        # 寻找与信号灯关联的交叉口
        junction = root.find(f".//junction[@id='{tl_id}']")
        if junction is None:
            return link_to_conn, conn_to_link

        for req in junction.findall('request'):
            link_index = int(req.get('index'))
            # response 字符串的长度应该与 incLanes 的车道数匹配
            # 这里需要找到这个 link index 对应的具体连接
            # 这通常在 connection 元素中定义，但 junction 定义中更直接
            # 我们需要一种方法从 index 映射回 (from, to, fromLane, toLane)
            # SUMO netconvert 会按顺序分配 index，但依赖这个顺序很脆弱
            # 最可靠的方式是从 <connection> 标签中直接读取 tl 和 linkIndex
            
        # 改为从 aconnection 标签中寻找
        for conn in root.findall(f".//connection[@tl='{tl_id}']"):
            link_index = conn.get('linkIndex')
            if link_index is not None:
                link_index = int(link_index)
                conn_tuple = (
                    conn.get('from'),
                    conn.get('to'),
                    int(conn.get('fromLane')),
                    int(conn.get('toLane'))
                )
                link_to_conn[link_index] = conn_tuple
                conn_to_link[conn_tuple] = link_index
        return link_to_conn, conn_to_link
    def has_g(self,state_list):
                            # 判断字符列表中是否有小写g
        return any(c.lower() == 'g' for c in state_list)
    def modify_tl_logic(self, net_file_path: str, output_net_file: str, output_net_file_filt: str, processed_from_lanes: Set[Tuple[str, str]]):
        """
        修改交通信号灯逻辑。
        未改变的TL沿用旧逻辑，改变的TL在旧逻辑基础上智能增改。
        """

        # 1. 解析新旧路网文件
        orig_tree = ET.parse(net_file_path)
        orig_root = orig_tree.getroot()
        new_tree = ET.parse(output_net_file)
        new_root = new_tree.getroot()
        
        

        # 2. 识别需要修改的信号灯集合
        edge_to_tl_map = {edge: tl for tl, edges in self.tl_to_edges.items() for edge in edges}
        tls_modifyset = set()
        for from_edge, _ in processed_from_lanes:
            if from_edge in edge_to_tl_map:
                tls_modifyset.add(edge_to_tl_map[from_edge])
        self.logger.info(f"识别出需要修改逻辑的信号灯: {tls_modifyset}")

        # 3. 提取新旧路网的所有连接
        orig_conns_set = self._get_connections_from_tree(orig_root)
        new_conns_set = self._get_connections_from_tree(new_root)
        added_conns_set = new_conns_set - orig_conns_set

        # 4. 遍历新路网中的所有信号灯，进行替换或修改
        all_new_tls = new_root.findall('.//tlLogic')
        for new_tl_logic in all_new_tls:
            tl_id = new_tl_logic.get('id')
           
            
            orig_tl_logic = orig_root.find(f".//tlLogic[@id='{tl_id}']")
            if orig_tl_logic is None:
                continue

            if tl_id not in tls_modifyset:
                # 案例A: 未修改的信号灯，完全替换为旧逻辑
                parent_map = {c: p for p in new_root.iter() for c in p}
                parent = parent_map.get(new_tl_logic)
                if parent is not None:
                    parent.remove(new_tl_logic)
                    parent.append(orig_tl_logic)
            else:
                
                # 获取新旧连接与linkIndex的映射
                _, orig_conn_to_link = self._get_link_maps_for_tl(orig_root, tl_id)
                new_link_to_conn, new_conn_to_link = self._get_link_maps_for_tl(new_root, tl_id)
                
                # 找到该TL新增的连接
                added_conns_for_tl = {conn for conn in added_conns_set if conn in new_conn_to_link}
                
                orig_phases = orig_tl_logic.findall('phase')
                new_phases_elements = []

                for orig_phase in orig_phases:
                    orig_state = list(orig_phase.get('state'))
                    # INSERT_YOUR_CODE
                    if not self.has_g(orig_state):
                        continue
                    # 首先，将所有右转灯强制变绿
                    for link, conn in self._get_link_maps_for_tl(orig_root, tl_id)[0].items():
                         if self.get_turn_type(conn[0], conn[1]) == 'right' and orig_state[link] == 'r':
                             if link < len(orig_state):
                                 orig_state[link] = 'g'
                    
                    modified_orig_state = "".join(orig_state)

                    # 创建新相位的状态列表，默认全红
                    num_new_links = len(new_link_to_conn)
                    final_state = ['r'] * num_new_links

                    # 继承旧连接的相位状态
                    for conn, old_link in orig_conn_to_link.items():
                
                        if conn in new_conn_to_link:
                            new_link = new_conn_to_link[conn]
                            final_state[new_link] = modified_orig_state[old_link]

                    # 为新增连接寻找参考并设置相位
                    for add_conn in added_conns_for_tl:
                        add_turn_type = self.get_turn_type(add_conn[0], add_conn[1])
                        ref_found = False
                        for orig_conn, orig_link in orig_conn_to_link.items():
                            if self.get_turn_type(orig_conn[0], orig_conn[1]) == add_turn_type and orig_conn[0][:-2] == add_conn[0][:-2]:
                                char_to_set = modified_orig_state[orig_link]
                                new_link = new_conn_to_link[add_conn]
                                final_state[new_link] = char_to_set
                                ref_found = True
                                break
                       
                    # 创建新的phase元素
                    new_phase_elem = ET.Element('phase', attrib=orig_phase.attrib)
                    new_phase_elem.set('state', "".join(final_state))
                    new_phases_elements.append(new_phase_elem)
                
                # 替换旧的相位列表
                for p in new_tl_logic.findall('phase'):
                    new_tl_logic.remove(p)
                new_tl_logic.extend(new_phases_elements)

        # 5. Reorder tlLogic elements to be before junction elements for SUMO compatibility/readability

        # Extract all tlLogic elements
        all_tllogics = new_root.findall('tlLogic')
        for tl in all_tllogics:
            new_root.remove(tl)

        # Find the first junction element to determine the insertion point
        first_junction = new_root.find('junction')
        if first_junction is not None:
            junction_index = list(new_root).index(first_junction)
            # Insert tlLogic elements before the first junction
            for i, tl_logic in enumerate(all_tllogics):
                new_root.insert(junction_index + i, tl_logic)
        else:
            # This is a fallback case, as a network with traffic lights should have junctions
            for tl_logic in all_tllogics:
                new_root.append(tl_logic)
                
        # 6. 保存最终修改的路网文件
        new_tree.write(output_net_file_filt, encoding="utf-8", xml_declaration=True)


    # def prepare_for_analysis(self, net_file_path: str) -> str:
    #     """
    #     准备分析所需的文件和数据。
    #     执行拆分和解析，并返回用于清理的临时目录路径。
    #     """
    #     net_file_path = os.path.abspath(net_file_path)
    #     base_prefix = os.path.splitext(os.path.basename(net_file_path))[0]
    #     base_prefix=os.path.basename(net_file_path).split('.')[0]
    #     # Create a unique temp directory for this analysis
    #     temp_dir = os.path.join(os.path.dirname(net_file_path), f"temp_analysis_{uuid.uuid4().hex[:8]}")
    #     os.makedirs(temp_dir, exist_ok=True)
        
    #     split_prefix = os.path.join(temp_dir, base_prefix)
        
    #     self.logger.info(f"Preparing analysis in temporary directory: {temp_dir}")
    #     self.split_network_file(net_file_path, split_prefix)
    #     self.parse_network_files(split_prefix)
        
    #     return temp_dir
        
    # def get_traffic_light_ids(self) -> List[str]:
    #     """返回已解析路网中所有交通信号灯的ID列表"""
    #     return list(self.traffic_lights.keys())

    def get_proposals_for_intersection(self, intersection_id: str, decisions_data: Dict) -> List[Dict]:
        """
        为单个交叉口生成所有潜在的优化方案。
        必须先调用 prepare_for_analysis。
        """
        intersection_data = decisions_data.get(intersection_id)
        if not intersection_data:
            return []
            
        preferred_phase = intersection_data.get('preferred_phase', '')
        connection_pressures = intersection_data.get('connection_pressures', {})
        
        if not preferred_phase or not connection_pressures:
            return []
            
        # process_intersection 已经实现了寻找和排序逻辑
        proposals = self.process_intersection(
            intersection_id, preferred_phase, connection_pressures
        )
        return proposals

    def process_single_proposal(self, net_file_path: str, proposal_to_apply: Dict) -> Tuple[str, str]:
        """
        仅针对一个给定的优化方案执行完整的文件生成流程。
        返回 (最终生成的路网文件路径, 临时工作目录路径)。
        """
        # 1. 准备工作目录
        base_prefix_orig = os.path.basename(net_file_path).split('.')[0]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        proposal_id = f"{proposal_to_apply['intersection_id']}_{proposal_to_apply.get('from_lane', 'N/A')}"
        iter_dir = os.path.join(os.path.dirname(net_file_path), "greedy_optimized", f"{base_prefix_orig}_{timestamp}_{proposal_id}_{uuid.uuid4().hex[:8]}")
        os.makedirs(iter_dir, exist_ok=True)
        
        split_prefix = os.path.join(iter_dir, base_prefix_orig)
        
        try:
            # 2. 拆分
            self.split_network_file(net_file_path, split_prefix)
            self.match_tl_to_edge(net_file_path)
            processed_from_lanes = set()

            # 3. 应用单个方案
            all_new_connections = defaultdict(list)
            
            new_connections = self.generate_fan_out_connections(proposal_to_apply['connection'], proposal_to_apply['from_lane'])
            from_edge = proposal_to_apply['connection'][0]
            best_from_lane = proposal_to_apply['from_lane']
            if new_connections:
                intersection_id = proposal_to_apply['intersection_id']
                all_new_connections[intersection_id].extend(new_connections)
                processed_from_lanes.add((from_edge, best_from_lane))

            # 4. 更新与合并
            if all_new_connections:
                connection_file = f"{split_prefix}.con.xml"
                # 需要重新解析原始连接文件来更新
                self._parse_connections(f"{split_prefix}.con.xml")
                self.update_connection_file(connection_file, all_new_connections)
            
            output_net_file = os.path.join(iter_dir, "modified.net.xml")
            self.merge_network_files(split_prefix, output_net_file)
            output_net_file_filt = os.path.join(iter_dir, "modified_filt.net.xml")
            self.modify_tl_logic(net_file_path, output_net_file, output_net_file_filt, processed_from_lanes)
            
            return output_net_file_filt, iter_dir

        except Exception as e:
            self.logger.error(f"处理单个方案时出错: {e}", exc_info=True)
            if os.path.exists(iter_dir):
                shutil.rmtree(iter_dir)
            return None, None
            

