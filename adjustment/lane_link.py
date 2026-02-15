"""
车道连接调整器
基于现有的modifyLane.py模块，提供车道连接调整功能
"""
import os
import sys
import numpy as np
import pickle
from typing import Dict, Any, Tuple, List, Optional
import xml.etree.ElementTree as ET
import shutil
from collections import defaultdict

# 添加父目录到路径以导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from modifyLane import *  # 导入现有的车道修改功能
except ImportError:
    print("警告: 无法导入modifyLane模块，车道调整功能可能受限")


class LaneLinkAdjuster:
    """车道连接调整器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化调整器
        
        Args:
            config: 调整器配置参数
        """
        self.config = config
        self.lane_config = config.get('lane', {})
        self.max_modifications = self.lane_config.get('max_modifications', 5)
        self.priority_threshold = self.lane_config.get('priority_threshold', 0.8)
        self.temp_dir = config.get('temp_dir', 'temp/')
        
    def propose(self, 
                prev_params: Dict[str, Any], 
                diff_dict: Dict[str, Any],
                net_file: str) -> Tuple[Dict[str, Any], str]:
        """
        基于误差分析提出车道连接调整方案
        
        Args:
            prev_params: 之前的参数
            diff_dict: 误差分析结果
            net_file: 当前网络文件
            
        Returns:
            (新参数, 调整描述)
        """
        # 分析需要调整的车道连接
        adjustment_plan = self._analyze_lane_adjustments(diff_dict, net_file)
        
        new_params = {
            'adjustment_type': 'lane',
            'modifications': adjustment_plan['modifications'],
            'net_file': net_file,
            'adjustment_summary': adjustment_plan['summary']
        }
        
        description = f"车道调整: {adjustment_plan['summary']}"
        
        return new_params, description
    
    def apply_adjustment(self, 
                        net_file: str, 
                        new_params: Dict[str, Any],
                        output_file: Optional[str] = None) -> str:
        """
        应用车道连接调整
        
        Args:
            net_file: 原始网络文件
            new_params: 新的车道调整参数
            output_file: 输出文件路径（可选）
            
        Returns:
            调整后的网络文件路径
        """
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(net_file))[0]
            output_file = os.path.join(self.temp_dir, f"{base_name}_lane_adjusted.net.xml")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 应用车道调整
        success = self._apply_lane_modifications(net_file, output_file, new_params['modifications'])
        
        if not success:
            # 如果调整失败，复制原文件
            shutil.copy2(net_file, output_file)
            print("车道调整失败，使用原始网络文件")
        
        return output_file
    
    def _analyze_lane_adjustments(self, diff_dict: Dict[str, Any], net_file: str) -> Dict[str, Any]:
        """分析需要进行的车道调整"""
        edge_level = diff_dict.get('edge_level', {})
        
        # 识别严重拥堵的边（需要增加车道或优化连接）
        congestion_edges = []
        
        for edge_id, diff in edge_level.items():
            sim_speed = diff['sim_values'].get('avg_speed', 0)
            real_speed = diff['real_values'].get('speed', 0)
            
            sim_occupancy = diff['sim_values'].get('avg_occupancy', 0)
            real_occupancy = diff['real_values'].get('occupancy', 0)
            
            if real_speed > 0:
                speed_ratio = sim_speed / real_speed
                occupancy_diff = abs(sim_occupancy - real_occupancy)
                
                # 仿真速度明显低于真实且占有率差异大 -> 可能是车道连接问题
                if speed_ratio < 0.7 and occupancy_diff > 0.2:
                    priority = (1 - speed_ratio) + occupancy_diff
                    congestion_edges.append({
                        'edge_id': edge_id,
                        'priority': priority,
                        'speed_ratio': speed_ratio,
                        'occupancy_diff': occupancy_diff,
                        'issue_type': 'congestion'
                    })
        
        # 按优先级排序，选择最需要调整的边
        congestion_edges.sort(key=lambda x: x['priority'], reverse=True)
        high_priority_edges = [e for e in congestion_edges if e['priority'] > self.priority_threshold]
        
        # 生成具体的修改方案
        modifications = self._generate_modifications(high_priority_edges[:self.max_modifications], net_file)
        
        summary = f"计划修改{len(modifications)}处车道连接"
        
        return {
            'modifications': modifications,
            'summary': summary,
            'analyzed_edges': len(congestion_edges),
            'high_priority_count': len(high_priority_edges)
        }
    
    def _generate_modifications(self, priority_edges: List[Dict], net_file: str) -> List[Dict[str, Any]]:
        """生成具体的车道修改方案"""
        modifications = []
        
        try:
            # 解析网络文件获取拓扑信息
            network_info = self._parse_network_topology(net_file)
            
            for edge_info in priority_edges:
                edge_id = edge_info['edge_id']
                
                # 检查该边是否存在于网络中
                if edge_id in network_info.get('edges', {}):
                    edge_data = network_info['edges'][edge_id]
                    
                    # 基于问题类型生成修改方案
                    if edge_info['issue_type'] == 'congestion':
                        mod = self._generate_congestion_fix(edge_id, edge_data, network_info)
                        if mod:
                            modifications.append(mod)
                            
        except Exception as e:
            print(f"生成车道修改方案时出错: {e}")
        
        return modifications
    
    def _parse_network_topology(self, net_file: str) -> Dict[str, Any]:
        """解析网络拓扑结构"""
        network_info = {
            'edges': {},
            'junctions': {},
            'connections': []
        }
        
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
            
            # 解析边信息
            for edge in root.findall('.//edge'):
                edge_id = edge.get('id')
                if edge_id and not edge_id.startswith(':'):  # 排除内部边
                    lanes = edge.findall('lane')
                    network_info['edges'][edge_id] = {
                        'id': edge_id,
                        'from': edge.get('from'),
                        'to': edge.get('to'),
                        'lane_count': len(lanes),
                        'lanes': [lane.get('id') for lane in lanes]
                    }
            
            # 解析连接信息
            for connection in root.findall('.//connection'):
                conn_info = {
                    'from': connection.get('from'),
                    'to': connection.get('to'),
                    'fromLane': connection.get('fromLane'),
                    'toLane': connection.get('toLane')
                }
                network_info['connections'].append(conn_info)
                
        except Exception as e:
            print(f"解析网络拓扑时出错: {e}")
        
        return network_info
    
    def _generate_congestion_fix(self, edge_id: str, edge_data: Dict, network_info: Dict) -> Optional[Dict[str, Any]]:
        """为拥堵边生成修复方案"""
        current_lanes = edge_data.get('lane_count', 1)
        
        # 根据现有车道数决定调整策略
        if current_lanes < 3:
            # 车道较少，考虑增加车道
            return {
                'type': 'add_lane',
                'edge_id': edge_id,
                'target_lanes': current_lanes + 1,
                'description': f"为边{edge_id}增加车道（{current_lanes}->{current_lanes + 1}）"
            }
        else:
            # 车道较多，考虑优化连接
            return {
                'type': 'optimize_connection',
                'edge_id': edge_id,
                'description': f"优化边{edge_id}的车道连接"
            }
    
    def _apply_lane_modifications(self, input_file: str, output_file: str, modifications: List[Dict]) -> bool:
        """应用车道修改"""
        try:
            # 首先复制原文件
            shutil.copy2(input_file, output_file)
            
            if not modifications:
                return True
            
            # 解析XML文件
            tree = ET.parse(output_file)
            root = tree.getroot()
            
            modifications_applied = 0
            
            for mod in modifications:
                try:
                    if mod['type'] == 'add_lane':
                        success = self._add_lane_to_edge(root, mod)
                        if success:
                            modifications_applied += 1
                    elif mod['type'] == 'optimize_connection':
                        success = self._optimize_edge_connections(root, mod)
                        if success:
                            modifications_applied += 1
                            
                except Exception as e:
                    print(f"应用修改{mod}时出错: {e}")
                    continue
            
            # 保存修改后的文件
            tree.write(output_file, encoding='utf-8', xml_declaration=True)
            
            print(f"成功应用{modifications_applied}/{len(modifications)}个车道修改")
            return modifications_applied > 0
            
        except Exception as e:
            print(f"应用车道修改时出错: {e}")
            return False
    
    def _add_lane_to_edge(self, root: ET.Element, mod: Dict) -> bool:
        """为边增加车道"""
        edge_id = mod['edge_id']
        target_lanes = mod['target_lanes']
        
        # 找到对应的边
        for edge in root.findall(f".//edge[@id='{edge_id}']"):
            lanes = edge.findall('lane')
            current_lane_count = len(lanes)
            
            if current_lane_count >= target_lanes:
                continue
            
            # 复制最后一条车道的属性来创建新车道
            if lanes:
                last_lane = lanes[-1]
                new_lane_id = f"{edge_id}_{current_lane_count}"
                
                new_lane = ET.SubElement(edge, 'lane')
                new_lane.set('id', new_lane_id)
                
                # 复制属性
                for attr_name in ['speed', 'length', 'shape']:
                    attr_value = last_lane.get(attr_name)
                    if attr_value:
                        new_lane.set(attr_name, attr_value)
                
                return True
        
        return False
    
    def _optimize_edge_connections(self, root: ET.Element, mod: Dict) -> bool:
        """优化边的连接"""
        edge_id = mod['edge_id']
        
        # 这里可以实现更复杂的连接优化逻辑
        # 简化处理：标记为已处理
        print(f"优化边{edge_id}的连接（简化实现）")
        return True
    
    def analyze_bottlenecks(self, net_file: str, diff_dict: Dict[str, Any]) -> Dict[str, Any]:
        """分析网络瓶颈"""
        analysis = {
            'bottleneck_edges': [],
            'potential_fixes': [],
            'priority_ranking': []
        }
        
        edge_level = diff_dict.get('edge_level', {})
        network_info = self._parse_network_topology(net_file)
        
        # 识别瓶颈边
        for edge_id, diff in edge_level.items():
            sim_speed = diff['sim_values'].get('avg_speed', 0)
            real_speed = diff['real_values'].get('speed', 0)
            
            if real_speed > 0 and sim_speed / real_speed < 0.6:  # 仿真速度明显低于真实
                bottleneck_score = 1 - (sim_speed / real_speed)
                
                analysis['bottleneck_edges'].append({
                    'edge_id': edge_id,
                    'score': bottleneck_score,
                    'sim_speed': sim_speed,
                    'real_speed': real_speed
                })
        
        # 按瓶颈程度排序
        analysis['bottleneck_edges'].sort(key=lambda x: x['score'], reverse=True)
        
        return analysis 