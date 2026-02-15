"""
OD矩阵调整器
基于现有的Carttils模块功能，提供OD矩阵微调功能
"""
import os
import sys
import numpy as np
import pickle
from typing import Dict, Any, Tuple, List, Optional
import copy
from collections import defaultdict

# 添加父目录到路径以导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool.Carttils import *


class ODAdjuster:
    """OD矩阵调整器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化调整器
        
        Args:
            config: 调整器配置参数
        """
        self.config = config
        self.od_config = config.get('od', {})
        self.max_adjustment_ratio = self.od_config.get('max_adjustment_ratio', 0.2)
        self.min_od_value = self.od_config.get('min_od_value', 1)
        self.adjustment_step = self.od_config.get('adjustment_step', 0.05)
        self.temp_dir = config.get('temp_dir', 'temp/')
        
    def propose(self, 
                prev_params: Dict[str, Any], 
                diff_dict: Dict[str, Any],
                od_file: str) -> Tuple[Dict[str, Any], str]:
        """
        基于误差分析提出新的OD调整参数
        
        Args:
            prev_params: 之前的参数
            diff_dict: 误差分析结果
            od_file: 当前OD文件路径
            
        Returns:
            (新参数, 调整描述)
        """
        # 分析哪些区域需要调整
        adjustment_plan = self._analyze_od_adjustments(diff_dict)
        
        # 生成调整参数
        new_params = {
            'adjustment_type': 'od',
            'adjustments': adjustment_plan['adjustments'],
            'od_file': od_file,
            'adjustment_summary': adjustment_plan['summary']
        }
        
        description = f"OD调整: {adjustment_plan['summary']}"
        
        return new_params, description
    
    def apply_adjustment(self, 
                        od_file: str, 
                        new_params: Dict[str, Any],
                        output_file: Optional[str] = None) -> str:
        """
        应用OD调整
        
        Args:
            od_file: 原始OD文件
            new_params: 新的OD调整参数
            output_file: 输出文件路径（可选）
            
        Returns:
            调整后的OD文件路径
        """
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(od_file))[0]
            output_file = os.path.join(self.temp_dir, f"{base_name}_od_adjusted.pkl")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 加载原始OD数据
        od_matrix = self._load_od_matrix(od_file)
        
        # 应用调整
        adjusted_od = self._apply_od_adjustments(od_matrix, new_params['adjustments'])
        
        # 保存调整后的OD矩阵
        self._save_od_matrix(adjusted_od, output_file)
        
        return output_file
    
    def generate_route_from_od(self, 
                              od_file: str, 
                              net_file: str,
                              output_rou_file: str,
                              scale: float = 1.0) -> str:
        """
        从OD矩阵生成路由文件
        
        Args:
            od_file: OD矩阵文件
            net_file: 网络文件
            output_rou_file: 输出路由文件
            scale: 流量规模
            
        Returns:
            生成的路由文件路径
        """
        try:
            # 使用现有的OD2rou功能
            od_matrix = self._load_od_matrix(od_file)
            
            # 调用现有的路由生成函数（需要根据实际的Carttils函数调整）
            # 这里假设存在类似的函数，需要根据实际情况修改
            generate_routes_from_od(od_matrix, net_file, output_rou_file, scale)
            
            return output_rou_file
            
        except Exception as e:
            print(f"从OD生成路由时出错: {e}")
            raise
    
    def _analyze_od_adjustments(self, diff_dict: Dict[str, Any]) -> Dict[str, Any]:
        """分析需要进行的OD调整"""
        edge_level = diff_dict.get('edge_level', {})
        
        # 识别需要增加流量的区域（速度过高，密度过低）
        increase_zones = []
        decrease_zones = []
        
        for edge_id, diff in edge_level.items():
            sim_speed = diff['sim_values'].get('avg_speed', 0)
            real_speed = diff['real_values'].get('speed', 0)
            
            sim_density = diff['sim_values'].get('avg_density', 0)
            real_density = diff['real_values'].get('density', 0)
            
            if real_speed > 0 and real_density > 0:
                speed_ratio = sim_speed / real_speed
                density_ratio = sim_density / real_density
                
                # 仿真速度明显高于真实，密度明显低于真实 -> 需要增加流量
                if speed_ratio > 1.2 and density_ratio < 0.8:
                    increase_zones.append({
                        'edge_id': edge_id,
                        'priority': (speed_ratio - 1) + (1 - density_ratio),
                        'adjustment_factor': min(1.2, 1 + self.adjustment_step)
                    })
                
                # 仿真速度明显低于真实，密度明显高于真实 -> 需要减少流量
                elif speed_ratio < 0.8 and density_ratio > 1.2:
                    decrease_zones.append({
                        'edge_id': edge_id,
                        'priority': (1 - speed_ratio) + (density_ratio - 1),
                        'adjustment_factor': max(0.8, 1 - self.adjustment_step)
                    })
        
        # 按优先级排序
        increase_zones.sort(key=lambda x: x['priority'], reverse=True)
        decrease_zones.sort(key=lambda x: x['priority'], reverse=True)
        
        # 生成调整计划
        adjustments = {
            'increase': increase_zones[:10],  # 最多调整前10个
            'decrease': decrease_zones[:10]
        }
        
        summary = f"增加流量区域{len(adjustments['increase'])}个, 减少流量区域{len(adjustments['decrease'])}个"
        
        return {
            'adjustments': adjustments,
            'summary': summary
        }
    
    def _load_od_matrix(self, od_file: str) -> Any:
        """加载OD矩阵"""
        try:
            with open(od_file, 'rb') as f:
                od_data = pickle.load(f)
            return od_data
        except Exception as e:
            print(f"加载OD矩阵出错: {e}")
            return {}
    
    def _save_od_matrix(self, od_matrix: Any, output_file: str):
        """保存OD矩阵"""
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(od_matrix, f)
        except Exception as e:
            print(f"保存OD矩阵出错: {e}")
            raise
    
    def _apply_od_adjustments(self, od_matrix: Any, adjustments: Dict[str, Any]) -> Any:
        """应用OD调整"""
        adjusted_od = copy.deepcopy(od_matrix)
        
        try:
            # 根据OD矩阵的实际格式进行调整
            # 这里需要根据实际的OD数据结构进行修改
            
            if isinstance(od_matrix, dict):
                # 如果是字典格式的OD矩阵
                self._adjust_dict_od(adjusted_od, adjustments)
            elif isinstance(od_matrix, np.ndarray):
                # 如果是numpy数组格式的OD矩阵
                self._adjust_array_od(adjusted_od, adjustments)
            else:
                print(f"不支持的OD矩阵格式: {type(od_matrix)}")
                
        except Exception as e:
            print(f"应用OD调整时出错: {e}")
            return od_matrix  # 返回原始数据
        
        return adjusted_od
    
    def _adjust_dict_od(self, od_dict: Dict, adjustments: Dict[str, Any]):
        """调整字典格式的OD矩阵"""
        increase_zones = adjustments.get('increase', [])
        decrease_zones = adjustments.get('decrease', [])
        
        # 为增加流量的边找到相关的OD对并增加
        for zone in increase_zones:
            edge_id = zone['edge_id']
            factor = zone['adjustment_factor']
            
            # 查找包含此边的OD对
            for od_key, od_value in od_dict.items():
                if isinstance(od_key, tuple) and len(od_key) == 2:
                    origin, destination = od_key
                    # 这里需要根据实际的边-区域映射关系判断
                    # 简化处理：如果edge_id在origin或destination区域内
                    if edge_id in str(origin) or edge_id in str(destination):
                        new_value = max(self.min_od_value, od_value * factor)
                        od_dict[od_key] = new_value
        
        # 为减少流量的边找到相关的OD对并减少
        for zone in decrease_zones:
            edge_id = zone['edge_id']
            factor = zone['adjustment_factor']
            
            # 查找包含此边的OD对
            for od_key, od_value in od_dict.items():
                if isinstance(od_key, tuple) and len(od_key) == 2:
                    origin, destination = od_key
                    if edge_id in str(origin) or edge_id in str(destination):
                        new_value = max(self.min_od_value, od_value * factor)
                        od_dict[od_key] = new_value
    
    def _adjust_array_od(self, od_array: np.ndarray, adjustments: Dict[str, Any]):
        """调整数组格式的OD矩阵"""
        # 这里需要根据实际的区域-数组索引映射关系进行调整
        # 简化处理：随机选择一些OD对进行调整
        
        increase_zones = adjustments.get('increase', [])
        decrease_zones = adjustments.get('decrease', [])
        
        rows, cols = od_array.shape
        
        # 为增加流量的区域增加OD值
        for zone in increase_zones[:min(5, len(increase_zones))]:
            factor = zone['adjustment_factor']
            # 随机选择一些OD对进行调整
            for _ in range(min(3, rows)):
                i = np.random.randint(0, rows)
                j = np.random.randint(0, cols)
                if od_array[i, j] > 0:
                    od_array[i, j] = max(self.min_od_value, od_array[i, j] * factor)
        
        # 为减少流量的区域减少OD值
        for zone in decrease_zones[:min(5, len(decrease_zones))]:
            factor = zone['adjustment_factor']
            for _ in range(min(3, rows)):
                i = np.random.randint(0, rows)
                j = np.random.randint(0, cols)
                if od_array[i, j] > 0:
                    od_array[i, j] = max(self.min_od_value, od_array[i, j] * factor)
    
    def analyze_od_coverage(self, od_file: str, edge_list: List[str]) -> Dict[str, Any]:
        """分析OD矩阵的覆盖情况"""
        od_matrix = self._load_od_matrix(od_file)
        
        analysis = {
            'total_od_pairs': 0,
            'non_zero_pairs': 0,
            'coverage_ratio': 0,
            'edge_coverage': {}
        }
        
        if isinstance(od_matrix, dict):
            analysis['total_od_pairs'] = len(od_matrix)
            analysis['non_zero_pairs'] = sum(1 for v in od_matrix.values() if v > 0)
            
        elif isinstance(od_matrix, np.ndarray):
            analysis['total_od_pairs'] = od_matrix.size
            analysis['non_zero_pairs'] = np.count_nonzero(od_matrix)
            
        if analysis['total_od_pairs'] > 0:
            analysis['coverage_ratio'] = analysis['non_zero_pairs'] / analysis['total_od_pairs']
        
        return analysis 