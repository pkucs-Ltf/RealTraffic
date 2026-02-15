"""
仿真结果与真实数据比较器
基于现有的Carttils模块功能，提供标准化的评估接口
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict
import math

# 添加父目录到路径以导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool.Carttils import *


class Comparator:
    """仿真结果与真实数据比较器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化比较器
        
        Args:
            config: 比较器配置参数
        """
        self.config = config
        self.weights = config.get('metrics_weights', {
            'speed': 0.6,
            'waiting': 0.4
        })
        
            
    def calculate_congest_R(self,result_data,average_waiting_vehicles,road_limit_file):
            # 初始化一个新字典来保存拥堵评分
            congestion_scores = {}
             
            
            # 遍历result_data中的每个路段ID
            for road_id in result_data:
                if road_id=='-1156810185#1' or road_id=='-1135361330#0':
                    a=1
                
   
           
                # 检查该路段ID是否也存在于road_limit_file中
                if road_id in road_limit_file:
                    # 计算自由流速度与实际速度的比值
                    a=result_data[road_id]
                    b=road_limit_file[road_id]
                    ratio = road_limit_file[road_id] / result_data[road_id]
                    # if road_id=='-146747108#0':
                    #     print(ratio)
                    # # print(ratio)
            
            
                    
                    # 根据比值分配拥堵评分
                    if ratio > 2 or average_waiting_vehicles[road_id] > 20:
                        congestion_scores[road_id] = 5  # 严重拥堵
                    elif 1.5 < ratio <= 2 or average_waiting_vehicles[road_id] > 20:
                        congestion_scores[road_id] = 3  # 中度拥堵
                    else:  # ratio <= 1.5
                        congestion_scores[road_id] = 0  # 畅通
            
            return congestion_scores

    def evalute_mae(self,real_traffic, sumo_result):
        
        # 初始化变量以存储匹配的值
        real_values = []
        sumo_values = []
        get_not_fit=[]
        
        # 遍历real_traffic字典中的每个路段ID
        for road_id in real_traffic.keys():
            if road_id not in sumo_result:
                continue
            if real_traffic[road_id] !=sumo_result[road_id]:
                get_not_fit.append([road_id,real_traffic[road_id],sumo_result[road_id]])
            # 检查该路段ID是否也存在于sumo_result中
            if road_id in sumo_result:
                real_values.append(real_traffic[road_id])
                sumo_values.append(sumo_result[road_id])

        # 将列表转换为numpy数组以便计算
        real_values = np.array(real_values)
        sumo_values = np.array(sumo_values)
        # 找出real_values中非零的索引
        non_zero_indices = np.where(real_values != 0)[0]

        # 使用非零值计算召回率
        if len(non_zero_indices) > 0:
            # 获取对应的sumo_values
            relevant_sumo_values = sumo_values[non_zero_indices]
            relevant_real_values = real_values[non_zero_indices]
            
            # 计算召回率 (真正例 / (真正例 + 假负例))
            true_positives = np.sum((relevant_sumo_values != 0) & (relevant_real_values != 0))
            false_negatives = np.sum((relevant_sumo_values == 0) & (relevant_real_values != 0))
            
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            print(f"召回率: {recall:.4f}")
        else:
            recall=0
            print("没有非零的真实值，无法计算召回率")
        

        # 计算准确率
        # 计算准确率 (真正例 / (真正例 + 假正例))
        true_positives = np.sum((sumo_values != 0) & (real_values != 0))
        false_positives = np.sum((sumo_values != 0) & (real_values == 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        print(f"准确率: {precision:.4f}")
            
        # 计算F1分数
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"F1分数: {f1_score:.4f}")
    
        
        # 计算平均绝对误差(MAE)
        mae = np.mean(np.abs(real_values - sumo_values))
        # with open('get_not_fit.pkl', 'wb') as f:
        #     pickle.dump(get_not_fit, f)

        return mae,precision,recall,f1_score
        
    def compare(self,real_data,real_data_type,sumo_average_speeds,average_waiting_vehicles,road_limit_file):
        """
        比较仿真结果与真实数据
        
        Args:
            sim_metrics_file: 仿真指标文件路径
            real_data_file: 真实数据文件路径
            real_data_type: 真实数据类型 ('shp' 或 'pkl')
            
        Returns:
            (差异详情字典, 综合评分)
        """
        # 加载仿真指标
        
        # 加载真实数据
   
        real_data = loadfile(real_data)
        limit_data =loadfile(road_limit_file)
        sumo_data = self.calculate_congest_R(sumo_average_speeds,average_waiting_vehicles,limit_data)

        
        mae,precision,recall,f1_score = self.evalute_mae(real_data,sumo_data)
        print(f"f1_score: {f1_score:.4f}")
        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")
        print(f"mae: {mae:.4f}")
        
        return f1_score
    
    def _load_sim_metrics(self, metrics_file: str) -> Dict[str, Any]:
        """加载仿真指标数据"""
        with open(metrics_file, 'rb') as f:
            return pickle.load(f)
    
    def _load_real_data(self, data_file: str, data_type: str) -> Dict[str, Any]:
        """加载真实数据"""
        if data_type == 'shp':
            return self._load_shapefile_data(data_file)
        elif data_type == 'pkl':
            return self._load_pickle_data(data_file)
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
    
    def _load_shapefile_data(self, shp_file: str) -> Dict[str, Any]:
        """加载shapefile格式的真实数据"""
        try:
            gdf = gpd.read_file(shp_file)
            
            # 提取edge_id和交通指标
            real_data = {}
            for _, row in gdf.iterrows():
                edge_id = str(row.get('edge_id', row.get('EDGE_ID', row.get('id', ''))))
                if edge_id:
                    real_data[edge_id] = {
                        'speed': float(row.get('speed', row.get('SPEED', 0))),
                        'density': float(row.get('density', row.get('DENSITY', 0))),
                        'occupancy': float(row.get('occupancy', row.get('OCCUPANCY', 0))),
                        'flow': float(row.get('flow', row.get('FLOW', 0)))
                    }
            
            return {'edge_data': real_data}
            
        except Exception as e:
            print(f"加载shapefile数据出错: {e}")
            return {'edge_data': {}}
    
    def _load_pickle_data(self, pkl_file: str) -> Dict[str, Any]:
        """加载pickle格式的真实数据"""
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # 标准化数据格式
            if isinstance(data, dict):
                if 'edge_data' in data:
                    return data
                else:
                    # 假设直接是edge_id -> metrics的映射
                    return {'edge_data': data}
            else:
                print(f"不支持的pickle数据格式: {type(data)}")
                return {'edge_data': {}}
                
        except Exception as e:
            print(f"加载pickle数据出错: {e}")
            return {'edge_data': {}}
    
    def _align_data(self, sim_metrics: Dict[str, Any], real_data: Dict[str, Any]) -> Dict[str, Any]:
        """对齐仿真和真实数据"""
        # 适应新的仿真结果格式
        sim_speeds = sim_metrics.get('average_speeds', {})
        sim_waiting = sim_metrics.get('average_waiting_vehicles', {})
        real_edges = real_data.get('edge_data', {})
        
        # 构造仿真数据的统一格式
        sim_edges = {}
        for edge_id in sim_speeds.keys():
            sim_edges[edge_id] = {
                'avg_speed': sim_speeds.get(edge_id, 0),
                'avg_waiting': sim_waiting.get(edge_id, 0),
                'avg_density': 0,  # 如果有额外的密度数据可以在这里添加
                'avg_occupancy': 0  # 如果有额外的占有率数据可以在这里添加
            }
        
        # 找到共同的edge_id
        common_edges = set(sim_edges.keys()) & set(real_edges.keys())
        
        aligned = {
            'common_edges': list(common_edges),
            'sim_data': {},
            'real_data': {},
            'missing_in_sim': list(set(real_edges.keys()) - set(sim_edges.keys())),
            'missing_in_real': list(set(sim_edges.keys()) - set(real_edges.keys()))
        }
        
        # 提取共同边的数据
        for edge_id in common_edges:
            aligned['sim_data'][edge_id] = sim_edges[edge_id]
            aligned['real_data'][edge_id] = real_edges[edge_id]
        
        return aligned
    
    def _calculate_differences(self, aligned_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算各项指标的差异"""
        sim_data = aligned_data['sim_data']
        real_data = aligned_data['real_data']
        common_edges = aligned_data['common_edges']
        
        differences = {
            'edge_level': {},
            'summary': {
                'speed': {'mae': 0, 'rmse': 0, 'mape': 0},
                'waiting': {'mae': 0, 'rmse': 0, 'mape': 0}
            },
            'statistics': {
                'total_edges': len(common_edges),
                'coverage': len(common_edges) / (len(common_edges) + len(aligned_data['missing_in_sim']) + len(aligned_data['missing_in_real'])) if common_edges else 0
            }
        }
        
        if not common_edges:
            return differences
        
        # 计算边级差异
        speed_errors = []
        waiting_errors = []
        
        for edge_id in common_edges:
            sim = sim_data[edge_id]
            real = real_data[edge_id]
            
            # 速度差异
            speed_diff = abs(sim.get('avg_speed', 0) - real.get('speed', 0))
            speed_rel_diff = speed_diff / max(real.get('speed', 1), 0.1)  # 避免除零
            
            # 等待车辆差异（如果真实数据中有相关指标）
            real_waiting = real.get('waiting', real.get('halting', real.get('density', 0)))
            waiting_diff = abs(sim.get('avg_waiting', 0) - real_waiting)
            waiting_rel_diff = waiting_diff / max(real_waiting, 0.1)
            
            differences['edge_level'][edge_id] = {
                'speed_diff': speed_diff,
                'speed_rel_diff': speed_rel_diff,
                'waiting_diff': waiting_diff,
                'waiting_rel_diff': waiting_rel_diff,
                'sim_values': sim,
                'real_values': real
            }
            
            speed_errors.append(speed_diff)
            waiting_errors.append(waiting_diff)
        
        # 计算汇总统计
        if speed_errors:
            differences['summary']['speed'] = {
                'mae': np.mean(speed_errors),
                'rmse': np.sqrt(np.mean(np.square(speed_errors))),
                'max_error': np.max(speed_errors),
                'min_error': np.min(speed_errors)
            }
        
        if waiting_errors:
            differences['summary']['waiting'] = {
                'mae': np.mean(waiting_errors),
                'rmse': np.sqrt(np.mean(np.square(waiting_errors))),
                'max_error': np.max(waiting_errors),
                'min_error': np.min(waiting_errors)
            }
        
        return differences
    
    def _calculate_overall_score(self, diff_details: Dict[str, Any]) -> float:
        """计算综合评分（越低越好）"""
        summary = diff_details['summary']
        
        # 使用加权MAE作为主要评分指标
        score = 0
        total_weight = 0
        
        for metric, weight in self.weights.items():
            if metric in summary and 'mae' in summary[metric]:
                score += weight * summary[metric]['mae']
                total_weight += weight
        
        if total_weight > 0:
            score = score / total_weight
        
        return score
    
    def analyze_error_patterns(self, diff_details: Dict[str, Any]) -> Dict[str, Any]:
        """分析误差模式，为调整策略提供建议"""
        edge_diffs = diff_details['edge_level']
        summary = diff_details['summary']
        
        analysis = {
            'dominant_errors': {},
            'problematic_edges': [],
            'adjustment_suggestions': []
        }
        
        # 识别主要误差类型
        speed_mae = summary['speed'].get('mae', 0)
        waiting_mae = summary['waiting'].get('mae', 0)
        
        dominant_error = max([
            ('speed', speed_mae),
            ('waiting', waiting_mae)
        ], key=lambda x: x[1])
        
        analysis['dominant_errors'] = {
            'type': dominant_error[0],
            'value': dominant_error[1],
            'all_errors': {
                'speed': speed_mae,
                'waiting': waiting_mae
            }
        }
        
        # 识别问题边（误差最大的前20%）
        if edge_diffs:
            edge_scores = []
            for edge_id, diff in edge_diffs.items():
                # 计算边的综合误差分数
                edge_score = (
                    self.weights['speed'] * diff['speed_rel_diff'] +
                    self.weights['waiting'] * diff['waiting_rel_diff']
                )
                edge_scores.append((edge_id, edge_score, diff))
            
            # 排序并取前20%
            edge_scores.sort(key=lambda x: x[1], reverse=True)
            top_20_percent = max(1, len(edge_scores) // 5)
            analysis['problematic_edges'] = edge_scores[:top_20_percent]
        
        # 生成调整建议
        analysis['adjustment_suggestions'] = self._generate_adjustment_suggestions(analysis)
        
        return analysis
    
    def _generate_adjustment_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """基于误差模式生成调整建议"""
        suggestions = []
        
        dominant_error = analysis['dominant_errors']
        all_errors = dominant_error['all_errors']
        
        # 基于主要误差类型给出建议
        if dominant_error['type'] == 'speed':
            if all_errors['speed'] > 10:  # 速度差异较大
                suggestions.append("建议调整整体scale：速度差异较大，可能是流量规模问题")
            else:
                suggestions.append("建议微调OD矩阵：速度差异适中，可通过OD调整改善")
        
        elif dominant_error['type'] == 'waiting':
            suggestions.append("建议调整OD矩阵：等待车辆数差异表明流量分布可能不合理")
            if len(analysis['problematic_edges']) > 0:
                suggestions.append("考虑车道连接调整：存在局部拥堵问题")
        

        
        # 基于问题边数量给出建议
        problematic_ratio = len(analysis['problematic_edges']) / max(1, len(analysis.get('edge_level', {})))
        if problematic_ratio > 0.3:
            suggestions.append("问题边较多，建议先进行scale调整")
        elif problematic_ratio > 0.1:
            suggestions.append("局部问题较多，建议进行OD调整")
        else:
            suggestions.append("问题较少，可进行精细化车道调整")
        
        return suggestions 