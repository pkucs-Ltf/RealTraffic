"""
流量规模调整器
基于现有的optimization_pipeline逻辑，提供scale调整功能
"""
from collections import defaultdict

import os
import sys
import numpy as np
import pickle
from typing import Dict, Any, Tuple, List, Optional
import xml.etree.ElementTree as ET
import shutil
from scipy.optimize import minimize_scalar

# 添加父目录到路径以导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool.Carttils import *
        

class ODScaler:
    def __init__(self, input_file, output_dir):
        """
        初始化ODScaler类
        
        参数:
        input_file: 输入路由文件路径
        output_dir: 输出文件夹路径
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.scale_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


class ScaleAdjuster:
    """流量规模调整器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化调整器
        
        Args:
            config: 调整器配置参数
        """
        self.config = config
        self.scale_config = config.get('scale', {})
        self.search_points = self.scale_config.get('search_points', [0.5, 1.0, 1.5, 2.0])
        self.fine_tune_range = self.scale_config.get('fine_tune_range', 0.1)
        self.temp_dir = config.get('temp_dir', 'temp/')
        self.current_scale = 1.0
        
    def sort_vehicles_by_depart_time(self, input_xml_path, output_xml_path):
        """
        读取XML文件，按照vehicle的depart时间排序，并保存到新文件
        """
        tree = ET.parse(input_xml_path)
        root = tree.getroot()
        
        vehicles = root.findall('vehicle')
        vehicles.sort(key=lambda v: float(v.get('depart', '0')))
        
        for vehicle in list(root):
            if vehicle.tag == 'vehicle':
                root.remove(vehicle)
                
        for vehicle in vehicles:
            root.append(vehicle)
            
        tree.write(output_xml_path, encoding='utf-8', xml_declaration=True, method='xml')

    def propose(self, 
                prev_params: Dict[str, Any], 
                diff_dict: Dict[str, Any],
                rou_file: str) -> Tuple[Dict[str, Any], str]:
        """
        基于误差分析提出新的scale参数
        
        Args:
            prev_params: 之前的参数
            diff_dict: 误差分析结果
            rou_file: 当前路由文件
            
        Returns:
            (新参数, 调整描述)
        """
        current_scale = prev_params.get('scale', 1.0)
        
        # 分析误差模式决定调整方向
        adjustment_direction = self._analyze_scale_direction(diff_dict)
        
        if adjustment_direction == 'increase':
            new_scale = self._suggest_increase(current_scale)
            description = f"增加scale从{current_scale:.2f}到{new_scale:.2f}：仿真流量过低"
        elif adjustment_direction == 'decrease':
            new_scale = self._suggest_decrease(current_scale)
            description = f"减少scale从{current_scale:.2f}到{new_scale:.2f}：仿真流量过高"
        else:
            # 进行精细调优
            new_scale = self._fine_tune_scale(current_scale, diff_dict)
            description = f"精细调整scale从{current_scale:.2f}到{new_scale:.2f}"
        
        new_params = {
            'scale': new_scale,
            'adjustment_type': 'scale',
            'previous_scale': current_scale
        }
        
        return new_params, description
    

    def apply_adjustment(self, 
                        rou_file: str, 
                        new_params: Dict[str, Any],
                        output_file: Optional[str] = None) -> str:
        new_scale = new_params['scale']
        scale=new_scale
        print(f"处理缩放比例: {scale}")
        
        tree = ET.parse(rou_file)
        root = tree.getroot()
        print(f"根标签: {root.tag}")
        print("所有标签:")
        
        root = tree.getroot()
        
        vehicles = root.findall('vehicle')
        od_matrix = defaultdict(list)
        
        for vehicle in vehicles:
            from_taz = vehicle.get('fromTaz')
            to_taz = vehicle.get('toTaz')
            
            if from_taz and to_taz:
                od_pair = (from_taz, to_taz)
                od_matrix[od_pair].append(vehicle)
        
        od_counts = {od_pair: len(vehicles_list) for od_pair, vehicles_list in od_matrix.items()}
        
        new_root = ET.Element(root.tag, root.attrib)
        
        for child in root:
            if child.tag != 'vehicle':
                new_root.append(ET.fromstring(ET.tostring(child, encoding='unicode')))
        
        kept_vehicles = []
        
        for od_pair, vehicles_list in od_matrix.items():
            if scale > 1:
                kept_vehicles.extend(vehicles_list)
                n = int(scale)
                decimal = scale - n
                
                for i in range(n-1):
                    for vehicle in vehicles_list:
                        new_vehicle = ET.fromstring(ET.tostring(vehicle, encoding='unicode'))
                        new_id = 's' * (i+1) + vehicle.get('id')
                        new_vehicle.set('id', new_id)
                        kept_vehicles.append(new_vehicle)
                
                if decimal > 0:
                    num_decimal = int(len(vehicles_list) * decimal)
                    if num_decimal > 0:
                        decimal_vehicles = random.sample(vehicles_list, num_decimal)
                        for vehicle in decimal_vehicles:
                            new_vehicle = ET.fromstring(ET.tostring(vehicle, encoding='unicode'))
                            new_id = 's' * n + vehicle.get('id')
                            new_vehicle.set('id', new_id)
                            kept_vehicles.append(new_vehicle)
            
            else:
                num_to_keep = int(round(len(vehicles_list) * scale))
                if num_to_keep > 0:
                    vehicles_to_keep = random.sample(vehicles_list, num_to_keep)
                    kept_vehicles.extend(vehicles_to_keep)
        
        for vehicle in kept_vehicles:
            new_root.append(ET.fromstring(ET.tostring(vehicle, encoding='unicode')))
        
        new_tree = ET.ElementTree(new_root)
        output_file = os.path.join(output_file, f"scaled_{int(scale*100)}percent.rou.xml")
        
        new_tree.write(output_file, encoding='utf-8', xml_declaration=True, method='xml')
        
        self.sort_vehicles_by_depart_time(output_file, output_file)
        
        print(f"生成文件: {output_file}, 保留了 {len(kept_vehicles)}/{len(vehicles)} 个车辆")
        print(f"原始OD矩阵包含 {len(od_matrix)} 个OD对")

        print("所有缩放处理完成!")
        return output_file


   
    
    def _analyze_scale_direction(self, diff_dict: Dict[str, Any]) -> str:
        """分析应该增加还是减少scale"""
        summary = diff_dict.get('summary', {})
        edge_level = diff_dict.get('edge_level', {})
        
        # 统计速度偏差的总体趋势
        speed_bias = 0
        density_bias = 0
        count = 0
        
        for edge_id, diff in edge_level.items():
            sim_speed = diff['sim_values'].get('avg_speed', 0)
            real_speed = diff['real_values'].get('speed', 0)
            
            sim_density = diff['sim_values'].get('avg_density', 0) 
            real_density = diff['real_values'].get('density', 0)
            
            if real_speed > 0:
                speed_bias += (sim_speed - real_speed) / real_speed
                count += 1
                
            if real_density > 0:
                density_bias += (sim_density - real_density) / real_density
        
        if count > 0:
            avg_speed_bias = speed_bias / count
            avg_density_bias = density_bias / count
            
            # 如果仿真速度普遍高于真实（密度低于真实），说明流量不够，需要增加scale
            if avg_speed_bias > 0.1 and avg_density_bias < -0.1:
                return 'increase'
            # 如果仿真速度普遍低于真实（密度高于真实），说明流量过多，需要减少scale
            elif avg_speed_bias < -0.1 and avg_density_bias > 0.1:
                return 'decrease'
        
        return 'fine_tune'
    
    def _suggest_increase(self, current_scale: float) -> float:
        """建议增加scale值"""
        # 根据当前scale值决定增加幅度
        if current_scale < 1.0:
            increment = 0.2
        elif current_scale < 5.0:
            increment = 0.5
        else:
            increment = 1.0
        
        new_scale = min(current_scale + increment, self.scale_config.get('max_value', 20.0))
        return new_scale
    
    def _suggest_decrease(self, current_scale: float) -> float:
        """建议减少scale值"""
        # 根据当前scale值决定减少幅度
        if current_scale > 5.0:
            decrement = 1.0
        elif current_scale > 1.0:
            decrement = 0.5
        else:
            decrement = 0.1
        
        new_scale = max(current_scale - decrement, self.scale_config.get('min_value', 0.2))
        return new_scale
    
    def _fine_tune_scale(self, current_scale: float, diff_dict: Dict[str, Any]) -> float:
        """精细调整scale值"""
        # 在当前scale附近进行小幅调整
        tune_range = self.fine_tune_range
        
        # 基于误差大小决定调整方向和幅度
        overall_error = diff_dict.get('summary', {}).get('speed', {}).get('mae', 0)
        
        if overall_error > 5:  # 误差较大
            adjustment = tune_range * 0.5
        else:  # 误差较小
            adjustment = tune_range * 0.2
        
        # 随机选择增加或减少（可以改进为基于梯度的方法）
        direction = np.random.choice([-1, 1])
        new_scale = current_scale + direction * adjustment
        
        # 确保在合理范围内
        min_scale = self.scale_config.get('min_value', 0.2)
        max_scale = self.scale_config.get('max_value', 20.0)
        new_scale = max(min_scale, min(max_scale, new_scale))
        
        return new_scale
    
    def _apply_scale_to_route_file(self, input_file: str, output_file: str, scale: float):
        """将scale应用到路由文件"""
        try:
            # 解析XML文件
            tree = ET.parse(input_file)
            root = tree.getroot()
            
            # 找到所有的vehicle和flow元素
            vehicles_modified = 0
            flows_modified = 0
            
            for vehicle in root.findall('.//vehicle'):
                # 修改vehicle的出发时间（通过scale调整密度）
                depart = vehicle.get('depart')
                if depart and depart != 'triggered':
                    try:
                        original_time = float(depart)
                        new_time = original_time / scale  # scale越大，车辆越密集
                        vehicle.set('depart', str(new_time))
                        vehicles_modified += 1
                    except ValueError:
                        pass
            
            for flow in root.findall('.//flow'):
                # 修改flow的频率
                if flow.get('vehsPerHour'):
                    original_rate = float(flow.get('vehsPerHour'))
                    new_rate = original_rate * scale
                    flow.set('vehsPerHour', str(new_rate))
                    flows_modified += 1
                elif flow.get('period'):
                    original_period = float(flow.get('period'))
                    new_period = original_period / scale
                    flow.set('period', str(new_period))
                    flows_modified += 1
                elif flow.get('probability'):
                    original_prob = float(flow.get('probability'))
                    new_prob = min(1.0, original_prob * scale)
                    flow.set('probability', str(new_prob))
                    flows_modified += 1
            
            # 保存修改后的文件
            tree.write(output_file, encoding='utf-8', xml_declaration=True)
            
            print(f"Scale调整完成: {vehicles_modified}个vehicle, {flows_modified}个flow被修改")
            
        except Exception as e:
            print(f"应用scale调整时出错: {e}")
            # 如果出错，复制原文件
            shutil.copy2(input_file, output_file)
    
    def grid_search_best_scale(self, 
                              rou_file: str,
                              evaluation_func: callable,
                              search_points: Optional[List[float]] = None) -> Tuple[float, float]:
        """
        网格搜索最佳scale值
        
        Args:
            rou_file: 路由文件
            evaluation_func: 评估函数，输入路由文件，返回评分
            search_points: 搜索点列表
            
        Returns:
            (最佳scale, 最佳评分)
        """
        if search_points is None:
            search_points = self.search_points
        
        best_scale = 1.0
        best_score = float('inf')
        
        for scale in search_points:
            # 创建临时文件
            temp_rou = self.apply_adjustment(rou_file, {'scale': scale})
            
            try:
                # 评估当前scale
                score = evaluation_func(temp_rou)
                
                if score < best_score:
                    best_score = score
                    best_scale = scale
                    
                print(f"Scale {scale:.2f}: 评分 {score:.4f}")
                
            except Exception as e:
                print(f"评估scale {scale}时出错: {e}")
                continue
            finally:
                # 清理临时文件
                if os.path.exists(temp_rou):
                    try:
                        os.remove(temp_rou)
                    except:
                        pass
        
        return best_scale, best_score 