"""
动态状态空间设计示例
展示根据实际路口结构动态生成状态空间的方案
"""

import numpy as np

def demonstrate_dynamic_state_space():
    """演示动态状态空间设计"""
    print("=== 动态状态空间设计示例 ===")
    print()
    
    print("🎯 核心改进：不再假设固定的方向数和车道类型")
    print("✅ 根据实际路口结构动态计算状态维度")
    print()
    
    # 示例1：简单T型路口
    print("示例1：T型路口（3个进入车道，2个相位）")
    print("   进入车道：['lane_north_0', 'lane_south_0', 'lane_east_0']")
    print("   相位数量：2个 (南北通行, 东西通行)")
    
    t_junction_lanes = ['lane_north_0', 'lane_south_0', 'lane_east_0']
    t_junction_vehicles = [5, 8, 3]  # 每个车道的车辆数
    t_junction_phase = 0  # 当前相位0
    
    # 道路状态：每个车道的车辆数（归一化）
    road_state = [v/50.0 for v in t_junction_vehicles]  # 假设最大容量50
    # 相位状态：2个相位的one-hot编码
    phase_state = [1.0, 0.0] if t_junction_phase == 0 else [0.0, 1.0]
    
    full_state = road_state + phase_state
    print(f"   状态向量：{[f'{x:.3f}' for x in full_state]}")
    print(f"   维度：{len(t_junction_lanes)}个车道 + 2个相位 = {len(full_state)}维")
    print()
    
    # 示例2：复杂十字路口
    print("示例2：复杂十字路口（8个进入车道，4个相位）")
    print("   进入车道：['n_left', 'n_straight', 's_left', 's_straight', 'e_left', 'e_straight', 'w_left', 'w_straight']")
    print("   相位数量：4个")
    
    cross_lanes = ['n_left', 'n_straight', 's_left', 's_straight', 'e_left', 'e_straight', 'w_left', 'w_straight']
    cross_vehicles = [2, 12, 4, 8, 1, 15, 3, 6]  # 每个车道的车辆数
    cross_phase = 2  # 当前相位2
    
    # 道路状态
    road_state = [v/50.0 for v in cross_vehicles]
    # 相位状态：4个相位的one-hot编码
    phase_state = [0.0, 0.0, 1.0, 0.0]  # 相位2激活
    
    full_state = road_state + phase_state
    print(f"   状态向量：{[f'{x:.3f}' for x in full_state]}")
    print(f"   维度：{len(cross_lanes)}个车道 + 4个相位 = {len(full_state)}维")
    print()
    
    # 示例3：不规则路口
    print("示例3：不规则路口（5个进入车道，3个相位）")
    print("   进入车道：['approach_a', 'approach_b1', 'approach_b2', 'approach_c', 'approach_d']")
    print("   相位数量：3个")
    
    irregular_lanes = ['approach_a', 'approach_b1', 'approach_b2', 'approach_c', 'approach_d']
    irregular_vehicles = [7, 4, 9, 2, 11]
    irregular_phase = 1
    
    road_state = [v/50.0 for v in irregular_vehicles]
    phase_state = [0.0, 1.0, 0.0]  # 相位1激活
    
    full_state = road_state + phase_state
    print(f"   状态向量：{[f'{x:.3f}' for x in full_state]}")
    print(f"   维度：{len(irregular_lanes)}个车道 + 3个相位 = {len(full_state)}维")
    print()


def demonstrate_dynamic_extraction():
    """演示动态状态提取过程"""
    print("=== 动态状态提取过程 ===")
    print()
    
    print("1. 路口分析阶段：")
    print("   ✅ 从 intersections[tl_id]['phase_available_lanelinks'] 获取所有 start_lanes")
    print("   ✅ 统计实际进入车道数量 N")
    print("   ✅ 统计实际相位数量 M")
    print("   ✅ 确定状态空间维度：N + M")
    print()
    
    print("2. 状态提取阶段：")
    print("   ✅ 按车道ID排序，确保顺序一致性")
    print("   ✅ 逐个提取每个车道的车辆数")
    print("   ✅ 归一化：vehicle_count / max_capacity")
    print("   ✅ 生成当前相位的one-hot编码")
    print()
    
    print("3. 智能体创建阶段：")
    print("   ✅ 为每个路口计算专属的state_dim和action_dim")
    print("   ✅ 创建适配该路口的神经网络")
    print("   ✅ 输出详细的维度信息用于调试")
    print()


def demonstrate_advantages():
    """演示动态设计的优势"""
    print("=== 动态设计的优势 ===")
    print()
    
    print("🎯 灵活性：")
    print("   ✅ 自动适应任何路口结构")
    print("   ✅ 不需要预设方向数和车道类型")
    print("   ✅ 支持T型、十字、环形、不规则路口")
    print()
    
    print("🎯 准确性：")
    print("   ✅ 状态空间完全匹配实际路口")
    print("   ✅ 没有冗余维度")
    print("   ✅ 没有缺失信息")
    print()
    
    print("🎯 可扩展性：")
    print("   ✅ 添加新路口无需修改代码")
    print("   ✅ 支持复杂的多相位信号")
    print("   ✅ 易于集成到现有系统")
    print()
    
    print("🎯 调试友好：")
    print("   ✅ 每个路口输出详细的维度信息")
    print("   ✅ 状态向量含义清晰")
    print("   ✅ 便于问题定位和优化")
    print()


def demonstrate_code_structure():
    """演示代码结构"""
    print("=== 核心代码结构 ===")
    print()
    
    print("```python")
    print("def _calculate_state_dim(self, tl_id: str) -> int:")
    print("    # 动态计算进入车道数量")
    print("    start_lanes = set()")
    print("    for phase_links in self.intersections[tl_id]['phase_available_lanelinks']:")
    print("        for start_lane, end_lane in phase_links[1]:")
    print("            start_lanes.add(start_lane)")
    print("    road_state_dim = len(start_lanes)")
    print("    ")
    print("    # 动态计算相位数量")
    print("    phase_state_dim = len(self.intersections[tl_id]['phase_available_lanelinks'])")
    print("    ")
    print("    return road_state_dim + phase_state_dim")
    print()
    
    print("def _extract_local_traffic_features(self, tl_id: str) -> List[float]:")
    print("    # 获取所有进入车道并排序")
    print("    sorted_lanes = sorted(list(start_lanes))")
    print("    ")
    print("    # 逐个提取车道车辆数")
    print("    for lane_id in sorted_lanes:")
    print("        vehicle_count = self.conn.lane.getLastStepVehicleNumber(lane_id)")
    print("        normalized_count = min(vehicle_count / 50.0, 1.0)")
    print("        features.append(normalized_count)")
    print("```")
    print()


if __name__ == "__main__":
    demonstrate_dynamic_state_space()
    demonstrate_dynamic_extraction()
    demonstrate_advantages()
    demonstrate_code_structure()
    
    print("🎉 新的动态状态空间设计完成！")
    print("✅ 完全根据实际路口结构生成状态空间")
    print("✅ 不再假设固定的方向数和车道类型")
    print("✅ 每个路口都有专属的状态维度")
