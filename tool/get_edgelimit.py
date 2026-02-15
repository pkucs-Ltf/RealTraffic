#!/usr/bin/env python3
"""
SUMO网络速度限制提取MCP服务器
提供从SUMO网络文件中提取边的速度限制功能
"""

import xml.etree.ElementTree as ET
import pickle
import os
from datetime import datetime
from typing import Optional, Dict
from mcp.server.fastmcp import FastMCP


def extract_edge_speed_limits(net_file: str) -> Dict[str, float]:
    """提取所有 edge 的最小车道限速"""
    tree = ET.parse(net_file)
    root = tree.getroot()

    speed_dict = {}

    # 遍历所有 edge 元素
    for edge in root.findall('edge'):
        
        edge_id = edge.get('id')
        if edge_id[0] ==":":
            continue
        lane_speeds = []

        # 遍历该 edge 下的所有 lane 元素
        for lane in edge.findall('lane'):
            speed = lane.get('speed')
            if speed:
                lane_speeds.append(float(speed))

        if lane_speeds:
            # 取最小速度作为该 edge 的限速
            speed_dict[edge_id] = min(lane_speeds)

    return speed_dict


def extract_speed_limits_from_network(
    output_pkl: Optional[str] = None,
    net_file: str = "network.net.xml",
    signal_name: str = "network"
) -> str:
    """
    从SUMO网络文件中提取边的速度限制并保存为pkl文件
    
    Args:
        net_file: SUMO网络文件路径 (.net.xml)
        output_pkl: 输出pkl文件路径（可选，默认自动生成）
        signal_name: 信号名称，用于生成默认文件名
    
    Returns:
        str: 操作结果信息
    """
    try:
        output_pkl='pkl/edges_limit_Lat_lon_dict_temp.pkl'
        net_file='peking_univ_network.net.xml'
        # 检查输入文件是否存在
        if not os.path.exists(net_file):
            return f"错误：网络文件 {net_file} 不存在"
        
        print(f"[*] 正在从 {net_file} 提取速度限制...")
        
        # 提取速度限制
        speed_limits = extract_edge_speed_limits(net_file)
        
        if not speed_limits:
            return "错误：未能从网络文件中提取到任何速度限制信息"


        # 生成输出文件名
        if output_pkl is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_pkl = f"pkl/edge_speed_limits_{signal_name}.pkl"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
        
        # 保存到pkl文件
        with open(output_pkl, 'wb') as f:
            pickle.dump(speed_limits, f)
        
        print(f"[+] 速度限制已保存到: {output_pkl}")
        
        # 统计信息
        speeds_ms = list(speed_limits.values())
        min_speed = min(speeds_ms)
        max_speed = max(speeds_ms)
        avg_speed = sum(speeds_ms) / len(speeds_ms)
        
        stats = f"""
统计信息:
- 总边数: {len(speed_limits)}
- 最小速度: {min_speed:.2f} m/s ({min_speed * 3.6:.2f} km/h)
- 最大速度: {max_speed:.2f} m/s ({max_speed * 3.6:.2f} km/h)
- 平均速度: {avg_speed:.2f} m/s ({avg_speed * 3.6:.2f} km/h)
"""
        print(stats)
        
        return f"成功：已提取 {len(speed_limits)} 条边的速度限制，保存到 {output_pkl}{stats}"
        
    except Exception as e:
        return f"错误：提取速度限制失败 - {str(e)}"


