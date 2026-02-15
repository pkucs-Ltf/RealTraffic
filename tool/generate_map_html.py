import os
import json
import webbrowser
import folium
from datetime import datetime
from typing import Tuple, Optional
from mcp.server.fastmcp import FastMCP



def generate_map_with_marker(
    latitude: float,
    longitude: float,
    zoom_level: int = 15,
    output_file: Optional[str] = None,
    auto_open: bool = True
) -> str:
    """
    生成带有红色透明标记的地图HTML文件
    
    Args:
        latitude: 纬度坐标
        longitude: 经度坐标
        zoom_level: 地图缩放级别，默认15（约1公里范围）
        output_file: 输出HTML文件名，默认为自动生成的时间戳文件名
        auto_open: 是否自动在浏览器中打开生成的HTML文件
    
    Returns:
        str: 操作结果信息
    """
    try:
        # 如果没有指定输出文件名，则生成带时间戳的文件名
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            output_file = f"traffic_status.html"
        
        # 确保输出文件有.html扩展名
        if not output_file.endswith('.html'):
            output_file += '.html'
        
        print(f"[*] 正在生成地图，坐标: ({latitude}, {longitude})")
        
        # 创建地图对象，以指定坐标为中心
        map_obj = folium.Map(
            location=[latitude, longitude],
            zoom_start=zoom_level,
            tiles='OpenStreetMap'
        )
        
        # 添加透明度50%的红色标记
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=10,
            popup=f'坐标: ({latitude}, {longitude})',
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.5,
            opacity=0.5
        ).add_to(map_obj)
        
        # 添加一个圆形区域显示1公里范围（可选）
        folium.Circle(
            location=[latitude, longitude],
            radius=1000,  # 1公里
            popup='1公里范围',
            color='blue',
            fill=False,
            opacity=0.3
        ).add_to(map_obj)
        
        # 保存地图到HTML文件
        map_obj.save(output_file)
        print(f"[+] 地图已保存到: {output_file}")
        
        # 自动在浏览器中打开文件
        if auto_open:
            try:
                # 获取文件的绝对路径
                abs_path = os.path.abspath(output_file)
                webbrowser.open(f'file://{abs_path}')
                print(f"[+] 已在浏览器中打开: {abs_path}")
            except Exception as e:
                print(f"[!] 无法自动打开浏览器: {str(e)}")
        
        return f"成功：地图已生成并保存到 {output_file}"
        
    except Exception as e:
        return f"错误：地图生成失败 - {str(e)}"



