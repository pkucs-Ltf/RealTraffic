import os
import json
import requests
import subprocess
from typing import Any, Tuple, List
from mcp.server.fastmcp import FastMCP
import geopandas as gpd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from shapely.geometry import Polygon
import pyproj
import os
import json
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from datetime import datetime
import math
import numpy as np
from shapely.ops import transform
import time
import random
from math import cos, radians
from typing import Any, Dict, List, Tuple
from mcp.server.fastmcp import FastMCP
import os
import pickle
from typing import Optional, Dict, List
from datetime import datetime
import geopandas as gpd
from mcp.server.fastmcp import FastMCP
from tool.Carttils import is_short_line_almost_on_long_line, transfer_status_to_num, find_max_value

from tool.get_edgelimit import extract_speed_limits_from_network

from tool.get_road_to_status import mcp_match_sumo_edges_to_real_road_status,convert_sumo_shapes_to_latlon


from tool.generate_map_html import generate_map_with_marker
# 初始化 MCP 服务器
mcp = FastMCP("RoadNetworkServer")
API_KEY_GAODE = '2f9703a291eedacb4e5f3b9f2e9b1e85'



# 坐标转换参数
x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 扁率

def generate_rectangle(center_point: str, radius_km: float) -> Dict[str, Any]:
    """通过中心点和半径生成矩形范围"""
    try:
        center_lon, center_lat = map(float, center_point.split(','))
        lat_diff = radius_km / 111.0
        lon_diff = radius_km / (111.0 * cos(radians(center_lat)))

        min_lon = center_lon - lon_diff
        min_lat = center_lat - lat_diff
        max_lon = center_lon + lon_diff
        max_lat = center_lat + lat_diff

        rectangle = f"{min_lon:.6f},{min_lat:.6f};{max_lon:.6f},{max_lat:.6f}"

        return {
            'rectangle': rectangle,
            'center': [center_lat, center_lon],
            'bounds': [[min_lat, min_lon], [max_lat, max_lon]]
        }
    except Exception as e:
        return None

def gcj02towgs84(lng: float, lat: float) -> Tuple[float, float]:
    """GCJ02(火星坐标系)转GPS84"""
    if out_of_china(lng, lat):
        return lng, lat
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]

def transformlat(lng: float, lat: float) -> float:
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 * math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 * math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret

def transformlng(lng: float, lat: float) -> float:
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 * math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 * math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret

def out_of_china(lng: float, lat: float) -> bool:
    """判断是否在国内"""
    return lng < 72.004 or lng > 137.8347 or lat < 0.8293 or lat > 55.8271

def transform_geometry(geom):
    """转换几何对象的坐标"""
    def transform_coords(x, y, z=None):
        wgs_x, wgs_y = gcj02towgs84(x, y)
        return wgs_x, wgs_y
    return transform(transform_coords, geom)

def get_traffic_data(rectangle: str, api_key: str) -> gpd.GeoDataFrame:
    """获取交通数据并返回GeoDataFrame"""
    try:
        current_time = datetime.now()
        data = {
            'name': [], 'status': [], 'status_desc': [], 'direction': [],
            'angle': [], 'speed': [], 'timestamp': [], 'date': [],
            'time': [], 'geometry': []
        }

        status_mapping = {
            "0": "未知", "1": "畅通", "2": "缓行", "3": "拥堵"
        }

        url = f'https://restapi.amap.com/v3/traffic/status/rectangle?rectangle={rectangle}&output=json&extensions=all&key={api_key}&level=6'
        res = requests.get(url, timeout=10).json()

        for road in res['trafficinfo']['roads']:
            try:
                polylines = [(float(y[0]), float(y[1])) for y in
                            [x.split(',') for x in road['polyline'].split(';')]]
                line = LineString(polylines)
                wgs84_line = transform_geometry(line)
                status = road.get('status', '0')

                data['geometry'].append(wgs84_line)
                data['name'].append(road.get('name', ''))
                data['status'].append(float(status))
                data['status_desc'].append(status_mapping.get(status, '未知'))
                data['direction'].append(road.get('direction', ''))
                data['angle'].append(float(road.get('angle', 0)))
                data['speed'].append(int(road.get('speed', 0)))
                data['timestamp'].append(current_time.strftime("%Y-%m-%d %H:%M:%S"))
                data['date'].append(current_time.strftime("%Y-%m-%d"))
                data['time'].append(current_time.strftime("%H:%M:%S"))
            except Exception:
                continue

        return gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
    except Exception:
        return None
def download_osm_data(bbox: Tuple[float, float, float, float], osm_file: str) -> bool:
    """
    从OpenStreetMap下载指定边界框内的路网数据
    
    Args:
        bbox: WGS84格式的地理边界框 (south, west, north, east)
        osm_file: 保存OSM数据的文件名
    
    Returns:
        bool: 下载是否成功
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    overpass_url = "https://overpass-api.de/api/map"
    params = {"bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}"}
    
    try:
        if  os.path.exists(osm_file):
            return True
        response = requests.get(overpass_url, params=params, timeout=180)
        response.raise_for_status()
        with open(osm_file, 'wb') as f:
            f.write(response.content)
        return True
    except requests.exceptions.RequestException as e:
        return False

def convert_to_sumo_net(osm_file: str, output_net_file: str) -> bool:
    """
    使用SUMO的netconvert工具将OSM数据转换为.net.xml文件
    
    Args:
        osm_file: OSM数据文件名
        output_net_file: 输出的SUMO路网文件名
    
    Returns:
        bool: 转换是否成功
    """
    # 检查输入文件是否存在
    if not os.path.exists(osm_file):
        print(f"错误：输入文件不存在: {osm_file}")
        return False
    
    netconvert_cmd = [
        'netconvert',
        '--osm-files', osm_file,
        '-o', output_net_file,
        '--tls.guess-signals',
        '--tls.discard-simple',
        '--tls.join',
        '--junctions.join',
        '--ramps.guess'
    ]
    
    try:
        result = subprocess.run(
            netconvert_cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            print(f"[+] netconvert 执行成功，输出文件: {output_net_file}")
            return True
        else:
            print(f"错误：netconvert 执行失败 (退出码: {result.returncode})")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
    except Exception as e:
        print(f"错误：执行netconvert时发生异常: {str(e)}")
        return False


def filter_sumo_network(input_net_file: str, output_net_file: str) -> bool:
    """
    使用 netconvert 过滤 SUMO 网络文件，移除指定类型的道路
    
    Args:
        input_net_file: 输入的 SUMO 网络文件路径
        output_net_file: 输出的过滤后网络文件路径
        
    Returns:
        bool: 处理是否成功
    """
    netconvert_cmd = [
        'netconvert',
        '--sumo-net-file', input_net_file,
        '--remove-edges.by-type', 'railway.subway,highway.footway,railway.rail,highway.pedestrian,railway.tram,highway.cycleway,highway.cycleway',
        '--output-file', output_net_file,
        '--junctions.join', 'true',
        '--junctions.join-dist', '50',
        '--remove-edges.isolated',
        '--tls.guess'
    ]
    

    print(f"正在处理网络文件: {input_net_file}")
    print(f"输出文件: {output_net_file}")
    print("执行命令:", ' '.join(netconvert_cmd))
    
    result = subprocess.run(
        netconvert_cmd,
        check=True,
        capture_output=True,
        text=True
    )
    
    print("网络过滤成功!")
    if result.stdout:
        print("输出信息:", result.stdout)
    return True
        
   

def generate_grid_polygons(
    min_lat: float,
    min_lon: float, 
    max_lat: float,
    max_lon: float,
    grid_size: int = 500,
    output_dir: str = "output",
    output_file: str = "taz_polygons.shp"
) -> str:
    """
    切割网格多边形并保存为shapefile
    
    Args:
        min_lat: 最小纬度
        min_lon: 最小经度
        max_lat: 最大纬度
        max_lon: 最大经度
        grid_size: 网格大小(米)
        output_dir: 输出目录
        output_file: 输出文件名
        
    Returns:
        str: 操作结果信息
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义坐标系
        wgs84 = pyproj.CRS('EPSG:4326')
        web_mercator = pyproj.CRS('EPSG:3857')
        transformer = pyproj.Transformer.from_crs(wgs84, web_mercator, always_xy=True)

        # 坐标转换
        min_x, min_y = transformer.transform(min_lon, min_lat)
        max_x, max_y = transformer.transform(max_lon, max_lat)

        # 生成网格
        polygons = []
        current_y = min_y
        while current_y < max_y:
            current_x = min_x
            while current_x < max_x:
                polygon = Polygon([
                    (current_x, current_y),
                    (current_x + grid_size, current_y),
                    (current_x + grid_size, current_y + grid_size),
                    (current_x, current_y + grid_size)
                ])
                polygons.append(polygon)
                current_x += grid_size
            current_y += grid_size

        # 创建GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=web_mercator)
        gdf = gdf.to_crs(wgs84)
        
        # 保存文件
        output_path = os.path.join(output_dir, output_file)
        gdf.to_file(output_path)
        
        print(f"[+] 已生成 {len(polygons)} 个网格多边形")
        print(f"[+] 已保存到: {output_path}")
        
        return f"成功：网格多边形已保存到 {output_path}"
        
    except Exception as e:
        return f"错误：生成网格多边形失败 - {str(e)}"




def get_real_traffic_state(
    center_point: str,
    radius_km: float,
    output_dir: str = "output",
    api_key: str = API_KEY_GAODE
) -> str:
    """
    获取指定区域的实时交通状态并保存为shapefile格式。
    
    Args:
        center_point: 中心点坐标，格式："经度,纬度"（如："114.4567,30.4830"）
        radius_km: 查询半径（公里）
        output_dir: 输出文件夹路径
        api_key: 高德地图API密钥
    
    Returns:
        str: 操作结果信息
    """
    try:
        html_filename = f'traffic_status.html'
        html_path = os.path.join(output_dir, html_filename)
        # if os.path.exists(html_path):
        #     return f"成功：已将交通数据保存到 {html_path}"
        # 生成矩形区域
        rect_info =  generate_rectangle(center_point, radius_km)
        if not rect_info:
            return "错误：无法生成查询区域"

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 获取交通数据
        gdf =  get_traffic_data(rect_info['rectangle'], api_key)
        if gdf is None or len(gdf) == 0:
            return "错误：未获取到交通数据"

        # 生成输出文件名
        current_time = datetime.now()
        filename = f'traffic_status_wgs84.shp'
        output_path = os.path.join(output_dir, filename)

        # 保存为shapefile
        gdf.to_file(output_path, encoding='utf-8')
        
        
        edges_latlon_file='pkl/edges_Lat_lon_dict_.pkl'
               
        print(f"[*] 正在转换路况数据...")
        
        # 转换数据
        # traffic_status = convert_real_data_to_sumo(output_path, edges_latlon_file, threshold=0.0002)
        
        if not traffic_status:
            return "错误：未能转换任何路况数据"
        
        # 生成输出文件名
     
        output_pkl = f"pkl/traffic_status.pkl"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
        
        # 保存到pkl文件
        with open(output_pkl, 'wb') as f:
            pickle.dump(traffic_status, f)
        
        print(f"[+] 路况数据已保存到: {output_pkl}")
        
        # 统计信息
        status_counts = {}
        for status in traffic_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1
        
        stats = f"""
        统计信息:
        - 总边数: {len(traffic_status)}
        - 状态分布:
        - 畅通(0): {status_counts.get(0, 0)} 条
        - 缓行(1): {status_counts.get(1, 0)} 条
        - 拥堵(2): {status_counts.get(2, 0)} 条
        """
        print(stats)
        
        print(f"成功：已转换 {len(traffic_status)} 条边的路况数据，保存到 {output_pkl}{stats}")
        


        # 可视化所有道路，根据 status_des 字段着色，并保存为HTML
        import folium

        # 计算地图中心
        if len(gdf) > 0:
            center_lat = gdf.geometry.centroid.y.mean()
            center_lon = gdf.geometry.centroid.x.mean()
        else:
            center_lat, center_lon = 30.0, 114.0  # 默认值

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')

        color_map = {
            "拥堵": "red",
            "缓行": "yellow",
            "畅通": "green"
        }

        for idx, row in gdf.iterrows():
            status = row.get("status_des", "")
            color = color_map.get(status, None)
            if color is None:
                continue
            geom = row.geometry
            # 只处理LineString和MultiLineString
            if geom.geom_type == "LineString":
                points = [[lat, lon] for lon, lat in geom.coords]
                folium.PolyLine(points, color=color, weight=5, opacity=0.8, popup=f"{status}").add_to(m)
            elif geom.geom_type == "MultiLineString":
                for linestring in geom.geoms:
                    points = [[lat, lon] for lon, lat in linestring.coords]
                    folium.PolyLine(points, color=color, weight=5, opacity=0.8, popup=f"{status}").add_to(m)

        html_filename = f'traffic_status_{current_time.strftime("%Y%m%d")}.html'
        html_path = os.path.join(output_dir, html_filename)
        m.save(html_path)

        return f"成功：已将交通数据保存到 {output_path}，共 {len(gdf)} 条道路数据"

    except Exception as e:
        return f"错误：处理失败 - {str(e)}"



def convert_real_data_to_sumo(
    shp_file: str,
    edges_latlon_file: str,
    threshold: float = 0.0002
) -> Dict[str, int]:

    # 读取shp文件
    gdf = gpd.read_file(shp_file)
    
    # 加载边界经纬度信息
    with open(edges_latlon_file, 'rb') as f:
        id_LatLon = pickle.load(f)
    
    # 转换数据
    dict_record_id_APILineStatus = {}
    for edge_id in id_LatLon.keys():
        for index, row in gdf.iterrows():
            geometry = row.geometry
            Geolist = [list(k) for k in geometry.coords]
            
            if is_short_line_almost_on_long_line(id_LatLon[edge_id], Geolist, threshold):
                if row['status_des'] == '未知':
                    continue
                if edge_id not in dict_record_id_APILineStatus:
                    dict_record_id_APILineStatus[edge_id] = [transfer_status_to_num(row['status_des'])]
                else:
                    dict_record_id_APILineStatus[edge_id].append(transfer_status_to_num(row['status_des']))
    
    # 处理状态值
    for edge_id in dict_record_id_APILineStatus:
        dict_record_id_APILineStatus[edge_id] = find_max_value(dict_record_id_APILineStatus[edge_id])
    
    return dict_record_id_APILineStatus

class Net_Query(BaseModel):
    bbox: Tuple[float, float, float, float] = Field(description="")
    output_net_file: str = Field(description="WGS84格式的地理边界框 (south, west, north, east)")
    osm_file: str = Field(description="The output file name of the OSM file")
    center_point: str = Field(description="The center point of the city")
    radius_km: float = Field(description="The radius of the city")


@tool(args_schema = Net_Query)
def create_sumo_net_from_bbox(
    bbox: Tuple[float, float, float, float],
    output_net_file: str = "network.net.xml",
    osm_file: str = "map.osm.xml",
    center_point: str = "114.4567,30.4830",
    radius_km: float = 10.0,
) -> str:
    """
    从OpenStreetMap下载指定边界框(BBOX)内的路网数据,并转换为SUMO路网文件,并预处理。
    
    Args:
        bbox: WGS84格式的地理边界框 (south, west, north, east)
        output_net_file: 输出的SUMO路网文件名，默认为 "network.net.xml"
        osm_file: 临时保存的OSM数据文件名，默认为 "map.osm.xml"
    
    Returns:
        str: 操作结果信息
    """
    # 下载OSM数据

    osm_file='peking_university_map.osm.xml'
    output_net_file='Manha.net.xml'
    if os.path.exists(output_net_file):
        return f"路网已生成，成功：路网已保存到 {output_net_file}"
    print(f"[*] 正在从 Overpass API 下载区域 {bbox} 的路网数据...")
    if not download_osm_data(bbox, osm_file):
        return "错误：下载OSM数据失败"
    print(f"[+] OSM 数据成功保存到: {osm_file}")
    
    # 转换为SUMO路网
    print("[*] 正在调用 SUMO netconvert...")
    if not convert_to_sumo_net(osm_file, output_net_file):
        return "错误：SUMO路网转换失败"
    
    filter_sumo_network(output_net_file, output_net_file)
    
    # # 浏览器展示sumo路网，展示中心点
    # generate_map_with_marker(center_point.split(',')[1],center_point.split(',')[0])
    
    # sumo 转换到latlon
    lat_lon_edges=convert_sumo_shapes_to_latlon(output_net_file)
    
    
    # 匹配SUMO边到实际道路状态路况
    real_data='pkl/dict_record_id_APILineStatus_core_500m.pkl'
    mcp_match_sumo_edges_to_real_road_status('peking_univ_output\\traffic_status_wgs84.shp',output_net_file,real_data,lat_lon_edges)
    
    
    # 提取速度限制并存下来
    extract_speed_limits_from_network(output_pkl='pkl/edges_limit_Lat_lon_dict_temp.pkl',net_file=output_net_file,signal_name='network')
    
    print("[+] SUMO 路网转换成功！")
    output_dir = "output"
    get_real_traffic_state(center_point,radius_km,output_dir,API_KEY_GAODE)
    
    generate_grid_polygons(bbox[0], bbox[1], bbox[2], bbox[3])
    
    return f"成功：路网已保存到 {output_net_file}"




