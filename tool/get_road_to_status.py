
import pdb
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import KMeans
from tool.Carttils import  *

import matplotlib.pyplot as plt


from geopy.distance import geodesic



import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import KMeans



import warnings
warnings.filterwarnings("ignore")







import matplotlib.pyplot as plt

import geopandas as gpd
import numpy as np



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks



import geopandas as gpd
import numpy as np
def process_xml_output(xml_content):
    # 解析XML内容
    tree = ET.parse(xml_content)
    root = tree.getroot()
    intervals = root.findall('interval')

    # 按时间顺序对interval进行排序
    intervals.sort(key=lambda x: float(x.get('begin')))

    result = {}

    for interval in intervals:

        for edge_elem in interval.findall('edge'):

            edge_id = edge_elem.get('id')

            # 检查是否包含所需的三个属性
            density = edge_elem.get('density')
            speed = edge_elem.get('speed')
            occupancy = edge_elem.get('occupancy')

            if density is not None and speed is not None and occupancy is not None:
                # 将属性值转换为浮点数
                density_val = float(density)
                speed_val = float(speed)
                occupancy_val = float(occupancy)

                # 初始化字典条目（如果不存在）
                if edge_id not in result:
                    result[edge_id] = [[], [], [],[]]

                # 按时间顺序添加数值
                result[edge_id][0].append(density_val)  # 密度
                result[edge_id][1].append(speed_val)  # 车道速度
                result[edge_id][2].append(occupancy_val)  # 占用率
                result[edge_id][3].append(interval.get('begin'))
    return result









def one_way_hausdorff(line1, line2):
    """计算单向Hausdorff距离（单位：米）"""
    if len(line1) == 0 or len(line2) == 0:
        return float('inf')

    distances = haversine_distance(line1, line2)
    return np.max(np.min(distances, axis=1))



from shapely.geometry import LineString
def is_short_line_almost_on_long_line(short_line_coords, long_line_coords, tolerance=1e-6):
    # 创建 LineString 对象
    short_line = LineString(short_line_coords)
    long_line = LineString(long_line_coords)

    # 为长线创建缓冲区
    long_line_buffer = long_line.buffer(tolerance)


    # 检查短线是否完全在长线的缓冲区内
    if long_line_buffer.contains(short_line):

        return True

    # 检查交集长度是否足够长
    intersection = short_line.intersection(long_line_buffer)
    if intersection.length >= short_line.length * (1 - tolerance):
        return True

    return False



def transfer_status_to_num(string1):
    if string1=='畅通':
        return 0
    elif string1=='缓行':
        return 1
    elif string1=='拥堵':
        return 2











def haversine_distance(line1, line2):
    """计算两个经纬度列表之间的Haversine距离矩阵"""
    line1 = np.array(line1)
    line2 = np.array(line2)

    lon1 = np.radians(line1[:, 0])
    lat1 = np.radians(line1[:, 1])
    lon2 = np.radians(line2[:, 0])
    lat2 = np.radians(line2[:, 1])

    dlon = lon2 - lon1[:, np.newaxis]
    dlat = lat2 - lat1[:, np.newaxis]

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1[:, np.newaxis]) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 * 1000  # 地球半径（米）
    return c * r


def one_way_hausdorff(line1, line2):
    """计算单向Hausdorff距离（单位：米）"""
    if len(line1) == 0 or len(line2) == 0:
        return float('inf')

    distances = haversine_distance(line1, line2)
    return np.max(np.min(distances, axis=1))



from shapely.geometry import LineString
def is_short_line_almost_on_long_line(short_line_coords, long_line_coords, tolerance=1e-6):
    # 创建 LineString 对象
    short_line = LineString(short_line_coords)
    long_line = LineString(long_line_coords)

    # 为长线创建缓冲区
    long_line_buffer = long_line.buffer(tolerance)


    # 检查短线是否完全在长线的缓冲区内
    if long_line_buffer.contains(short_line):

        return True

    # 检查交集长度是否足够长
    intersection = short_line.intersection(long_line_buffer)
    if intersection.length >= short_line.length * (1 - tolerance):
        return True

    return False




import numpy as np


def calculate_mae_rmse_recall(actual, predicted):
    # 确保两个列表长度相同
    if len(actual) != len(predicted):
        raise ValueError("两个列表的长度必须相同")

    # 计算召回率
    recall = len([x for x in predicted if x != 0]) / len(actual)


    # 删除两个列表中相同位置都为0的元素
    actual_filtered = []
    predicted_filtered = []
    for a, p in zip(actual, predicted):
        if a != 0 or p != 0:
            actual_filtered.append(a)
            predicted_filtered.append(p)

    # 计算MAE
    mae = np.mean(np.abs(np.array(actual_filtered) - np.array(predicted_filtered)))

    # 计算RMSE
    rmse = np.sqrt(np.mean((np.array(actual_filtered) - np.array(predicted_filtered)) ** 2))

    # 计算准确率
    accuracy = np.mean(np.array(actual_filtered) == np.array(predicted_filtered))



    return mae, rmse, accuracy, recall







def detect_congestion(data, N=3,edge_id=None):
    """
    交通拥堵检测函数
    参数：
    data - 包含道路密度、车道密度、占用率、时间戳的四层嵌套列表
    N    - 峰值倍数阈值（默认3倍）

    返回：
    拥堵时间区间列表 或 None（无拥堵）
    """
    # 解包数据
    road_density, speed, occupancy, times = data
    time_list = [t for t in times]  # 时间戳转换为浮点数
    
    speed_limits=loadfile('pkl/edge_speed_limits_core_500m.pkl')
    speed_limit=speed_limits[edge_id]
    
    # 检查速度差异来判断拥堵
    congestion_windows = []
    
    # 确保数据有效
    if len(speed) < 1 or len(time_list) < 1:
        return False, None
    
    # 遍历所有时间点
    for i in range(len(speed)):
        # 如果自由流速度比当前平均速度大2以上，判定为拥堵
        if speed[i] > 0 and speed_limit / speed[i] > 2:
            # 添加拥堵时间戳
            congestion_windows.append(time_list[i])
    
    # 如果存在拥堵时间点，返回True和排序后的时间戳列表
    if congestion_windows:
        return True, congestion_windows
    # 如果没有检测到拥堵
    return False, None









def find_max_value(lst):
    """
    查找列表中的最大值
    
    参数:
        lst (list): 输入的数字列表
        
    返回:
        最大值，如果列表为空则返回None
    """
    if not lst:
        return None
    
    max_val = lst[0]
    for val in lst:
        if val > max_val:
            max_val = val
    
    return max_val

def match_sumo_edges_to_real_road_status(gdf, id_latlon):
    """匹配SUMO边到实际道路状态"""
    dict_record_id_api_line_status = {}
    for m in id_latlon.keys():
        if m[0] ==':':
            continue
        cal_length=[[lat, lon] for lon, lat in id_latlon[m]]
        length = sum(geodesic(cal_length[i], cal_length[i + 1]).meters for i in range(len(cal_length) - 1))
        if length<5:
            continue
        for index, row in gdf.iterrows():
            geometry = row.geometry
            geolist = [list(k) for k in geometry.coords]

            if is_short_line_almost_on_long_line(id_latlon[m], geolist, 0.0002):

                if row['status_des'] == '未知':
                    continue
                if m not in dict_record_id_api_line_status.keys():
                    dict_record_id_api_line_status[m] = [transfer_status_to_num(row['status_des'])]
                else:
                    dict_record_id_api_line_status[m].append(transfer_status_to_num(row['status_des']))

    # 处理dict_record_id_api_line_status，将列表值加和但限制最大值为2
    for key in dict_record_id_api_line_status.keys():
        # 计算列表中所有值的总和
        sum_value = find_max_value(dict_record_id_api_line_status[key])
        # 确保总和不超过2
        dict_record_id_api_line_status[key] = sum_value

    return dict_record_id_api_line_status


def visualize_real_road_status(dict_record_id_api_line_status, id_latlon, output_file='traffic_real_status_visualization.html'):
    """可视化实际道路状态"""
    red_list = []
    yellow_list = []
    green_list = []

    for k in dict_record_id_api_line_status.keys():
        if dict_record_id_api_line_status[k] == 0:
            green_list.append(id_latlon[k])
        elif dict_record_id_api_line_status[k] == 2:
            red_list.append(id_latlon[k])
        elif dict_record_id_api_line_status[k] == 1:
            yellow_list.append(id_latlon[k])

    # 计算实际数据的中心点
    all_coords = []
    for road_list in [green_list, red_list, yellow_list]:
        for road in road_list:
            all_coords.extend(road)
    
    if all_coords:
        center_lat = sum(coord[1] for coord in all_coords) / len(all_coords)
        center_lon = sum(coord[0] for coord in all_coords) / len(all_coords)
    else:
        center_lat, center_lon = 30.487398, 114.498611  # 默认武汉坐标

    # 创建地图，先添加基础瓦片图层
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # 绘制绿色道路
    line_color = '#00FF00'
    edges = [[[lon, lat] for lat, lon in edge] for edge in green_list]
    for edge in edges:
        folium.PolyLine(locations=edge, color=line_color, weight=3, opacity=0.7).add_to(m)

    # 绘制红色道路
    line_color = '#FF0000'
    edges = [[[lon, lat] for lat, lon in edge] for edge in red_list]
    for edge in edges:
        folium.PolyLine(locations=edge, color=line_color, weight=3, opacity=0.7).add_to(m)

    # 绘制黄色道路
    line_color = '#FFFF00'
    edges = [[[lon, lat] for lat, lon in edge] for edge in yellow_list]
    for edge in edges:
        folium.PolyLine(locations=edge, color=line_color, weight=3, opacity=0.7).add_to(m)

    # 保存地图
    m.save(output_file)
    print(f"交通状态可视化结果已保存到: {output_file}")











def mcp_match_sumo_edges_to_real_road_status(
    shp_path: str = None,
    output_pkl: str = 'pkl/dict_record_id_APILineStatus_core_500m.pkl',
    egdes_latlon: dict = None,
) -> str:
    """
    匹配SUMO边到实际道路状态，并保存为pkl
    Args:
        shp_path: 交通状态shapefile路径
        id_latlon_path: SUMO网络道路经纬度数据pkl路径
        output_pkl: 匹配结果输出pkl路径
    Returns:
        str: 操作结果信息
    """
    try:
        # if os.path.exists(output_pkl):
        #     return f"成功：已匹配 {len(result)} 条道路，结果保存到 {output_pkl}"
        shp_path='peking_univ_output\\traffic_status_wgs84.shp'
        gdf = gpd.read_file(shp_path)
        id_latlon = egdes_latlon
        result = match_sumo_edges_to_real_road_status(gdf, id_latlon)
        dumpfile(result, output_pkl)
        return f"成功：已匹配 {len(result)} 条道路，结果保存到 {output_pkl}"
    except Exception as e:
        return f"错误：匹配失败 - {str(e)}"




import pyproj
from pyproj import Transformer
from pyproj import Proj, transform
import warnings
warnings.filterwarnings('ignore')
from pyproj import Proj, transform


def convert_sumo_shapes_to_latlon(net_file: str) -> dict:
    """
    将SUMO网络道路位置信息转换为经纬度坐标
    Args:
        net_file: SUMO网络文件路径
    Returns:
        dict: 包含道路ID和经纬度坐标的字典
    """

    # 解析XML文件
    tree = ET.parse(net_file)
    root = tree.getroot()

    # 获取投影参数
    location_tag = root.find("location")
    if location_tag is None:
        raise ValueError("net.xml 中缺少 <location> 标签")

    netOffset_str = location_tag.attrib["netOffset"]
    projParameter = location_tag.attrib["projParameter"]
    netOffsetX, netOffsetY = map(float, netOffset_str.split(","))

    # 初始化投影转换器
    transformer = pyproj.Transformer.from_proj(
        pyproj.Proj(projParameter),
        pyproj.Proj(proj="latlong", datum="WGS84"),
        always_xy=True
    )

    edges = {}
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        if not edge_id:
            continue
            
        # 获取该edge的所有车道
        lanes = edge.findall('lane')
        if not lanes:
            continue
            
        # 选择中间车道作为代表
        middle_lane_index = len(lanes) // 2
        middle_lane = lanes[middle_lane_index]
        
        shape = middle_lane.get('shape')
        if not shape:
            continue
            
        # 转换每个坐标点
        latlon_points = []
        for point in shape.split():
            x, y = map(float, point.split(','))
            
            # 转换坐标
            x_proj = x - netOffsetX
            y_proj = y - netOffsetY
            lon, lat = transformer.transform(x_proj, y_proj)
            latlon_points.append([lon, lat])
            
        edges[edge_id] = latlon_points
    dumpfile(edges, 'pkl/edges_Lat_lon_dict_.pkl')
    # import pdb;pdb.set_trace()
    return edges


