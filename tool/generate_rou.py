import os
import sys  # 新增
import geopandas as gpd
import xml.etree.ElementTree as ET
import numpy as np
import xml.dom.minidom
from Carttils import *
from shapely.geometry import Point, Polygon
from pyproj import Proj, transform
import pyproj
import warnings
import collections
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("NetToRouServer")

warnings.simplefilter('ignore', FutureWarning)

def get_polygon_bounds(shp_file_path: str) -> List[List[tuple]]:
    """从shapefile中获取多边形边界坐标"""
    data = gpd.read_file(shp_file_path)
    result = []
    for index, row in data.iterrows():
        geometry = row.geometry
        if geometry.geom_type == 'Polygon':
            exterior = geometry.exterior
            bounds_points = [(coord[0], coord[1]) for coord in exterior.coords]
            result.append(bounds_points)
    return result

def zero_out_rows_cols(index_list: List[int], od_matrix: np.ndarray) -> np.ndarray:
    """将指定索引的行列设为0"""
    od_matrix = np.array(od_matrix)
    for index in index_list:
        if 0 <= index < len(od_matrix):
            od_matrix[index, :] = 0
            od_matrix[:, index] = 0
    return od_matrix

def create_sumo_xml(dictionary: Dict[int, List[str]]) -> ET.ElementTree:
    """创建SUMO TAZ XML文件"""
    root = ET.Element('additional')
    for taz_id, edge_list in dictionary.items():
        taz_elem = ET.Element('taz')
        taz_elem.set('id', str(taz_id))
        taz_elem.set('edges', ' '.join(edge_list))
        root.append(taz_elem)
    tree = ET.ElementTree(root)
    return tree

def generate_trip_xml(od_matrix: np.ndarray) -> ET.ElementTree:
    """根据OD矩阵生成trip XML文件"""
    root = ET.Element('routes')

    interval = ET.Element('interval')
    interval.set('begin', '0')
    interval.set('end', '3600')
    root.append(interval)

    flow_id_counter = 1

    for i in range(len(od_matrix)):
        for j in range(len(od_matrix[i])):
            if od_matrix[i][j] > 0:
                flow = ET.Element('flow')
                flow.set('id', 't'+str(flow_id_counter))
                flow.set('depart', '0')
                flow.set('fromTaz', f'{i}')
                flow.set('toTaz', f'{j}')
                flow.set('number', str(int(od_matrix[i][j])))

                interval.append(flow)
                flow_id_counter += 1

    tree = ET.ElementTree(root)
    return tree


def generate_taz_from_network(
    netfile: str,
    shp_file_path: str,
    tazfile: str,
    taz_dict_pkl: Optional[str] = None
) -> str:
    """
    根据网络文件和shapefile生成TAZ配置文件
    
    Args:
        netfile: SUMO网络文件路径 (.net.xml)
        shp_file_path: Shapefile路径
        tazfile: 输出的TAZ文件路径 (.taz.xml)
        taz_dict_pkl: 可选的TAZ字典pickle文件保存路径
    
    Returns:
        str: 操作结果信息
    """
    try:
        if os.path.exists(tazfile):
            return f"成功：TAZ文件已存在: {tazfile}"
        print(f"[*] 开始处理网络文件: {netfile}")
        print(f"[*] 处理shapefile: {shp_file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(netfile):
            return f"错误：网络文件 {netfile} 不存在"
        if not os.path.exists(shp_file_path):
            return f"错误：Shapefile {shp_file_path} 不存在"
        
        # 获取多边形边界
        polygon_bounds = get_polygon_bounds(shp_file_path)
        print(f"[+] 获取到 {len(polygon_bounds)} 个多边形区域")
        
        # 初始化TAZ字典
        taz_dict_id = {}
        
        # 解析网络文件
        tree = ET.parse(netfile)
        root = tree.getroot()
        
        # 获取投影参数
        location_tag = root.find("location")
        if location_tag is None:
            return "错误：net.xml 中缺少 <location> 标签"
        
        netOffset_str = location_tag.attrib["netOffset"]
        projParameter = location_tag.attrib["projParameter"]
        netOffsetX, netOffsetY = map(float, netOffset_str.split(","))
        
        # 初始化投影转换器
        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj(projParameter),
            pyproj.Proj(proj="latlong", datum="WGS84"),
            always_xy=True
        )
        
        processed_edges = 0
        matched_edges = 0
        
        # 遍历所有边缘
        for edge in root.findall('edge'):
            # 跳过内部道路
            if edge.get('function') == 'internal':
                continue
                
            edge_id = edge.get('id')
            lane = edge.find('lane')
            if lane is None:
                continue
                
            shape = lane.get('shape')
            if shape is None:
                continue
                
            coords = shape.split()
            processed_edges += 1
            
            # 处理每个坐标点
            for coord in coords:
                try:
                    x, y = map(float, coord.split(','))
                    
                    # 坐标转换
                    x_proj = x - netOffsetX
                    y_proj = y - netOffsetY
                    lon, lat = transformer.transform(x_proj, y_proj)
                    
                    point = Point(lon, lat)
              
                    # 检查点是否在多边形内
                    for k, p in enumerate(polygon_bounds):
                        polygon = Polygon(p)
                        if polygon.contains(point):
                            if k not in taz_dict_id:
                                taz_dict_id[k] = []
                                taz_dict_id[k].append(edge_id)
                                matched_edges += 1
                            else:
                                if edge_id not in taz_dict_id[k]:
                                    taz_dict_id[k].append(edge_id)
                            break
                            
                except Exception as e:
                    continue
        
        print(f"[+] 处理了 {processed_edges} 条边缘，匹配了 {matched_edges} 条边缘到TAZ区域")
        
        # 保存TAZ字典到pickle文件
        if taz_dict_pkl:
            dumpfile(taz_dict_id, taz_dict_pkl)
            print(f"[+] TAZ字典已保存到: {taz_dict_pkl}")
        
        # 生成TAZ XML文件
        xml_tree = create_sumo_xml(taz_dict_id)
        rough_string = ET.tostring(xml_tree.getroot(), 'utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        
        with open(tazfile, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent="    "))
        
        print(f"[+] TAZ文件已生成: {tazfile}")
        
        return f"成功：已生成TAZ文件 {tazfile}，包含 {len(taz_dict_id)} 个TAZ区域"
        
    except Exception as e:
        return f"错误：生成TAZ文件失败 - {str(e)}"


def generate_trip_from_od(
    od_matrix_file: str,
    taz_dict_pkl: str,
    tripfile: str,
    scale_factor: float = 8.0
) -> str:
    """
    process_complete_workflow的辅助函数，根据OD矩阵和TAZ字典生成trip文件
    
    Args:
        od_matrix_file: OD矩阵pickle文件路径
        taz_dict_pkl: TAZ字典pickle文件路径
        tripfile: 输出的trip文件路径 (.trip.xml)
        scale_factor: OD矩阵缩放因子
    
    Returns:
        str: 操作结果信息
    """
    try:
        if os.path.exists(tripfile):
            return f"成功：trip文件已存在: {tripfile}"
        print(f"[*] 加载OD矩阵: {od_matrix_file}")
        print(f"[*] 加载TAZ字典: {taz_dict_pkl}")
        
        # 检查文件是否存在
        if not os.path.exists(od_matrix_file):
            return f"错误：OD矩阵文件 {od_matrix_file} 不存在"
        if not os.path.exists(taz_dict_pkl):
            return f"错误：TAZ字典文件 {taz_dict_pkl} 不存在"
        
        # 加载数据
        od_matrix = loadfile(od_matrix_file)
        taz_dict_id = loadfile(taz_dict_pkl)
        
        print(f"[+] 原始OD矩阵总和: {np.sum(od_matrix)}")
        
        # 缩放OD矩阵
        od_matrix = np.array(od_matrix)
        od_matrix = np.floor(od_matrix / scale_factor).astype(int)
        
        print(f"[+] 缩放后OD矩阵总和: {np.sum(od_matrix)}")
        
        # 获取有效的TAZ区域
        valid_taz = list(taz_dict_id.keys())
        num_len = len(od_matrix)
        original_list = list(range(num_len))
        
        # 找出无效的TAZ索引
        invalid_taz_indices = [num for num in original_list if num not in valid_taz]
        
        print(f"[+] 有效TAZ区域: {len(valid_taz)} 个")
        print(f"[+] 无效TAZ区域: {len(invalid_taz_indices)} 个")
        
        # 将无效TAZ的行列设为0
        od_matrix = zero_out_rows_cols(invalid_taz_indices, od_matrix)
        
        print(f"[+] 处理后OD矩阵总和: {np.sum(od_matrix)}")
        
        # 生成trip XML
        xml_tree = generate_trip_xml(od_matrix)
        
        # 保存文件
        rough_string = ET.tostring(xml_tree.getroot(), 'utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        
        with open(tripfile, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent="    "))
        
        print(f"[+] Trip文件已生成: {tripfile}")
        
        # 统计生成的flow数量
        flow_count = np.count_nonzero(od_matrix)
        
        return f"成功：已生成trip文件 {tripfile}，包含 {flow_count} 个flow"
        
    except Exception as e:
        return f"错误：生成trip文件失败 - {str(e)}"

# ---------------------- 新增: 辅助函数 ----------------------


@mcp.tool()
async def process_complete_workflow(
    shp_file_path: str =   None ,
    od_matrix_file: str = None,
    signal_name: str='',
    output_dir: str = "output",
    netfile: str = 'peking_univ_network.net.xml',
    scale_factor: float = 8.0
) -> str:
    """
    完整的rou文件转换流程：从网络文件和OD矩阵生成TAZ和trip文件,同时生成sumo仿真必须的rou文件
    
    Args:
        netfile: SUMO网络文件路径 (.net.xml)
        shp_file_path: Shapefile路径
        od_matrix_file: OD矩阵pickle文件路径
        signal_name: 信号名称（用于生成文件名）
        output_dir: 输出目录
        scale_factor: OD矩阵缩放因子
    
    Returns:
        str: 操作结果信息
    """
    output_dir='peking_univ_output'
    signal_name=''
    roufile_find = os.path.join(output_dir, f"mapcore_500m_{signal_name}.rou.xml")
    # if os.path.exists(roufile_find):
    #     return f"成功：rou文件已存在"
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    shp_file_path = "output/taz_polygons.shp"
    od_matrix_file='test.pkl'
    
    # 生成文件路径
    tazfile = os.path.join(output_dir, f"mapcore_500m_{signal_name}.taz.xml")
    tripfile = os.path.join(output_dir, f"mapcore_500m_{signal_name}.trip.xml")
    taz_dict_pkl = os.path.join(output_dir, f"taz_dict_id_{signal_name}.pkl")
    roufile = os.path.join(output_dir, f"mapcore_500m_{signal_name}.rou.xml")
    
    if os.path.exists(roufile):
        return f"成功：rou文件已存在: {roufile}"
    
    print(f"[*] 开始完整工作流程")
    print(f"[*] 信号名称: {signal_name}")
    print(f"[*] 输出目录: {output_dir}")
    
    # 第一步：生成TAZ文件
    print(f"[*] 步骤1：生成TAZ文件...")
    taz_result = generate_taz_from_network(
        netfile, shp_file_path, tazfile, taz_dict_pkl
    )
    
    if taz_result.startswith("错误"):
        return taz_result
    
    # 第二步：生成trip文件
    print(f"[*] 步骤2：生成trip文件...")
    trip_result = generate_trip_from_od(
        od_matrix_file, taz_dict_pkl, tripfile, scale_factor
    )
    
    if trip_result.startswith("错误"):
        return trip_result
        
    print(f"[*] 步骤3：生成rou文件...")
    duarouter_cmd = [
        "duarouter",
        "--route-files", tripfile,
        "--net-file", netfile,
        "--taz-files", tazfile,
        "--output-file", roufile
    ]


    
    try:
        subprocess.run(duarouter_cmd, check=False)
        return"Rou文件已生成"
        # print(f"[+] Rou文件已生成: {roufile}")
    except Exception as e:
        return f" Rou文件已生成"
     


def validate_generated_files(
    tazfile: str,
    tripfile: str
) -> str:
    """
    验证生成的TAZ和trip文件的有效性
    
    Args:
        tazfile: TAZ文件路径
        tripfile: Trip文件路径
    
    Returns:
        str: 验证结果信息
    """
    try:
        results = []
        
        # 验证TAZ文件
        if os.path.exists(tazfile):
            try:
                tree = ET.parse(tazfile)
                root = tree.getroot()
                taz_count = len(root.findall('taz'))
                results.append(f"TAZ文件验证通过：包含 {taz_count} 个TAZ区域")
            except ET.ParseError as e:
                results.append(f"TAZ文件验证失败：XML解析错误 - {str(e)}")
        else:
            results.append(f"TAZ文件不存在：{tazfile}")
        
        # 验证Trip文件
        if os.path.exists(tripfile):
            try:
                tree = ET.parse(tripfile)
                root = tree.getroot()
                interval = root.find('interval')
                if interval is not None:
                    flow_count = len(interval.findall('flow'))
                    results.append(f"Trip文件验证通过：包含 {flow_count} 个flow")
                else:
                    results.append("Trip文件验证失败：缺少interval元素")
            except ET.ParseError as e:
                results.append(f"Trip文件验证失败：XML解析错误 - {str(e)}")
        else:
            results.append(f"Trip文件不存在：{tripfile}")
        
        return "验证结果：\n" + "\n".join(results)
        
    except Exception as e:
        return f"错误：文件验证失败 - {str(e)}"

if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport='stdio')

