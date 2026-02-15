import geopandas as gpd
from generate_od import generator
from shapely.geometry import Polygon
import pyproj


from Carttils import *

def generate_grid_polygons(min_lat, min_lon, max_lat, max_lon, grid_size=500):
    # 定义WGS84地理坐标系
    wgs84 = pyproj.CRS('EPSG:4326')
    # 定义一个适合距离计算的投影坐标系，这里以Web Mercator为例
    web_mercator = pyproj.CRS('EPSG:3857')
    transformer = pyproj.Transformer.from_crs(wgs84, web_mercator, always_xy=True)

    min_x, min_y = transformer.transform(min_lon, min_lat)
    
    
    
    max_x, max_y = transformer.transform(max_lon, max_lat)

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

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=web_mercator)
    # 转换回WGS84坐标系
    gdf = gdf.to_crs(wgs84)
    return gdf

def save_to_shp(gdf, output_path):
    gdf.to_file(output_path)

if __name__ == "__main__":
    # 北京大学中心点坐标
    min_lat = 39.9838  # 39.9928 - 0.009
    max_lat = 40.0018  # 39.9928 + 0.009
    min_lon = 116.2945 # 116.3055 - 0.011
    max_lon = 116.3165 # 116.3055 + 0.011
    grid_size = 500

    # 生成网格
    grid_polygons = generate_grid_polygons(min_lat, min_lon, max_lat, max_lon, grid_size)

    # 初始化生成器并生成OD数据
    my_generator = generator.Generator()
    my_generator.set_satetoken("123")  # for World_Imagery, applied from ArcGIS
    
    # 直接使用生成的grid_polygons作为area
    area = grid_polygons
    my_generator.load_area(area)
    ODfile = my_generator.generate()
    dumpfile(ODfile, 'OD_PKU_2km.pkl')
    
    
    
