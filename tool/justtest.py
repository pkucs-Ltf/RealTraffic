

osm_file='peking_university_map.osm'
output_net_file='peking_univ_network.net.xml'


import os
import json
import requests
import subprocess
from typing import Any, Tuple, List
from mcp.server.fastmcp import FastMCP
import geopandas as gpd

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
from Carttils import is_short_line_almost_on_long_line, transfer_status_to_num, find_max_value

from get_edgelimit import extract_speed_limits_from_network

from get_road_to_status import mcp_match_sumo_edges_to_real_road_status,convert_sumo_shapes_to_latlon

from getroadnetwork import *

from generate_map_html import generate_map_with_marker

get_real_traffic_state(center_point,radius_km,output_dir,API_KEY_GAODE)

generate_grid_polygons(bbox[0], bbox[1], bbox[2], bbox[3])