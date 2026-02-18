#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTraffic 数据采集入口脚本

负责:
  1. 从 OpenStreetMap 下载指定区域路网并转换为 SUMO .net.xml
  2. 调用高德地图 (Amap) 或 Azure Maps 采集交通拥堵快照
  3. 按天/时间窗口整理输出目录结构，供 DTW 对齐使用

支持两种模式:
  --realtime   单次实时采集（快速测试，不等待时间窗口）
  --days N     多天峰期采集（用于 DTW 对齐，推荐方式）

用法示例:
  # 单次实时采集
  python getdata.py --bbox 39.90 116.30 39.95 116.40 --realtime \\
      --output data/my_region/raw_traffic

  # 多天高峰期采集（5天早高峰，每20分钟一次）
  python getdata.py --bbox 39.90 116.30 39.95 116.40 \\
      --days 5 --time_window "07:00-09:00" --interval 20 \\
      --output data/my_region/raw_traffic
"""

import argparse
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from math import cos, radians
from typing import Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("getdata")


# ---------------------------------------------------------------------------
# API key 加载
# ---------------------------------------------------------------------------

def load_api_keys(config_path: str = "configs/api_keys.yaml") -> dict:
    """从配置文件或环境变量加载 API keys。

    优先级: configs/api_keys.yaml > 环境变量
    """
    keys = {
        "amap":       os.environ.get("AMAP_KEY", ""),
        "azure_maps": os.environ.get("AZURE_MAPS_KEY", ""),
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            if cfg.get("amap", {}).get("api_key"):
                keys["amap"] = cfg["amap"]["api_key"]
            if cfg.get("azure_maps", {}).get("subscription_key"):
                keys["azure_maps"] = cfg["azure_maps"]["subscription_key"]
        except Exception as e:
            logger.warning(f"读取 API key 配置文件失败: {e}")
    return keys


# ---------------------------------------------------------------------------
# 坐标辅助
# ---------------------------------------------------------------------------

def bbox_to_center_and_rect(lat_min: float, lon_min: float,
                             lat_max: float, lon_max: float):
    """将 WGS84 bbox 转换为中心点字符串和高德矩形查询字符串。"""
    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0
    center_point = f"{center_lon:.6f},{center_lat:.6f}"
    rectangle = f"{lon_min:.6f},{lat_min:.6f};{lon_max:.6f},{lat_max:.6f}"
    radius_km = max(
        (lat_max - lat_min) * 111.0,
        (lon_max - lon_min) * 111.0 * cos(radians(center_lat))
    ) / 2.0
    return center_point, rectangle, radius_km


# ---------------------------------------------------------------------------
# 路网准备（OSM → SUMO）
# ---------------------------------------------------------------------------

def setup_network(bbox: Tuple[float, float, float, float],
                  output_dir: str) -> str:
    """下载 OSM 路网并转换为过滤后的 SUMO .net.xml，返回最终文件路径。"""
    from tool.getroadnetwork import (
        download_osm_data,
        convert_to_sumo_net,
        filter_sumo_network,
    )

    net_dir = os.path.join(output_dir, "net")
    os.makedirs(net_dir, exist_ok=True)

    osm_file  = os.path.join(net_dir, "region.osm.xml")
    raw_net   = os.path.join(net_dir, "region_raw.net.xml")
    final_net = os.path.join(net_dir, "region.net.xml")

    if os.path.exists(final_net):
        logger.info(f"路网文件已存在，跳过下载: {final_net}")
        return final_net

    logger.info("正在从 OpenStreetMap 下载路网数据...")
    if not download_osm_data(bbox, osm_file):
        logger.error("OSM 数据下载失败，请检查网络连接或 bbox 参数")
        sys.exit(1)

    logger.info("正在调用 netconvert 转换路网（需要 SUMO_HOME 已设置）...")
    if not convert_to_sumo_net(osm_file, raw_net):
        logger.error("SUMO 路网转换失败，请确认 SUMO_HOME 环境变量已正确设置")
        sys.exit(1)

    filter_sumo_network(raw_net, final_net)
    logger.info(f"路网准备完成: {final_net}")
    return final_net


# ---------------------------------------------------------------------------
# 快照采集：Amap（高德地图）
# ---------------------------------------------------------------------------

def collect_amap_snapshot(rectangle: str, api_key: str,
                           output_dir: str, snapshot_name: str) -> bool:
    """调用高德地图交通态势 API 采集单次快照，保存为 .pkl（GeoDataFrame）。"""
    from tool.getroadnetwork import get_traffic_data

    gdf = get_traffic_data(rectangle, api_key)
    if gdf is None or len(gdf) == 0:
        logger.warning("Amap：未获取到交通数据")
        return False

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{snapshot_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(gdf, f)
    logger.info(f"Amap 快照已保存: {out_path}（{len(gdf)} 条道路）")
    return True


# ---------------------------------------------------------------------------
# 快照采集：Azure Maps
# ---------------------------------------------------------------------------

def collect_azure_snapshot(lat_min: float, lon_min: float,
                            lat_max: float, lon_max: float,
                            api_key: str, output_dir: str,
                            snapshot_name: str) -> bool:
    """调用 Azure Maps Traffic Flow API 采集单次快照，保存为 .pkl。

    使用 Traffic Flow Tile / Segment API，覆盖整个 bbox。
    """
    import requests

    try:
        # Traffic Flow Segment API：查询 bbox 中心点的流量信息
        center_lat = (lat_min + lat_max) / 2.0
        center_lon = (lon_min + lon_max) / 2.0
        url = (
            f"https://atlas.microsoft.com/traffic/flow/segment/json"
            f"?api-version=1.0&style=absolute&zoom=12"
            f"&query={center_lat},{center_lon}"
            f"&subscription-key={api_key}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        snapshot = {
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bbox":            [lat_min, lon_min, lat_max, lon_max],
            "flowSegmentData": data.get("flowSegmentData", {}),
        }

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{snapshot_name}_azure.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(snapshot, f)
        logger.info(f"Azure Maps 快照已保存: {out_path}")
        return True

    except Exception as e:
        logger.warning(f"Azure Maps 采集失败: {e}")
        return False


# ---------------------------------------------------------------------------
# 单天采集循环
# ---------------------------------------------------------------------------

def collect_one_day(rectangle: str,
                    bbox: Tuple[float, float, float, float],
                    api_keys: dict,
                    provider: str,
                    time_window: str,
                    interval_min: int,
                    day_output_dir: str) -> int:
    """在指定时间窗口内，每隔 interval_min 分钟采集一次快照。

    Returns:
        int: 本日实际采集次数
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    start_str, end_str = time_window.split("-")

    now = datetime.now()
    start_time = now.replace(
        hour=int(start_str.split(":")[0]),
        minute=int(start_str.split(":")[1]),
        second=0, microsecond=0
    )
    end_time = now.replace(
        hour=int(end_str.split(":")[0]),
        minute=int(end_str.split(":")[1]),
        second=0, microsecond=0
    )

    if datetime.now() > end_time:
        logger.warning(f"当前时间已超过时间窗口 {time_window}，本日跳过采集")
        return 0

    wait_secs = (start_time - datetime.now()).total_seconds()
    if wait_secs > 0:
        logger.info(
            f"等待时间窗口开始 ({start_str})，"
            f"还需 {wait_secs / 60:.1f} 分钟..."
        )
        time.sleep(wait_secs)

    os.makedirs(day_output_dir, exist_ok=True)
    sample_idx = 0

    while datetime.now() <= end_time:
        ts   = datetime.now().strftime("%H%M%S")
        name = f"snapshot_{sample_idx:03d}_{ts}"
        logger.info(
            f"采集第 {sample_idx + 1} 次快照 "
            f"（{datetime.now().strftime('%H:%M:%S')}）"
        )

        if provider in ("amap", "both"):
            collect_amap_snapshot(
                rectangle, api_keys["amap"], day_output_dir, name
            )
        if provider in ("azure", "both"):
            collect_azure_snapshot(
                lat_min, lon_min, lat_max, lon_max,
                api_keys["azure_maps"], day_output_dir, name
            )

        sample_idx += 1
        next_sample = datetime.now() + timedelta(minutes=interval_min)
        if next_sample > end_time:
            break
        sleep_secs = (next_sample - datetime.now()).total_seconds()
        if sleep_secs > 0:
            time.sleep(sleep_secs)

    logger.info(
        f"本日采集完成，共 {sample_idx} 次快照，保存至: {day_output_dir}"
    )
    return sample_idx


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RealTraffic 数据采集工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单次实时采集（快速测试）
  python getdata.py --bbox 39.90 116.30 39.95 116.40 --realtime \\
      --output data/my_region/raw_traffic

  # 多天高峰期采集（推荐，用于 DTW 对齐）
  python getdata.py --bbox 39.90 116.30 39.95 116.40 \\
      --days 5 --time_window "07:00-09:00" --interval 20 \\
      --output data/my_region/raw_traffic

  # 使用 Azure Maps
  python getdata.py --bbox 39.90 116.30 39.95 116.40 --realtime \\
      --provider azure --output data/my_region/raw_traffic
        """
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
        required=True,
        help="地理边界框（WGS84 格式）"
    )
    parser.add_argument(
        "--output", type=str, default="data/region/raw_traffic",
        help="输出根目录（默认: data/region/raw_traffic）"
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="单次实时采集模式（立即采集，不等待时间窗口）"
    )
    parser.add_argument(
        "--days", type=int, default=1,
        help="连续采集天数（默认: 1）"
    )
    parser.add_argument(
        "--time_window", type=str, default="07:00-09:00",
        help="采集时间窗口，格式 HH:MM-HH:MM（默认: 07:00-09:00）"
    )
    parser.add_argument(
        "--interval", type=int, default=20,
        help="快照采样间隔（分钟，默认: 20）"
    )
    parser.add_argument(
        "--provider", type=str,
        choices=["amap", "azure", "both"],
        default="amap",
        help="数据来源（amap / azure / both，默认: amap）"
    )
    parser.add_argument(
        "--skip-network", action="store_true",
        help="跳过路网下载步骤（已有路网时使用）"
    )
    parser.add_argument(
        "--api-config", type=str, default="configs/api_keys.yaml",
        help="API 密钥配置文件路径（默认: configs/api_keys.yaml）"
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    lat_min, lon_min, lat_max, lon_max = args.bbox
    bbox = (lat_min, lon_min, lat_max, lon_max)

    # 加载 API keys
    api_keys = load_api_keys(args.api_config)
    if args.provider in ("amap", "both") and not api_keys["amap"]:
        logger.error(
            "未找到 Amap API key，请在 configs/api_keys.yaml 中设置 "
            "amap.api_key 或导出环境变量 AMAP_KEY"
        )
        sys.exit(1)
    if args.provider in ("azure", "both") and not api_keys["azure_maps"]:
        logger.error(
            "未找到 Azure Maps key，请在 configs/api_keys.yaml 中设置 "
            "azure_maps.subscription_key 或导出环境变量 AZURE_MAPS_KEY"
        )
        sys.exit(1)

    center_point, rectangle, radius_km = bbox_to_center_and_rect(
        lat_min, lon_min, lat_max, lon_max
    )
    logger.info(
        f"采集区域: bbox={bbox}, 中心={center_point}, "
        f"半径≈{radius_km:.2f} km，数据来源={args.provider}"
    )

    # 准备路网（仅一次）
    if not args.skip_network:
        net_file = setup_network(bbox, args.output)
        logger.info(f"SUMO 路网: {net_file}")

    # ------------------------------------------------------------------
    # 实时单次模式
    # ------------------------------------------------------------------
    if args.realtime:
        day_dir = os.path.join(
            args.output,
            f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(day_dir, exist_ok=True)
        ts   = datetime.now().strftime("%H%M%S")
        name = f"snapshot_000_{ts}"

        logger.info("实时模式：采集单次快照...")
        if args.provider in ("amap", "both"):
            collect_amap_snapshot(rectangle, api_keys["amap"], day_dir, name)
        if args.provider in ("azure", "both"):
            collect_azure_snapshot(
                lat_min, lon_min, lat_max, lon_max,
                api_keys["azure_maps"], day_dir, name
            )
        logger.info(f"实时快照已保存至: {day_dir}")
        return

    # ------------------------------------------------------------------
    # 多天采集模式
    # ------------------------------------------------------------------
    logger.info(
        f"多天采集模式：{args.days} 天，"
        f"时间窗口 {args.time_window}，间隔 {args.interval} 分钟"
    )

    for day_idx in range(args.days):
        today_str = (
            datetime.now() + timedelta(days=day_idx)
        ).strftime("%Y-%m-%d")
        day_dir = os.path.join(args.output, f"day_{today_str}")

        logger.info(
            f"=== 第 {day_idx + 1}/{args.days} 天（{today_str}）==="
        )

        # 从第二天起，等待到下一天时间窗口开始
        if day_idx > 0:
            start_str = args.time_window.split("-")[0]
            next_day_start = (datetime.now() + timedelta(days=1)).replace(
                hour=int(start_str.split(":")[0]),
                minute=int(start_str.split(":")[1]),
                second=0, microsecond=0
            )
            wait_secs = (next_day_start - datetime.now()).total_seconds()
            if wait_secs > 0:
                logger.info(
                    f"等待第 {day_idx + 1} 天时间窗口，"
                    f"还需 {wait_secs / 3600:.1f} 小时..."
                )
                time.sleep(wait_secs)

        collect_one_day(
            rectangle, bbox, api_keys, args.provider,
            args.time_window, args.interval, day_dir
        )

    logger.info(
        f"全部 {args.days} 天采集完成，数据保存于: {args.output}"
    )


if __name__ == "__main__":
    main()
