#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTraffic data collection entry script.

Responsibilities:
  1. Download the road network for a given region from OpenStreetMap and convert it to a SUMO .net.xml file.
  2. Collect traffic congestion snapshots via Amap (AutoNavi) or Azure Maps.
  3. Organise output directories by day/time window for subsequent DTW alignment.

Two modes are supported:
  --realtime   Single real-time snapshot (quick test; no time-window waiting).
  --days N     Multi-day peak-period collection (recommended for DTW alignment).

Usage examples:
  # Single real-time snapshot
  python getdata.py --bbox 39.90 116.30 39.95 116.40 --realtime \
      --output data/my_region/raw_traffic

  # Multi-day peak-period collection (5 days, morning peak, every 20 minutes)
  python getdata.py --bbox 39.90 116.30 39.95 116.40 \
      --days 5 --time_window "07:00-09:00" --interval 20 \
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
# API key loading
# ---------------------------------------------------------------------------

def load_api_keys(config_path: str = "configs/api_keys.yaml") -> dict:
    """Load API keys from a config file or environment variables.

    Priority: configs/api_keys.yaml > environment variables.
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
            logger.warning(f"Failed to read API key config file: {e}")
    return keys


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def bbox_to_center_and_rect(lat_min: float, lon_min: float,
                             lat_max: float, lon_max: float):
    """Convert a WGS84 bounding box to a centre-point string and an Amap rectangle query string."""
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
# Network preparation (OSM → SUMO)
# ---------------------------------------------------------------------------

def setup_network(bbox: Tuple[float, float, float, float],
                  output_dir: str) -> str:
    """Download the OSM road network and convert it to a filtered SUMO .net.xml file.

    Returns the path to the final network file.
    """
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
        logger.info(f"Network file already exists, skipping download: {final_net}")
        return final_net

    logger.info("Downloading road network from OpenStreetMap...")
    if not download_osm_data(bbox, osm_file):
        logger.error("OSM data download failed. Check your network connection or bbox parameters.")
        sys.exit(1)

    logger.info("Running netconvert to convert network (SUMO_HOME must be set)...")
    if not convert_to_sumo_net(osm_file, raw_net):
        logger.error("SUMO network conversion failed. Ensure SUMO_HOME is correctly configured.")
        sys.exit(1)

    filter_sumo_network(raw_net, final_net)
    logger.info(f"Network ready: {final_net}")
    return final_net


# ---------------------------------------------------------------------------
# Snapshot collection: Amap (AutoNavi)
# ---------------------------------------------------------------------------

def collect_amap_snapshot(rectangle: str, api_key: str,
                           output_dir: str, snapshot_name: str) -> bool:
    """Call the Amap Traffic Status API to collect a single snapshot and save it as a .pkl (GeoDataFrame)."""
    from tool.getroadnetwork import get_traffic_data

    gdf = get_traffic_data(rectangle, api_key)
    if gdf is None or len(gdf) == 0:
        logger.warning("Amap: no traffic data received.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{snapshot_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(gdf, f)
    logger.info(f"Amap snapshot saved: {out_path} ({len(gdf)} road segments)")
    return True


# ---------------------------------------------------------------------------
# Snapshot collection: Azure Maps
# ---------------------------------------------------------------------------

def collect_azure_snapshot(lat_min: float, lon_min: float,
                            lat_max: float, lon_max: float,
                            api_key: str, output_dir: str,
                            snapshot_name: str) -> bool:
    """Call the Azure Maps Traffic Flow API to collect a single snapshot and save it as a .pkl.

    Uses the Traffic Flow Segment API centred on the bbox midpoint.
    """
    import requests

    try:
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
        logger.info(f"Azure Maps snapshot saved: {out_path}")
        return True

    except Exception as e:
        logger.warning(f"Azure Maps collection failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Single-day collection loop
# ---------------------------------------------------------------------------

def collect_one_day(rectangle: str,
                    bbox: Tuple[float, float, float, float],
                    api_keys: dict,
                    provider: str,
                    time_window: str,
                    interval_min: int,
                    day_output_dir: str) -> int:
    """Collect snapshots at regular intervals within a specified time window.

    Returns:
        int: Number of snapshots actually collected for this day.
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
        logger.warning(f"Current time has passed the collection window {time_window}; skipping today.")
        return 0

    wait_secs = (start_time - datetime.now()).total_seconds()
    if wait_secs > 0:
        logger.info(
            f"Waiting for collection window to open ({start_str}); "
            f"{wait_secs / 60:.1f} minutes remaining..."
        )
        time.sleep(wait_secs)

    os.makedirs(day_output_dir, exist_ok=True)
    sample_idx = 0

    while datetime.now() <= end_time:
        ts   = datetime.now().strftime("%H%M%S")
        name = f"snapshot_{sample_idx:03d}_{ts}"
        logger.info(
            f"Collecting snapshot {sample_idx + 1} "
            f"({datetime.now().strftime('%H:%M:%S')})"
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
        f"Day collection complete: {sample_idx} snapshots saved to: {day_output_dir}"
    )
    return sample_idx


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RealTraffic data collection tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single real-time snapshot (quick test)
  python getdata.py --bbox 39.90 116.30 39.95 116.40 --realtime \
      --output data/my_region/raw_traffic

  # Multi-day peak-period collection (recommended for DTW alignment)
  python getdata.py --bbox 39.90 116.30 39.95 116.40 \
      --days 5 --time_window "07:00-09:00" --interval 20 \
      --output data/my_region/raw_traffic

  # Use Azure Maps as data provider
  python getdata.py --bbox 39.90 116.30 39.95 116.40 --realtime \
      --provider azure --output data/my_region/raw_traffic
        """
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
        required=True,
        help="Geographic bounding box in WGS84 format"
    )
    parser.add_argument(
        "--output", type=str, default="data/region/raw_traffic",
        help="Root output directory (default: data/region/raw_traffic)"
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="Real-time single-snapshot mode (collect immediately, no time-window waiting)"
    )
    parser.add_argument(
        "--days", type=int, default=1,
        help="Number of consecutive days to collect (default: 1)"
    )
    parser.add_argument(
        "--time_window", type=str, default="07:00-09:00",
        help="Collection time window, format HH:MM-HH:MM (default: 07:00-09:00)"
    )
    parser.add_argument(
        "--interval", type=int, default=20,
        help="Snapshot sampling interval in minutes (default: 20)"
    )
    parser.add_argument(
        "--provider", type=str,
        choices=["amap", "azure", "both"],
        default="amap",
        help="Data provider: amap / azure / both (default: amap)"
    )
    parser.add_argument(
        "--skip-network", action="store_true",
        help="Skip network download step (use when network file already exists)"
    )
    parser.add_argument(
        "--api-config", type=str, default="configs/api_keys.yaml",
        help="Path to API key config file (default: configs/api_keys.yaml)"
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    lat_min, lon_min, lat_max, lon_max = args.bbox
    bbox = (lat_min, lon_min, lat_max, lon_max)

    # Load API keys
    api_keys = load_api_keys(args.api_config)
    if args.provider in ("amap", "both") and not api_keys["amap"]:
        logger.error(
            "Amap API key not found. Set amap.api_key in configs/api_keys.yaml "
            "or export the AMAP_KEY environment variable."
        )
        sys.exit(1)
    if args.provider in ("azure", "both") and not api_keys["azure_maps"]:
        logger.error(
            "Azure Maps key not found. Set azure_maps.subscription_key in configs/api_keys.yaml "
            "or export the AZURE_MAPS_KEY environment variable."
        )
        sys.exit(1)

    center_point, rectangle, radius_km = bbox_to_center_and_rect(
        lat_min, lon_min, lat_max, lon_max
    )
    logger.info(
        f"Collection region: bbox={bbox}, centre={center_point}, "
        f"radius≈{radius_km:.2f} km, provider={args.provider}"
    )

    # Prepare network (once only)
    if not args.skip_network:
        net_file = setup_network(bbox, args.output)
        logger.info(f"SUMO network: {net_file}")

    # ------------------------------------------------------------------
    # Real-time single-snapshot mode
    # ------------------------------------------------------------------
    if args.realtime:
        day_dir = os.path.join(
            args.output,
            f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(day_dir, exist_ok=True)
        ts   = datetime.now().strftime("%H%M%S")
        name = f"snapshot_000_{ts}"

        logger.info("Real-time mode: collecting a single snapshot...")
        if args.provider in ("amap", "both"):
            collect_amap_snapshot(rectangle, api_keys["amap"], day_dir, name)
        if args.provider in ("azure", "both"):
            collect_azure_snapshot(
                lat_min, lon_min, lat_max, lon_max,
                api_keys["azure_maps"], day_dir, name
            )
        logger.info(f"Real-time snapshot saved to: {day_dir}")
        return

    # ------------------------------------------------------------------
    # Multi-day collection mode
    # ------------------------------------------------------------------
    logger.info(
        f"Multi-day collection mode: {args.days} day(s), "
        f"window {args.time_window}, interval {args.interval} min"
    )

    for day_idx in range(args.days):
        today_str = (
            datetime.now() + timedelta(days=day_idx)
        ).strftime("%Y-%m-%d")
        day_dir = os.path.join(args.output, f"day_{today_str}")

        logger.info(
            f"=== Day {day_idx + 1}/{args.days} ({today_str}) ==="
        )

        # From the second day onward, wait until the next day's window opens
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
                    f"Waiting for day {day_idx + 1} window to open; "
                    f"{wait_secs / 3600:.1f} hours remaining..."
                )
                time.sleep(wait_secs)

        collect_one_day(
            rectangle, bbox, api_keys, args.provider,
            args.time_window, args.interval, day_dir
        )

    logger.info(
        f"All {args.days} day(s) of collection complete. Data saved to: {args.output}"
    )


if __name__ == "__main__":
    main()
