#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTW 时序对齐工具

对多天交通速度数据进行 DTW 重心平均（DBA），然后在对齐后的
标准化日数据中寻找全网最拥堵时刻，提取该时刻所有道路的速度状态向量。

输入:
  --input   多天采集数据根目录（包含 day_YYYY-MM-DD/ 子目录）
            或直接指定单个 .npy 文件（D x N x T 或 D x T x N 格式）

输出目录（--output）包含:
  aligned_pattern.npy     N x T  DTW 对齐后的标准化日速度矩阵
  peak_state.npy          N      全网最拥堵时刻各道路速度状态向量
  global_speed_curve.npy  T      全网平均速度时序曲线
  meta.pkl                       元数据（峰值时间步、道路数、天数等）
  dtw_alignment_plot.png         可视化图表（使用 --plot 时生成）

用法示例:
  python utils/dtw_alignment.py \\
      --input  data/my_region/raw_traffic \\
      --output data/my_region/aligned_pattern

  python utils/dtw_alignment.py \\
      --input  traffic_evolution_5days_top20_speed_3d.npy \\
      --output data/my_region/aligned_pattern --plot
"""

import argparse
import logging
import os
import pickle
import sys
from typing import Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dtw_alignment")


# ---------------------------------------------------------------------------
# 核心算法（与 DTW.py 保持一致）
# ---------------------------------------------------------------------------

def extract_peak_network_state(
    data_dnt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """对 D x N x T 数据做 DTW 重心平均，提取全网最拥堵时刻状态。

    Args:
        data_dnt: shape (D, N, T)
                  D = 天数，N = 路段数，T = 每天时间步数

    Returns:
        aligned_matrix:      shape (N, T)  DTW 对齐后的标准化日矩阵
        global_speed_curve:  shape (T,)    全网平均速度曲线
        peak_time_idx:       int           最拥堵时间步索引
        target_state_vector: shape (N,)    该时刻各道路速度向量
    """
    try:
        from tslearn.barycenters import dtw_barycenter_averaging
    except ImportError:
        logger.error(
            "缺少依赖包 tslearn，请运行: pip install tslearn"
        )
        sys.exit(1)

    import time as _time

    D, N, T = data_dnt.shape
    logger.info(f"输入数据维度: Days={D}, Roads={N}, TimeSteps={T}")

    aligned_matrix = np.zeros((N, T))
    logger.info("正在进行 DTW 重心平均计算 (DBA)，请耐心等待...")
    t0 = _time.time()

    for n in range(N):
        # tslearn 要求输入格式: (n_samples, sz, d)
        series_n = data_dnt[:, n, :].reshape(D, T, 1)
        barycenter = dtw_barycenter_averaging(series_n, max_iter=5, tol=1e-3)
        aligned_matrix[n, :] = barycenter.flatten()

    logger.info(f"DBA 计算完成，耗时: {_time.time() - t0:.2f} 秒")

    # 全网平均速度曲线：找速度最低点（最拥堵时刻）
    global_speed_curve = np.mean(aligned_matrix, axis=0)
    peak_time_idx = int(np.argmin(global_speed_curve))
    peak_speed_val = float(global_speed_curve[peak_time_idx])

    logger.info(
        f"锁定全网最拥堵时刻: TimeStep={peak_time_idx}, "
        f"全网平均速度={peak_speed_val:.2f} km/h"
    )

    target_state_vector = aligned_matrix[:, peak_time_idx]
    return aligned_matrix, global_speed_curve, peak_time_idx, target_state_vector


# ---------------------------------------------------------------------------
# 数据加载工具
# ---------------------------------------------------------------------------

def load_day_snapshots(day_dir: str):
    """从某天的数据目录加载所有快照，返回 N x T 矩阵。

    支持:
      - .pkl 文件（GeoDataFrame，取 speed 列）
      - .npy 文件（直接堆叠）

    Returns:
        np.ndarray of shape (N, T)，或 None（无有效数据时）
    """
    # 按文件名排序保证时间顺序
    all_files = sorted(os.listdir(day_dir))
    pkl_files = [f for f in all_files if f.endswith(".pkl") and "snapshot" in f]
    npy_files = [f for f in all_files if f.endswith(".npy")]

    files = pkl_files if pkl_files else npy_files
    if not files:
        logger.warning(f"目录 {day_dir} 中未找到有效快照文件，跳过")
        return None

    snapshots = []
    for fname in files:
        fpath = os.path.join(day_dir, fname)
        try:
            if fname.endswith(".pkl"):
                with open(fpath, "rb") as f:
                    obj = pickle.load(f)
                if hasattr(obj, "columns") and "speed" in obj.columns:
                    snapshots.append(obj["speed"].values.astype(float))
            else:
                snapshots.append(np.load(fpath).astype(float))
        except Exception as e:
            logger.warning(f"  跳过文件 {fname}: {e}")

    if not snapshots:
        return None

    min_len = min(len(s) for s in snapshots)
    matrix = np.stack([s[:min_len] for s in snapshots], axis=1)  # N x T
    logger.info(
        f"  {os.path.basename(day_dir)}: "
        f"{len(snapshots)} 个快照, "
        f"路段数={matrix.shape[0]}, 时间步={matrix.shape[1]}"
    )
    return matrix


def load_npy_file(input_path: str) -> np.ndarray:
    """加载 .npy 文件，自动识别并转置为 D x N x T 格式。"""
    data = np.load(input_path)
    logger.info(f"从 {input_path} 加载 npy 数据，shape={data.shape}")
    if data.ndim == 3 and data.shape[1] > data.shape[2]:
        # 推测为 D x T x N，转置为 D x N x T
        data = np.transpose(data, (0, 2, 1))
        logger.info(f"检测到 D×T×N 格式，已转置为 D×N×T，shape={data.shape}")
    return data


def load_input(input_path: str) -> np.ndarray:
    """统一加载入口：自动区分 .npy 文件与目录。"""
    if input_path.endswith(".npy"):
        return load_npy_file(input_path)

    if not os.path.isdir(input_path):
        logger.error(f"输入路径不存在或不是目录: {input_path}")
        sys.exit(1)

    day_dirs = sorted([
        os.path.join(input_path, d)
        for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d))
    ])
    if not day_dirs:
        logger.error(f"在 {input_path} 下未找到任何子目录")
        sys.exit(1)

    day_matrices = []
    for dd in day_dirs:
        mat = load_day_snapshots(dd)
        if mat is not None:
            day_matrices.append(mat)

    if not day_matrices:
        logger.error("未能从任何子目录加载有效数据")
        sys.exit(1)

    min_roads = min(m.shape[0] for m in day_matrices)
    min_time  = min(m.shape[1] for m in day_matrices)
    raw_data  = np.stack(
        [m[:min_roads, :min_time] for m in day_matrices], axis=0
    )  # D x N x T
    logger.info(
        f"共加载 {raw_data.shape[0]} 天数据，"
        f"D×N×T = {raw_data.shape}"
    )
    return raw_data


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_results(raw_data, aligned_matrix, global_curve,
                 peak_idx, peak_state, output_dir: str):
    """生成并保存 DTW 对齐结果可视化图表。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("未找到 matplotlib，跳过可视化")
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    # 图1：全网平均速度曲线 + 峰值标注
    axes[0].plot(global_curve, color="steelblue", linewidth=2)
    axes[0].axvline(
        x=peak_idx, color="red", linestyle="--", linewidth=1.5,
        label=f"Peak T={peak_idx}"
    )
    axes[0].scatter(
        peak_idx, global_curve[peak_idx],
        color="red", s=80, zorder=5
    )
    axes[0].set_title("Global Network Speed Curve\n(Step 2: Find Worst Congestion)")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Avg Speed (km/h)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 图2：最拥堵时刻各路段速度
    colors = ["#e74c3c" if s < 30 else "#2ecc71" for s in peak_state]
    axes[1].bar(range(len(peak_state)), peak_state, color=colors)
    axes[1].set_title(
        f"Peak State Vector at T={peak_idx}\n(Output: Input for Calibration)"
    )
    axes[1].set_xlabel("Road ID")
    axes[1].set_ylabel("Speed (km/h)")
    axes[1].grid(True, axis="y", alpha=0.3)

    # 图3：某条路段原始 vs DTW 对齐
    road_id = min(10, raw_data.shape[1] - 1)
    for d in range(raw_data.shape[0]):
        axes[2].plot(
            raw_data[d, road_id, :],
            alpha=0.45, linewidth=1, label=f"Day {d + 1}"
        )
    axes[2].plot(
        aligned_matrix[road_id, :],
        color="black", linewidth=2.5, label="DTW Barycenter"
    )
    axes[2].set_title(
        f"Road {road_id}: Raw vs DTW-Aligned\n(Step 1: DBA)"
    )
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Speed (km/h)")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "dtw_alignment_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"可视化图表已保存: {plot_path}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DTW 时序对齐：提取多天交通数据中的高峰期拥堵模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 getdata.py 采集的目录对齐
  python utils/dtw_alignment.py \\
      --input  data/my_region/raw_traffic \\
      --output data/my_region/aligned_pattern

  # 直接从 .npy 文件读取并生成可视化
  python utils/dtw_alignment.py \\
      --input  traffic_evolution_5days_top20_speed_3d.npy \\
      --output data/my_region/aligned_pattern --plot
        """
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help=(
            "原始数据输入：包含 day_* 子目录的目录路径，"
            "或直接指定 .npy 文件（D×N×T 或 D×T×N 格式）"
        )
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="对齐结果保存目录"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help=(
            "对输入数据添加高斯噪声的幅度比例（0~1，默认 0 即不加噪声），"
            "例如 0.02 表示 2%% 的速度扰动"
        )
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="生成可视化图表并保存为 PNG（需要 matplotlib）"
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. 加载数据 → D x N x T
    raw_data = load_input(args.input)

    # 2. 可选：添加噪声（用于数据增强或鲁棒性测试）
    if args.noise > 0:
        rng = np.random.default_rng(42)
        raw_data = raw_data * (
            1.0 + rng.normal(0.0, args.noise, size=raw_data.shape)
        )
        raw_data = np.clip(raw_data, 0.0, None)
        logger.info(f"已添加 {args.noise * 100:.1f}% 高斯噪声")

    # 3. DTW 对齐 + 寻找最拥堵时刻
    aligned_matrix, global_curve, peak_idx, peak_state = (
        extract_peak_network_state(raw_data)
    )

    # 4. 保存结果
    aligned_path = os.path.join(args.output, "aligned_pattern.npy")
    peak_path    = os.path.join(args.output, "peak_state.npy")
    curve_path   = os.path.join(args.output, "global_speed_curve.npy")
    meta_path    = os.path.join(args.output, "meta.pkl")

    np.save(aligned_path, aligned_matrix)
    np.save(peak_path, peak_state)
    np.save(curve_path, global_curve)

    meta = {
        "peak_time_idx": peak_idx,
        "peak_speed_kmh": float(global_curve[peak_idx]),
        "n_roads":        int(aligned_matrix.shape[0]),
        "n_timesteps":    int(aligned_matrix.shape[1]),
        "n_days":         int(raw_data.shape[0]),
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    logger.info(f"对齐矩阵已保存:         {aligned_path}  shape={aligned_matrix.shape}")
    logger.info(f"最拥堵状态向量已保存:   {peak_path}  shape={peak_state.shape}")
    logger.info(f"全网速度曲线已保存:     {curve_path}")
    logger.info(f"元数据已保存:           {meta_path}")
    logger.info(
        f"最拥堵时刻: TimeStep={peak_idx}, "
        f"全网平均速度={global_curve[peak_idx]:.2f} km/h"
    )

    # 5. 可选可视化
    if args.plot:
        plot_results(
            raw_data, aligned_matrix, global_curve,
            peak_idx, peak_state, args.output
        )


if __name__ == "__main__":
    main()
