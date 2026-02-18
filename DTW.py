import numpy as np
import matplotlib.pyplot as plt
from tslearn.barycenters import dtw_barycenter_averaging
import time

# ==========================================
# 1. 数据模拟生成 (Mock Data Generation)
# ==========================================
# ==========================================
# 2. 核心处理流程 (Core Processing)
# ==========================================
def extract_peak_network_state(data_dnt):
    """
    输入: D x N x T 矩阵
    输出: N 维向量 (全网最堵时刻的状态)
    """
    D, N, T = data_dnt.shape
    print(f"输入数据维度: Days={D}, Roads={N}, TimeSteps={T}")
    
    # Step 1: 降维 (Collapse Days) - 使用 DTW DBA
    # 输出将是 N x T (标准的一天)
    aligned_matrix = np.zeros((N, T))
    
    print("正在进行 DTW 重心平均计算 (DBA)...")
    start_time = time.time()
    
    for n in range(N):
        # 提取第 n 条路的所有天数数据 (D x T)
        # tslearn 需要输入格式为 (n_samples, sz, d)，所以我们要 reshape
        series_n = data_dnt[:, n, :].reshape(D, T, 1)
        
        # 计算 DBA 重心 (Barycenter)
        # max_iter 控制迭代次数，通常 5-10 次收敛
        barycenter = dtw_barycenter_averaging(series_n, max_iter=5, tol=1e-3)
        
        # 结果展平存入矩阵
        aligned_matrix[n, :] = barycenter.flatten()
        
    print(f"DBA 计算完成，耗时: {time.time() - start_time:.2f} 秒")
    
    # Step 2: 寻找全网最拥堵时刻 (Find Global Peak)
    # 计算每一个时间步的全网平均速度
    global_speed_curve = np.mean(aligned_matrix, axis=0)
    
    # 找到平均速度最低的时间点 (最堵时刻)
    peak_time_idx = np.argmin(global_speed_curve)
    peak_speed_val = global_speed_curve[peak_time_idx]
    
    print(f"锁定全网最拥堵时刻: Time Step {peak_time_idx}, 平均车速 {peak_speed_val:.2f} km/h")
    
    # Step 3: 切片提取状态 (Slicing)
    target_state_vector = aligned_matrix[:, peak_time_idx]
    
    return aligned_matrix, global_speed_curve, peak_time_idx, target_state_vector

# ==========================================
# 3. 运行与可视化 (Execution & Plotting)
# ==========================================

# A. 生成数据 
# 5天, 20条路, 每天60个时间点
# raw_data = generate_mock_data(D=5, N=20, T=60) 
data=np.load('traffic_evolution_5days_top20_speed_3d.npy')
print(data.shape)

# 对所有数据添加随机扰动：2% 高斯噪声（相对幅度）
# 例如 speed=20km/h 时，噪声标准差约为 1km/h
rng = np.random.default_rng(123)
noise_level = 0.02
data = data * (1.0 + rng.normal(loc=0.0, scale=noise_level, size=data.shape))
# 速度不应为负，做一个下限裁剪
data = np.clip(data, 0.0, None)

# 如果原始数据是 D x T x N 格式，转置为 D x N x T
# 从 D x T x N 转置为 D x N x T
raw_data = np.transpose(data, (0, 2, 1))

# base_array = np.array([
#     [[16.0, 16.0, 16.0, 16.0, 16.0, 15.0, 12.0, 15.0, 16.0]],
#     [[16.0, 16.0, 16.0, 16.0, 15.0, 12.0, 15.0, 16.0, 16.0]],
#     [[16.0, 16.0, 16.0, 15.0, 12.0, 15.0, 16.0, 15.0, 16.0]],
# ])

# # 复制多两个版本，并分别增长20%
# array_20 = base_array * 1.2
# array_40 = base_array * 1.4

# # 按顺序拼接，变成 shape (9, 1, 9)
# raw_data = np.concatenate([base_array, array_20, array_40], axis=1)
print(raw_data.shape)
# B. 执行算法
aligned_data, global_curve, peak_idx, final_state = extract_peak_network_state(raw_data)

# ==========
# Save matrices
# 1) 多天平均为一天：N x T
# 2) DTW 之后的 barycenter：N x T
# ==========
avg_day_matrix = np.mean(raw_data, axis=0)  # (N, T)
np.save("avg_day_speed_nt.npy", avg_day_matrix)
np.save("dtw_barycenter_speed_nt.npy", aligned_data)
print(f"✓ Saved avg_day_speed_nt.npy shape={avg_day_matrix.shape}")
print(f"✓ Saved dtw_barycenter_speed_nt.npy shape={aligned_data.shape}")

# C. 绘图展示
plt.figure(figsize=(15, 10))

# 图1: 展示某一条易堵路段的原始数据 (展示时间偏移)
plt.subplot(2, 2, 1)
road_id = 10  # 选择一条路段（0..79）
for d in range(raw_data.shape[0]):
    plt.plot(raw_data[d, road_id, :], label=f'Day {d+1}', alpha=0.6)

plt.title(f"Raw Data for Road {road_id} (Input)\nNotice Temporal Misalignment", fontweight='bold')
plt.xlabel("Time Step")
plt.ylabel("Speed (km/h)")
plt.legend()
plt.grid(True, alpha=0.3)

# 图2: 展示该路段经过 DTW 对齐后的结果
plt.subplot(2, 2, 2)
plt.plot(aligned_data[road_id, :], color='red', linewidth=2.5, label='DTW Barycenter')
plt.title(f"Aligned Pattern for Road {road_id} (Step 1 Output)\nOutliers Filtered, Peak Preserved", fontweight='bold')
plt.xlabel("Time Step")
plt.ylabel("Speed (km/h)")
plt.legend()
plt.grid(True, alpha=0.3)

# 图3: 全网平均速度曲线 (寻找最痛点)
plt.subplot(2, 2, 3)
plt.plot(global_curve, color='blue')
plt.axvline(x=peak_idx, color='green', linestyle='--', linewidth=2, label=f'Peak Time: {peak_idx}')
plt.scatter(peak_idx, global_curve[peak_idx], color='red', s=100, zorder=5)
plt.title("Global Network Speed Curve (Step 2)\nIdentifying the Worst Congestion Moment", fontweight='bold')
plt.xlabel("Time Step")
plt.ylabel("Avg Network Speed (km/h)")
plt.legend()
plt.grid(True, alpha=0.3)

# 图4: 最终提取的全网状态向量
plt.subplot(2, 2, 4)
colors = ['red' if s < 30 else 'green' for s in final_state]
plt.bar(range(len(final_state)), final_state, color=colors)
plt.title(f"Final Target State Vector at T={peak_idx} (Output)\nThe Input for Calibration", fontweight='bold')
plt.xlabel("Road ID")
plt.ylabel("Target Speed (km/h)")
plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 打印最终输出
print("\n=== 最终结果 (Final Output) ===")
print(f"全网路段数: {len(final_state)}")
print(f"目标状态向量 (部分): {final_state[:10]} ...")