"""
强化学习交通信号控制系统使用示例
展示如何使用改进后的LTF_Traci进行RL训练和推理
"""

import os
import sys
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool.Car_simulate_class import LTF_Traci
from tool.rl_config import (
    get_default_rl_config, 
    get_training_config, 
    get_inference_config,
    validate_rl_config,
    create_experiment_configs,
    get_intersection_config
)


def example_rl_training():
    """示例：RL训练"""
    print("=== RL训练示例 ===")
    
    # 基础参数
    net_file = "your_network.net.xml"  # 替换为实际的网络文件
    route_file = "your_routes.rou.xml"  # 替换为实际的路由文件
    
    # 获取训练配置
    training_config = get_training_config()
    
    # 自定义RL路口（根据实际网络调整）
    training_config['rl_tls_ids'] = ['tl_001', 'tl_002', 'tl_003']  # 替换为实际路口ID
    training_config['num_episodes'] = 50  # 训练回合数
    
    # 验证配置
    if not validate_rl_config(training_config['rl_config']):
        print("配置验证失败，退出训练")
        return
    
    try:
        # 创建LTF实例（训练模式）
        ltf = LTF_Traci(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,  # 训练时通常不使用GUI
            end_time=3600,  # 1小时仿真
            **training_config
        )
        
        # 开始训练
        print(f"开始RL训练，共 {training_config['num_episodes']} 回合...")
        training_stats = ltf.run()
        
        # 分析训练结果
        print("训练完成！")
        print(f"训练统计: {len(training_stats['episode_metrics'])} 回合数据")
        
        # 可以在这里添加训练结果可视化
        # plot_training_results(training_stats)
        
        return training_stats
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return None


def example_rl_inference():
    """示例：RL推理"""
    print("\n=== RL推理示例 ===")
    
    # 基础参数
    net_file = "your_network.net.xml"  # 替换为实际的网络文件
    route_file = "your_routes.rou.xml"  # 替换为实际的路由文件
    
    # 获取推理配置
    inference_config = get_inference_config()
    
    # 自定义RL路口（应与训练时一致）
    inference_config['rl_tls_ids'] = ['tl_001', 'tl_002', 'tl_003']
    
    try:
        # 创建LTF实例（推理模式）
        ltf = LTF_Traci(
            net_file=net_file,
            route_file=route_file,
            use_gui=True,  # 推理时可以使用GUI观察
            end_time=3600,
            **inference_config
        )
        
        # 运行推理
        print("开始RL推理...")
        results = ltf.run()
        
        # 分析结果
        metrics_history, average_speeds, road_vehicles, average_waiting_vehicles, _ = results
        
        print("推理完成！")
        print(f"收集到 {len(metrics_history)} 个时间步的指标")
        
        # 计算性能指标
        if metrics_history:
            avg_waiting = np.mean([m['waiting_time'] for m in metrics_history])
            avg_queue = np.mean([m['queue_length'] for m in metrics_history])
            avg_speed = np.mean([m.get('mean_speed', 0) for m in metrics_history])
            
            print(f"平均等待时间: {avg_waiting:.2f}")
            print(f"平均排队长度: {avg_queue:.2f}")
            print(f"平均车速: {avg_speed:.2f} m/s")
        
        return results
        
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        return None


def example_baseline_comparison():
    """示例：基线对比实验"""
    print("\n=== 基线对比实验 ===")
    
    net_file = "your_network.net.xml"
    route_file = "your_routes.rou.xml"
    
    results = {}
    
    # 1. 原始MaxPressure基线
    print("运行MaxPressure基线...")
    try:
        ltf_baseline = LTF_Traci(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            end_time=1800,  # 30分钟
            rl_mode='none'  # 不使用RL
        )
        
        baseline_results = ltf_baseline.run()
        results['maxpressure'] = baseline_results
        print("MaxPressure基线完成")
        
    except Exception as e:
        print(f"MaxPressure基线运行失败: {e}")
    
    # 2. RL方法
    print("运行RL方法...")
    try:
        inference_config = get_inference_config()
        inference_config['rl_tls_ids'] = ['tl_001', 'tl_002']  # 部分路口使用RL
        inference_config['num_episodes'] = 1
        
        ltf_rl = LTF_Traci(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            end_time=1800,
            **inference_config
        )
        
        rl_results = ltf_rl.run()
        results['rl'] = rl_results
        print("RL方法完成")
        
    except Exception as e:
        print(f"RL方法运行失败: {e}")
    
    # 3. 对比分析
    if len(results) >= 2:
        print("\n=== 性能对比 ===")
        compare_results(results)
    
    return results


def compare_results(results: Dict):
    """对比不同方法的结果"""
    for method_name, result in results.items():
        if result and len(result) >= 1:
            metrics_history = result[0]  # 第一个元素是metrics_history
            
            if metrics_history:
                avg_waiting = np.mean([m['waiting_time'] for m in metrics_history])
                avg_queue = np.mean([m['queue_length'] for m in metrics_history])
                avg_speed = np.mean([m.get('mean_speed', 0) for m in metrics_history])
                
                print(f"\n{method_name.upper()}:")
                print(f"  平均等待时间: {avg_waiting:.2f}")
                print(f"  平均排队长度: {avg_queue:.2f}")
                print(f"  平均车速: {avg_speed:.2f} m/s")


def example_experiment_configs():
    """示例：实验配置管理"""
    print("\n=== 实验配置管理示例 ===")
    
    # 创建实验配置
    experiments = create_experiment_configs()
    
    print(f"可用实验配置: {list(experiments.keys())}")
    
    # 选择一个实验配置
    experiment_name = 'ppo_baseline'
    config = experiments[experiment_name]
    
    print(f"\n使用实验配置: {experiment_name}")
    print(f"算法: {config['algorithm']}")
    print(f"奖励权重: 局部={config['reward_config']['local_weight']}, "
          f"群体={config['reward_config']['rl_group_weight']}, "
          f"全局={config['reward_config']['global_weight']}")
    
    # 验证配置
    is_valid = validate_rl_config(config)
    print(f"配置验证: {'通过' if is_valid else '失败'}")
    
    return config


def example_intersection_selection():
    """示例：路口选择策略"""
    print("\n=== 路口选择策略示例 ===")
    
    # 获取预定义的路口组合
    small_network = get_intersection_config('small_network')
    medium_network = get_intersection_config('medium_network')
    
    print(f"小型网络路口: {small_network}")
    print(f"中型网络路口: {medium_network}")
    
    # 自定义路口选择策略
    def select_critical_intersections(all_intersections: List[str], 
                                    max_rl_intersections: int = 3) -> List[str]:
        """选择关键路口进行RL控制"""
        # 这里可以实现更复杂的选择逻辑
        # 例如基于交通流量、拥堵程度等
        return all_intersections[:max_rl_intersections]
    
    # 示例使用
    all_tls = ['tl_001', 'tl_002', 'tl_003', 'tl_004', 'tl_005']
    selected_tls = select_critical_intersections(all_tls, 3)
    
    print(f"选择的RL控制路口: {selected_tls}")
    
    return selected_tls


def plot_training_results(training_stats: Dict):
    """绘制训练结果"""
    try:
        import matplotlib.pyplot as plt
        
        episode_metrics = training_stats['episode_metrics']
        
        if not episode_metrics:
            print("没有训练数据可绘制")
            return
        
        # 提取每回合的平均指标
        episodes = range(1, len(episode_metrics) + 1)
        avg_waiting_times = []
        avg_queue_lengths = []
        avg_speeds = []
        
        for metrics_list in episode_metrics:
            if metrics_list:
                avg_waiting_times.append(np.mean([m['waiting_time'] for m in metrics_list]))
                avg_queue_lengths.append(np.mean([m['queue_length'] for m in metrics_list]))
                avg_speeds.append(np.mean([m.get('mean_speed', 0) for m in metrics_list]))
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('RL训练结果', fontsize=16)
        
        # 等待时间
        axes[0, 0].plot(episodes, avg_waiting_times, 'b-', linewidth=2)
        axes[0, 0].set_title('平均等待时间')
        axes[0, 0].set_xlabel('回合')
        axes[0, 0].set_ylabel('等待时间')
        axes[0, 0].grid(True)
        
        # 排队长度
        axes[0, 1].plot(episodes, avg_queue_lengths, 'r-', linewidth=2)
        axes[0, 1].set_title('平均排队长度')
        axes[0, 1].set_xlabel('回合')
        axes[0, 1].set_ylabel('排队长度')
        axes[0, 1].grid(True)
        
        # 平均速度
        axes[1, 0].plot(episodes, avg_speeds, 'g-', linewidth=2)
        axes[1, 0].set_title('平均车速')
        axes[1, 0].set_xlabel('回合')
        axes[1, 0].set_ylabel('速度 (m/s)')
        axes[1, 0].grid(True)
        
        # 综合性能指标（等待时间的负值，越高越好）
        performance_score = [-wt for wt in avg_waiting_times]
        axes[1, 1].plot(episodes, performance_score, 'm-', linewidth=2)
        axes[1, 1].set_title('性能得分（负等待时间）')
        axes[1, 1].set_xlabel('回合')
        axes[1, 1].set_ylabel('得分')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('rl_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("训练结果图表已保存为 rl_training_results.png")
        
    except ImportError:
        print("matplotlib未安装，跳过绘图")
    except Exception as e:
        print(f"绘图出错: {e}")


def main():
    """主函数：运行所有示例"""
    print("强化学习交通信号控制系统示例")
    print("=" * 50)
    
    # 1. 配置管理示例
    example_experiment_configs()
    
    # 2. 路口选择示例
    example_intersection_selection()
    
    # 3. 基线对比示例（需要实际的网络文件）
    # example_baseline_comparison()
    
    # 4. RL训练示例（需要实际的网络文件）
    # training_stats = example_rl_training()
    # if training_stats:
    #     plot_training_results(training_stats)
    
    # 5. RL推理示例（需要实际的网络文件）
    # example_rl_inference()
    
    print("\n示例运行完成！")
    print("要运行实际的RL训练或推理，请：")
    print("1. 准备SUMO网络文件(.net.xml)和路由文件(.rou.xml)")
    print("2. 修改示例中的文件路径")
    print("3. 根据实际网络调整路口ID")
    print("4. 取消注释相应的示例函数调用")


if __name__ == "__main__":
    main()
