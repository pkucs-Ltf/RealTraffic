"""
LTF_Traci 基本功能测试
"""

from Car_simulate_class import LTF_Traci


def test_ltf_traci_basic():
    """测试 LTF_Traci 的基本功能"""
    print("=== 测试 LTF_Traci 基本功能 ===")
    
    # 手动设置参数
    net_file = "test_RL//test.net.xml"  # 替换为实际的网络文件
    route_file = "test_RL//test.rou.xml"  # 替换为实际的路由文件
    
    # 创建 LTF_Traci 实例
    ltf = LTF_Traci(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        end_time=1000,  # 运行1000秒
        rl_mode='ppo',  # 使用PPO算法
        rl_tls_ids=['cluster_10210027521_10210027556_10219731278_1422005338_#2more', 'cluster_1422005356_1771677079_380720703','321557749','cluster_1497364720_1497364722'],  # 示例路口ID
        non_rl_policy='greedy',
        training=True,  # 训练模式
        checkpoint_dir='test_pth/rl_models',  # 模型保存路径
        rl_config={
            'algorithm': 'ppo',
            'reward_config': {
                'local_weight': 0.75,
                'rl_group_weight': 0.15,
                'global_weight': 0.10,
                'use_local_queue': True,
                'use_local_waiting': True,
                'use_local_throughput': True
            },
            'ppo_params': {
                'lr': 3e-4,
                'n_steps': 2048,
                'batch_size': 2,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2
            }
        },
        num_episodes=10  # 训练回合数
    )
    
    # 运行仿真
    try:
        import pdb
        pdb.set_trace()
        results = ltf.run()
        print("仿真完成，结果:", results)
    except Exception as e:
        print(f"仿真过程中出现错误: {e}")


if __name__ == "__main__":
    test_ltf_traci_basic()
