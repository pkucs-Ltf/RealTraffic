"""
RL集成测试脚本
测试强化学习功能的基本集成是否正常
"""

import sys
import os
import numpy as np
from typing import Dict, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试导入功能"""
    print("=== 测试导入功能 ===")
    
    try:
        from tool.rl_config import get_default_rl_config, validate_rl_config
        print("✓ RL配置模块导入成功")
        
        from tool.rl_algorithms import DQNAgent, PPOAgent
        print("✓ RL算法模块导入成功")
        
        from tool.rl_controllers import (
            TrafficLightController, 
            GreedyController, 
            StaticController,
            DQNController,
            PPOController,
            StateExtractor,
            RewardCalculator,
            TrafficLightControllerManager
        )
        print("✓ RL控制器模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_config_validation():
    """测试配置验证功能"""
    print("\n=== 测试配置验证功能 ===")
    
    try:
        from tool.rl_config import get_default_rl_config, validate_rl_config
        
        # 测试默认配置
        config = get_default_rl_config()
        is_valid = validate_rl_config(config)
        
        if is_valid:
            print("✓ 默认配置验证通过")
        else:
            print("✗ 默认配置验证失败")
            return False
        
        # 测试无效配置
        invalid_config = {'algorithm': 'invalid_algo'}
        is_valid = validate_rl_config(invalid_config)
        
        if not is_valid:
            print("✓ 无效配置正确被拒绝")
        else:
            print("✗ 无效配置验证应该失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 配置验证测试失败: {e}")
        return False


def test_rl_algorithms():
    """测试RL算法基本功能"""
    print("\n=== 测试RL算法基本功能 ===")
    
    try:
        from tool.rl_algorithms import DQNAgent, PPOAgent
        
        # 测试DQN
        state_dim = 20
        action_dim = 4
        
        dqn_agent = DQNAgent(state_dim, action_dim, lr=1e-3, memory_size=1000)
        
        # 测试预测
        state = np.random.random(state_dim)
        action = dqn_agent.predict(state, deterministic=True)
        
        if 0 <= action < action_dim:
            print("✓ DQN动作预测正常")
        else:
            print(f"✗ DQN动作预测异常: {action}")
            return False
        
        # 测试PPO
        ppo_agent = PPOAgent(state_dim, action_dim, lr=3e-4, n_steps=64)
        
        action = ppo_agent.predict(state, deterministic=True)
        
        if 0 <= action < action_dim:
            print("✓ PPO动作预测正常")
        else:
            print(f"✗ PPO动作预测异常: {action}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ RL算法测试失败: {e}")
        return False


def test_controllers():
    """测试控制器功能"""
    print("\n=== 测试控制器功能 ===")
    
    try:
        from tool.rl_controllers import (
            GreedyController, 
            StaticController,
            StateExtractor,
            RewardCalculator
        )
        
        # 模拟交叉路口数据
        intersections = {
            'tl_001': {
                'phase_available_lanelinks': [
                    (0, [('lane_1', 'lane_2'), ('lane_3', 'lane_4')]),
                    (1, [('lane_5', 'lane_6'), ('lane_7', 'lane_8')])
                ]
            }
        }
        
        # 测试静态控制器
        static_programs = {
            'tl_001': [
                {'duration': 30, 'state': 0},
                {'duration': 5, 'state': 1},
                {'duration': 25, 'state': 1},
                {'duration': 5, 'state': 0}
            ]
        }
        
        static_controller = StaticController(static_programs, intersections)
        
        observation = {'current_phase': 0}
        action = static_controller.decide_action('tl_001', observation, 0)
        
        if isinstance(action, int):
            print("✓ 静态控制器工作正常")
        else:
            print(f"✗ 静态控制器返回异常: {action}")
            return False
        
        # 测试状态提取器（需要模拟连接）
        class MockConnection:
            def __init__(self):
                self.lane = MockLane()
        
        class MockLane:
            def getLastStepVehicleNumber(self, lane_id):
                return np.random.randint(0, 10)
            
            def getLastStepHaltingNumber(self, lane_id):
                return np.random.randint(0, 5)
            
            def getLastStepMeanSpeed(self, lane_id):
                return np.random.uniform(0, 15)
            
            def getLastStepVehicleIDs(self, lane_id):
                return [f'veh_{i}' for i in range(np.random.randint(0, 5))]
        
        mock_conn = MockConnection()
        state_extractor = StateExtractor(intersections, mock_conn, ['tl_001'])
        
        current_phases = {'tl_001': 0}
        current_phase_times = {'tl_001': 10}
        
        state = state_extractor.extract_observation('tl_001', current_phases, current_phase_times)
        
        if isinstance(state, np.ndarray) and len(state) > 0:
            print("✓ 状态提取器工作正常")
        else:
            print(f"✗ 状态提取器返回异常: {state}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 控制器测试失败: {e}")
        return False


def test_ltf_integration():
    """测试LTF_Traci集成（不需要实际SUMO文件）"""
    print("\n=== 测试LTF_Traci集成 ===")
    
    try:
        from tool.rl_config import get_default_rl_config
        
        # 测试参数解析
        rl_config = get_default_rl_config()
        
        # 模拟LTF_Traci初始化参数
        ltf_params = {
            'net_file': 'dummy.net.xml',
            'route_file': 'dummy.rou.xml',
            'use_gui': False,
            'end_time': 100,
            'rl_mode': 'ppo',
            'rl_tls_ids': ['tl_001', 'tl_002'],
            'non_rl_policy': 'greedy',
            'training': True,
            'checkpoint_dir': 'test_checkpoints',
            'rl_config': rl_config,
            'num_episodes': 5
        }
        
        # 验证参数完整性
        required_rl_params = ['rl_mode', 'rl_tls_ids', 'non_rl_policy', 'training', 'rl_config']
        
        for param in required_rl_params:
            if param not in ltf_params:
                print(f"✗ 缺少必需参数: {param}")
                return False
        
        print("✓ LTF_Traci RL参数完整")
        
        # 验证RL配置
        from tool.rl_config import validate_rl_config
        if validate_rl_config(ltf_params['rl_config']):
            print("✓ RL配置验证通过")
        else:
            print("✗ RL配置验证失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ LTF集成测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("强化学习交通信号控制系统集成测试")
    print("=" * 50)
    import pdb
    pdb.set_trace()
    
    tests = [
        test_imports,
        test_config_validation,
        test_rl_algorithms,
        test_controllers,
        test_ltf_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"测试失败: {test.__name__}")
        except Exception as e:
            print(f"测试异常: {test.__name__} - {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！RL功能集成成功")
        return True
    else:
        print("❌ 部分测试失败，请检查相关功能")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
