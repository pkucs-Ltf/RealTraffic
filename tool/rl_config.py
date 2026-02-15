"""
强化学习配置管理模块
提供默认配置和配置验证功能
"""

from typing import Dict, List, Any
import os
import json


def get_default_rl_config() -> Dict[str, Any]:
    """获取默认的RL配置"""
    return {
        'algorithm': 'ppo',  # 'dqn' or 'ppo'
        'agent_mode': 'independent',  # 每路口独立智能体
        
        'state_config': {
            'include_global_features': True,  # 包含全局效率特征
            'normalize_method': 'minmax'  # 'minmax', 'zscore', 'none'
        },
        
        'reward_config': {
            # 三层奖励权重配置
            'local_weight': 0.75,      # 局部路口奖励权重
            'rl_group_weight': 0.15,   # RL路口群体效率权重
            'global_weight': 0.10,     # 全局效率权重
            
            # 局部奖励组件开关
            'use_local_queue': True,
            'use_local_waiting': True,
            'use_local_throughput': True,
            
            # 奖励归一化参数
            'reward_scale': 1.0,
            'reward_clip': [-100, 100]
        },
        
        'dqn_params': {
            'lr': 1e-3,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update_freq': 100,
            'gamma': 0.99,
            'hidden_dims': [128, 128]
        },
        
        'ppo_params': {
            'lr': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'vf_coef': 0.5,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5,
            'hidden_dims': [128, 128]
        }
    }


def get_training_config() -> Dict[str, Any]:
    """获取训练配置"""
    return {
        'rl_mode': 'ppo',
        'rl_tls_ids': ['cluster_123', 'cluster_456', 'cluster_789'],  # 示例路口ID
        'non_rl_policy': 'greedy',  # 其他路口使用贪心策略
        'training': True,
        'checkpoint_dir': 'pkl/rl_checkpoints',
        'rl_config': get_default_rl_config(),
        'num_episodes': 100
    }


def get_inference_config() -> Dict[str, Any]:
    """获取推理配置"""
    return {
        'rl_mode': 'ppo',
        'rl_tls_ids': ['cluster_123', 'cluster_456', 'cluster_789'],  # 示例路口ID
        'non_rl_policy': 'greedy',
        'training': False,  # 推理模式
        'checkpoint_dir': 'pkl/rl_checkpoints',  # 加载训练好的模型
        'rl_config': get_default_rl_config(),
        'num_episodes': 1
    }


def validate_rl_config(config: Dict[str, Any]) -> bool:
    """验证RL配置的有效性"""
    try:
        # 检查必需的顶级键
        required_keys = ['algorithm', 'reward_config']
        for key in required_keys:
            if key not in config:
                print(f"缺少必需的配置键: {key}")
                return False
        
        # 检查算法类型
        if config['algorithm'] not in ['dqn', 'ppo']:
            print(f"不支持的算法类型: {config['algorithm']}")
            return False
        
        # 检查奖励配置
        reward_config = config['reward_config']
        weight_sum = (reward_config.get('local_weight', 0) + 
                     reward_config.get('rl_group_weight', 0) + 
                     reward_config.get('global_weight', 0))
        
        if abs(weight_sum - 1.0) > 0.01:
            print(f"警告: 奖励权重之和不等于1.0，当前为: {weight_sum}")
        
        # 检查算法特定参数
        if config['algorithm'] == 'dqn' and 'dqn_params' not in config:
            print("DQN算法缺少dqn_params配置")
            return False
        
        if config['algorithm'] == 'ppo' and 'ppo_params' not in config:
            print("PPO算法缺少ppo_params配置")
            return False
        
        return True
        
    except Exception as e:
        print(f"配置验证出错: {e}")
        return False


def save_config(config: Dict[str, Any], filepath: str):
    """保存配置到文件"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"配置已保存到: {filepath}")
    except Exception as e:
        print(f"保存配置失败: {e}")


def load_config(filepath: str) -> Dict[str, Any]:
    """从文件加载配置"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"配置已从 {filepath} 加载")
        return config
    except FileNotFoundError:
        print(f"配置文件不存在: {filepath}，使用默认配置")
        return get_default_rl_config()
    except Exception as e:
        print(f"加载配置失败: {e}，使用默认配置")
        return get_default_rl_config()


def create_experiment_configs() -> Dict[str, Dict[str, Any]]:
    """创建不同实验配置"""
    base_config = get_default_rl_config()
    
    experiments = {
        # 基础PPO实验
        'ppo_baseline': {
            **base_config,
            'algorithm': 'ppo',
            'reward_config': {
                **base_config['reward_config'],
                'local_weight': 0.75,
                'rl_group_weight': 0.15,
                'global_weight': 0.10
            }
        },
        
        # 基础DQN实验
        'dqn_baseline': {
            **base_config,
            'algorithm': 'dqn',
            'reward_config': {
                **base_config['reward_config'],
                'local_weight': 0.75,
                'rl_group_weight': 0.15,
                'global_weight': 0.10
            }
        },
        
        # 纯局部奖励实验
        'local_only': {
            **base_config,
            'algorithm': 'ppo',
            'reward_config': {
                **base_config['reward_config'],
                'local_weight': 1.0,
                'rl_group_weight': 0.0,
                'global_weight': 0.0
            }
        },
        
        # 强调全局协调实验
        'global_emphasis': {
            **base_config,
            'algorithm': 'ppo',
            'reward_config': {
                **base_config['reward_config'],
                'local_weight': 0.5,
                'rl_group_weight': 0.25,
                'global_weight': 0.25
            }
        },
        
        # 高学习率实验
        'high_lr': {
            **base_config,
            'algorithm': 'ppo',
            'ppo_params': {
                **base_config['ppo_params'],
                'lr': 1e-3  # 提高学习率
            }
        }
    }
    
    return experiments


# 预定义的路口组合配置
INTERSECTION_GROUPS = {
    'small_network': ['tl_1', 'tl_2', 'tl_3'],
    'medium_network': ['tl_1', 'tl_2', 'tl_3', 'tl_4', 'tl_5'],
    'large_network': ['tl_1', 'tl_2', 'tl_3', 'tl_4', 'tl_5', 'tl_6', 'tl_7', 'tl_8'],
    'arterial_road': ['arterial_1', 'arterial_2', 'arterial_3', 'arterial_4'],
    'city_center': ['center_1', 'center_2', 'center_3', 'center_4', 'center_5']
}


def get_intersection_config(network_type: str) -> List[str]:
    """获取预定义的路口配置"""
    return INTERSECTION_GROUPS.get(network_type, [])


if __name__ == "__main__":
    # 示例：创建和保存配置
    print("=== RL配置管理示例 ===")
    
    # 获取默认配置
    default_config = get_default_rl_config()
    print("默认配置:")
    print(json.dumps(default_config, indent=2, ensure_ascii=False))
    
    # 验证配置
    is_valid = validate_rl_config(default_config)
    print(f"\n配置验证结果: {'通过' if is_valid else '失败'}")
    
    # 保存配置
    save_config(default_config, 'configs/default_rl_config.json')
    
    # 创建实验配置
    experiments = create_experiment_configs()
    for name, config in experiments.items():
        save_config(config, f'configs/experiment_{name}.json')
    
    print(f"\n已创建 {len(experiments)} 个实验配置")
    print("配置文件保存在 configs/ 目录下")
