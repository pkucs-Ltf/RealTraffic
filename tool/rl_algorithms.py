"""
强化学习算法实现模块
包含DQN和PPO算法的完整实现
"""

import torch
import numpy as np
import random
import os
os.environ["PYTHONHASHSEED"] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
# 为确定性结果牺牲一部分性能
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque
from typing import Dict, List, Tuple, Optional

# 放到任意公共位置，例如 utils/random_utils.py
def set_global_seed(seed: int):
    import os, random
    import numpy as np
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 为确定性结果牺牲一部分性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 某些算子需要
    except Exception:
        pass

class DQNNetwork(nn.Module):
    """DQN神经网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000,
                 batch_size: int = 32,target_update_freq: int = 200,
                 gamma: float = 0.99, hidden_dims: List[int] = [128, 128]):
        set_global_seed(42)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        
        print(torch.__version__) # PyTorch版本
        print(torch.version.cuda) # CUDA版本
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 主网络和目标网络
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 初始化目标网络
        self.update_target_network()
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=memory_size)
        self.update_count = 0
        
        # 训练统计
        self.loss_history = []
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def predict(self, state: np.ndarray, mask: np.ndarray = None, 
                deterministic: bool = False) -> int:
        """预测动作"""
        # epsilon-greedy策略
        if not deterministic and random.random() < self.epsilon:
            # 随机动作
            if mask is not None:
                valid_actions = np.where(mask)[0]
                return np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
            else:
                return random.randint(0, self.action_dim - 1)
        
        # 贪婪动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
            # 应用动作掩码
            if mask is not None:
                q_values = q_values.cpu().numpy()[0]
                q_values[~mask] = -np.inf
                action = np.argmax(q_values)
            else:
                action = q_values.argmax().item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, mask: np.ndarray = None):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self) -> Dict[str, float]:
        """学习更新，返回训练指标"""
        if len(self.memory) < self.batch_size:
            return {}
        
        # 采样经验
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 下一状态的最大Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()
        
        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 记录损失
        self.loss_history.append(loss.item())
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_mean': current_q_values.mean().item()
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'loss_history': self.loss_history
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.update_count = checkpoint.get('update_count', 0)
            self.loss_history = checkpoint.get('loss_history', [])
            print(f"已加载DQN模型: {path}")
        else:
            print(f"模型文件不存在: {path}")


class PPONetwork(nn.Module):
    """PPO神经网络（Actor-Critic）"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super(PPONetwork, self).__init__()
        
        # 共享特征提取层
        shared_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Actor网络（策略）
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络（价值函数）
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        shared_features = self.shared_network(x)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value
    
    def get_action_and_value(self, x, action=None):
        """获取动作概率和状态价值"""
        action_probs, state_value = self.forward(x)
        
        if action is None:
            # 采样动作
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        else:
            # 计算给定动作的对数概率
            dist = torch.distributions.Categorical(action_probs)
            action_logprob = dist.log_prob(action)
        
        return action, action_logprob, state_value, dist.entropy()


class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 n_steps: int = 128, batch_size: int = 32, n_epochs: int = 10,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip_range: float = 0.2,
                 vf_coef: float = 0.5, ent_coef: float = 0.01, max_grad_norm: float = 0.5,
                 hidden_dims: List[int] = [128, 128]):
        set_global_seed(42)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络和优化器
        self.network = PPONetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 轨迹缓冲区
        self.reset_buffer()
        
        # 训练统计
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': []
        }
    
    def reset_buffer(self):
        """重置轨迹缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.step_count = 0
    
    def predict(self, state: np.ndarray, mask: np.ndarray = None, 
                deterministic: bool = False) -> int:
        """预测动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.network(state_tensor)
            
            if deterministic:
                # 确定性策略：选择概率最大的动作
                if mask is not None:
                    action_probs = action_probs.cpu().numpy()[0]
                    action_probs[~mask] = 0
                    action_probs = action_probs / action_probs.sum()
                    action = np.argmax(action_probs)
                else:
                    action = action_probs.argmax().item()
            else:
                # 随机策略：按概率采样
                if mask is not None:
                    action_probs = action_probs.cpu().numpy()[0]
                    action_probs[~mask] = 0
                    action_probs = action_probs / action_probs.sum()
                    action = np.random.choice(len(action_probs), p=action_probs)
                else:
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample().item()
        return action


    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, mask: np.ndarray = None):
        """存储经验"""
        # 获取动作的对数概率和状态价值
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, log_prob, value, _ = self.network.get_action_and_value(
                state_tensor, torch.tensor([action]).to(self.device)
            )
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.item())
        self.log_probs.append(log_prob.item())
        self.step_count += 1
    
    def learn(self) -> Dict[str, float]:
        """学习更新，返回训练指标"""
        if self.step_count < self.n_steps:
            return {}
        
        # 计算优势函数和回报
        advantages, returns = self._compute_gae()
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.n_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            # 分批训练
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                _, new_log_probs, values, entropy = self.network.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO裁剪损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失
                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 累计损失
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # 记录训练统计
        avg_policy_loss = total_policy_loss / (self.n_epochs * len(range(0, len(states), self.batch_size)))
        avg_value_loss = total_value_loss / (self.n_epochs * len(range(0, len(states), self.batch_size)))
        avg_entropy_loss = total_entropy_loss / (self.n_epochs * len(range(0, len(states), self.batch_size)))
        
        self.training_stats['policy_loss'].append(avg_policy_loss)
        self.training_stats['value_loss'].append(avg_value_loss)
        self.training_stats['entropy_loss'].append(avg_entropy_loss)
        self.training_stats['total_loss'].append(avg_policy_loss + avg_value_loss + avg_entropy_loss)
        
        # 重置缓冲区
        self.reset_buffer()
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_policy_loss + avg_value_loss + avg_entropy_loss
        }
    
    def _compute_gae(self) -> Tuple[List[float], List[float]]:
        """计算广义优势估计（GAE）"""
        advantages = []
        returns = []
        
        # 添加最后一个状态的价值（如果没有结束）
        if not self.dones[-1]:
            # 这里简化处理，假设最后状态价值为0
            next_value = 0
        else:
            next_value = 0
        
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_value = self.values[i + 1]
            
            delta = self.rewards[i] + self.gamma * next_value * next_non_terminal - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
        
        return advantages, returns
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            print(f"已加载PPO模型: {path}")
        else:
            print(f"模型文件不存在: {path}")
