from . import RLAgent

from agent import utils
import numpy as np
import os
import random
from collections import deque
import gym

from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class PPOAgent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        self.buffer_size = 1000  # 默认缓冲区大小
        self.replay_buffer_env = deque(maxlen=self.buffer_size)
        self.replay_buffer_prob = deque(maxlen=self.buffer_size)
        # TODO: delete later
        self.env_buffer_count = 0
        self.prob_buffer_count = 0

        self.world = world
        self.sub_agents = 1
        self.rank = rank

        self.phase = True  # 默认相位设置
        self.one_hot = True  # 默认one_hot设置
        self.model_dict = {}  # 默认模型字典

        # get generator for each DQNAgent
        inter_id = self.world.intersection_ids[self.rank]
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = LaneVehicleGenerator(self.world,  self.inter, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(world,  self.inter, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world,  self.inter, ["lane_waiting_count"],
                                                     in_only=True, average='all', negative=True)
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[inter_id].phases))

        if self.phase:
            if self.one_hot:
                self.ob_length = self.ob_generator.ob_length + len(self.world.id2intersection[inter_id].phases)
            else:
                self.ob_length = self.ob_generator.ob_length + 1
        else:
            self.ob_length = self.ob_generator.ob_length

        self.gamma = 0.99
        self.grad_clip = 0.5
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.vehicle_max = 100
        self.batch_size = 64
        self.update_interval = 1000
        assert(self.update_interval == self.buffer_size)

        # generate samples
        self.actor = Actor(self.ob_length, self.action_space.n)
        # update policy
        self.critic = Critic(self.ob_length)
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.learning_rate,
                                       alpha=0.9, centered=False, eps=1e-7)


    def reset(self):
        inter_id = self.tld
        inter_obj = self.world.id2intersection[inter_id]
        self.inter = inter_obj
        self.ob_generator = 
        self.phase_generator = IntersectionPhaseGenerator(self.world, inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator =

    def get_ob(self):
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        rewards = []
        rewards.append(self.reward_generator.generate())
        rewards = np.squeeze(np.array(rewards)) * 12
        return rewards

    def get_phase(self):
        phase = []
        phase.append(self.phase_generator.generate())
        phase = np.concatenate(phase, dtype=np.int8)
        return phase

    def get_action(self, ob, phase, test=False):
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        if self.phase:
            if self.one_hot:
                feature = np.concatenate([ob, utils.idx2onehot(phase, self.action_space.n)], axis=1)
            else:
                feature = np.concatenate([ob, phase], axis=1)
        else:
            feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        actions = self.model(observation, train=True)
        actions = actions.clone().detach().numpy()
        return np.argmax(actions, axis=1)

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def remember(self, last_obs, last_phase, actions, rewards, obs, cur_phase, key):
        self.replay_buffer_env.append((key, (np.reshape(last_obs, [1, self.ob_generator.ob_length]),
                                             np.reshape(last_phase, [1, 1]),
                                             np.reshape(actions, [1, 1]),
                                             np.reshape(rewards * 0.01, [1, 1]),
                                             np.reshape(obs, [1, self.ob_generator.ob_length]),
                                             np.reshape(cur_phase), [1, 1])))
        self.env_buffer_count += 1
        self._check_buffer_update()

    def _check_buffer_update(self):
        assert(self.env_buffer_count == self.prob_buffer_count)
        if self.env_buffer_count == self.update_interval:
            self.AC_train()

    def _prepare_data(self):
        obs_t = np.concatenate([item[1][0] for item in self.replay_buffer_env])
        obs_tp = np.concatenate([item[1][4] for item in self.replay_buffer_env])
        old_policy = np.concatenate([item for item in self.replay_buffer_prob])
        actions = np.concatenate([item[1][2] for item in self.replay_buffer_env])
        rewards = np.concatenate([item[1][3] for item in self.replay_buffer_env])
        if self.phase:
            if self.one_hot:
                phase_t = np.concatenate([utils.idx2onehot(item[1][1], self.action_space.n)
                                          for item in self.replay_buffer_env])
                phase_tp = np.concatenate([utils.idx2onehot(item[1][5], self.action_space.n)
                                           for item in self.replay_buffer_env])
            else:
                phase_t = np.concatenate([item[1][1] for item in self.replay_buffer_env])
                phase_tp = np.concatenate([item[1][5] for item in self.replay_buffer_env])
            feature_t = np.concatenate([obs_t, phase_t], axis=1)
            feature_tp = np.concatenate([obs_tp, phase_tp], axis=1)
        else:
            feature_t = obs_t
            feature_tp = obs_tp
        state_t = torch.tensor(feature_t, dtype=torch.float32)
        state_tp = torch.tensor(feature_tp, dtype=torch.float32)

    def train(self):
        # TODO: we do not train here
        pass

    def update_target_network(self):
        pass

    def load_model(self, e):
        model_name = os.path.join('output_path', 'model', f'{e}_{self.rank}.pt')  # 默认路径
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = self._build_model()
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join('output_path', 'model')  # 默认路径
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)


class Actor(object):
    def __init__(self, input_dim, output_dim):
        self.state_dim = input_dim
        self.action_dim = output_dim
        self.model = self._build_model()

    def _build_model(self):
        model = PPO_ActDQN(self.state_dim, self.action_dim)
        return model


class Critic(object):
    def __init__(self, input_dim):
        self.state_dim = input_dim
        self.model = self._build_model()


    def _build_model(self):
        model = PPO_CrtDQN(self.state_dim)
        return model


class PPO_ActDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO_ActDQN, self).__init__()
        self.dense_1 = nn.Linear(input_dim, 32)
        self.dense_2 = nn.Linear(32, 16)
        self.dense_3 = nn.Linear(16, output_dim)

    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = nn.Softmax(self.dense_3(x))
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class PPO_CrtDQN(nn.Module):
    def __init__(self, input_dim):
        super(PPO_CrtDQN, self).__init__()
        self.dense_1 = nn.Linear(input_dim, 32)
        self.dense_2 = nn.Linear(32, 16)
        self.dense_3 = nn.Linear(16, 16)
        self.dense_4 = nn.Linear(16, 1)

    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = self.dense_4(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)






