import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import os
from collections import deque
from metric_logger import MetricLogger

import torch
import torch.nn as nn
from torchvision import transforms    

import gym
import gym_super_mario_bros
from gym.spaces import Box
from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace
import datetime
from pathlib import Path
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info
    
# skip 4 frames, take max over the last 2 frames
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)   
    
    def observation(self, obs):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(obs, (2, 0, 1)).copy(), dtype=torch.float))

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env,shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        
    def observation(self, obs):
        transform = transforms.Compose([
            transforms.Resize(self.shape, antialias=True),
            transforms.Normalize(0, 255)
        ])
        return transform(obs).squeeze(0)

class DDQNSolvere(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.online = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input,model):
        if model == 'online':
            return self.online(input)
        else:
            return self.target(input)

class MarioNet(nn.Module):
    """mini CNN structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
    Outputs:
    ``action_idx`` (``int``): An integer representing which action Mario will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
    

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        self.gamma = 0.9

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")
    
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


class DDQNAgent:
    def __init__(self,action_dim,save_dir):
        self.action_dim = action_dim
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.net = DDQNSolvere(self.action_dim).to(device)
        self.exploration_rate = 0.75 #控制随机探索的概率
        self.exploration_rate_decay = 0.999998 # 探索率衰减
        self.exploration_rate_min = 0.01 # 探索率最小值
        self.curr_step = 0 # 当前步数
        self.memory = deque(maxlen=10000) # 经验回放
        self.batch_size = 32 # 批量大小
        self.gamma = 0.78 # 折扣因子
        self.loss_fn = nn.SmoothL1Loss() # 损失函数
        self.sync_period = 10 # 同步频率
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=0.0001) # 优化器
        self.episode_reward = [] # 记录每局游戏的奖励
        self.moving_average_episode_reward = [] # 记录每局游戏的移动平均奖励
        self.current_episode_reward = 0.0 
    
    def log_episode(self):
        self.episode_reward.append(self.current_episode_reward)
        self.current_episode_reward = 0.0
    
    def log_period(self,episode,epsilon,step):
        self.moving_average_episode_reward.append(np.round(np.mean(self.episode_reward[-checkpoint_period:]),3))
        print('Episode:{},Step:{},Epsilon:{},Moving Average Reward:{}'.format(episode,step,epsilon,self.moving_average_episode_reward[-1]))
        return self.moving_average_episode_reward[-1]

    
    def remember(self,state,next_state,action,reward,done):
        self.memory.append((torch.tensor(state.__array__()),
        torch.tensor(next_state.__array__()),
        torch.tensor([action]),torch.tensor([reward]),torch.tensor([done])))

    def experience_replay(self,step_reward):
        self.current_episode_reward += step_reward # 记录当前局游戏的奖励
        if self.curr_step % self.sync_period == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

        if self.batch_size > len(self.memory):
            return

        state, next_state, action, reward, done = self.recall()

        q_values = self.net(state.to(device),model='online')[np.arange(0,self.batch_size),action.to(device).long()]

        with torch.no_grad():
            best_action = torch.argmax(self.net(next_state.to(device),model='online'),dim=1)

            next_q_values = self.net(next_state.to(device),model='target')[np.arange(0,self.batch_size),best_action]

            q_target = reward.to(device) + (1 - done.to(device).float()) * self.gamma * next_q_values

        loss = self.loss_fn(q_values.float(),q_target.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    


    def recall(self):
        state, next_state,action, reward,done = map(torch.stack,zip(*random.sample(self.memory,self.batch_size)))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    
    def act(self,state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            action_values = self.net(torch.tensor(state.__array__()).to(device).unsqueeze(0),model='online')
            action = torch.argmax(action_values,dim=1).item()
        
        # 更新探索率
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min,self.exploration_rate)
        self.curr_step += 1
        return action
        
    def load(self,path):
        checkpoint = torch.load(path,map_location='cpu')
        self.net.load_state_dict(checkpoint['model'])
        self.exploration_rate = checkpoint['exploration_rate']
    
    def save_checkpoint(self):
        filename = os.path.join(self.save_dir,'checkpoint_{}.pth'.format(episode))
        torch.save(dict(model=self.net.state_dict(),exploration_rate=self.exploration_rate),filename)
        print('---checkpoint saved---')

if __name__ == "__main__":
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = 40
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)