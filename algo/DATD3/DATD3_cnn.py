#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import copy
from algo.DATD3.net_cnn import PolicyNet_CNN, QValueNet_CNN
import numpy as np

class DATD3_CNN:
    ''' DATD3算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, sigma, tau, gamma, policy_noise, noise_clip, policy_freq, device):
        '''
        用于初始化TD3算法中的各项参数，
        初始化策略网络与估值网络

        Args:
            state_dim (tuple):       状态空间维数
            hidden_dim (int):      隐藏层大小
            action_dim (int):      动作空间维数
            action_bound (float):  动作空间限幅
            actor_lr (float):      策略网络学习率
            critic_lr (float):     估值网络学习率
            sigma (float):         高斯噪声的标准差
            tau (float):           目标网络软更新参数
            gamma (float):         折扣因子
            policy_noise (float):  策略噪声
            noise_clip (float):    噪声限幅
            policy_freq (int):     延迟更新频率
            device (any):          训练设备

        Returns:
            None
        '''
        self.action_dim = action_dim
        self.actor = PolicyNet_CNN(state_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet_CNN(state_dim, action_dim).to(device)
        self.target_actor = PolicyNet_CNN(state_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet_CNN(state_dim, action_dim).to(device)
        # 初始化目标价值网络并使其参数和价值网络一样
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并使其参数和策略网络一样
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_bound = action_bound
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.device = device
        self.total_it = 0

    def take_action(self, state):
        '''
        由策略网络选择动作，
        并加入高斯噪声增加探索效率

        Args:
            state (array):  当前智能体状态

        Returns:
            action (array): 智能体的下一步动作
        '''
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        # 给动作添加噪声，增加探索
        # action = (action + np.random.normal(0, self.action_bound * self.sigma, size=self.action_dim))
        return action

    def soft_update(self, net, target_net):
        '''
        软更新策略，
        采用当前网络参数和一部分过去网络参数一起更新，使得网络参数更加平滑

        Args:
            net (any):  更新网络
            target_net (any): 目标更新网络

        Returns:
            None
        '''
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def train(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        self.total_it += 1
        noise = (
                torch.randn_like(actions) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)

        next_action = (
                self.target_actor(next_states) + noise
        ).clamp(-self.action_bound, self.action_bound)

        target_Q1, target_Q2 = self.target_critic(next_states, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = rewards + (1 - dones) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # critic_loss.requires_grad_(True)  # 加入此句就行了

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # 策略网络就是为了使Q值最大化
            actor_loss = -torch.mean(self.critic.Q1(states, self.actor(states)))
            # actor_loss.requires_grad_(True)  # 加入此句就行了
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
            self.soft_update(self.critic, self.target_critic)  # 软更新价值网络
        return critic_loss.cpu().detach().numpy()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pt")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pt")

        torch.save(self.actor.state_dict(), filename + "_actor.pt")
        # torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pt")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pt"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))
        self.actor_target = copy.deepcopy(self.actor)