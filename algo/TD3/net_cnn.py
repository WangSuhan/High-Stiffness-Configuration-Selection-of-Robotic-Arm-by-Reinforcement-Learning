#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class PolicyNet_CNN(torch.nn.Module):
    '''策略网络'''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet_CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(state_dim, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(256, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        # x = x.unsqueeze(1)  # 增加一个维度,以匹配卷积层的输入要求
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # 展平卷积层的输出
        x = F.leaky_relu(self.fc1(x.T))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet_CNN(torch.nn.Module):
    '''估值网络'''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet_CNN, self).__init__()
        # Q1 architecture
        self.conv1 = torch.nn.Conv1d(state_dim + action_dim, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(256, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.conv5 = torch.nn.Conv1d(state_dim + action_dim, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc3 = torch.nn.Linear(256, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        cat = torch.cat([state, action], dim=1)  # 拼接状态和动作
        cat = cat.unsqueeze(1)  # 增加一个维度,以匹配卷积层的输入要求

        q1 = F.leaky_relu(self.conv1(cat))
        q1 = F.leaky_relu(self.conv2(q1))
        q1 = F.leaky_relu(self.conv3(q1))
        q1 = F.leaky_relu(self.conv4(q1))
        q1 = q1.view(q1.size(0), -1)  # 展平卷积层的输出
        q1 = F.leaky_relu(self.fc1(q1))
        q1 = self.fc2(q1)

        q2 = F.leaky_relu(self.conv5(cat))
        q2 = F.leaky_relu(self.conv6(q2))
        q2 = F.leaky_relu(self.conv7(q2))
        q2 = F.leaky_relu(self.conv8(q2))
        q2 = q2.view(q2.size(0), -1)  # 展平卷积层的输出
        q2 = F.leaky_relu(self.fc3(q2))
        q2 = self.fc4(q2)

        return q1, q2

    def Q1(self, state, action):
        cat = torch.cat([state, action], dim=1)  # 拼接状态和动作
        cat = cat.unsqueeze(1)  # 增加一个维度,以匹配卷积层的输入要求

        q1 = F.leaky_relu(self.conv1(cat))
        q1 = F.leaky_relu(self.conv2(q1))
        q1 = F.leaky_relu(self.conv3(q1))
        q1 = F.leaky_relu(self.conv4(q1))
        q1 = q1.view(q1.size(0), -1)  # 展平卷积层的输出
        q1 = F.leaky_relu(self.fc1(q1))
        q1 = self.fc2(q1)
        return q1