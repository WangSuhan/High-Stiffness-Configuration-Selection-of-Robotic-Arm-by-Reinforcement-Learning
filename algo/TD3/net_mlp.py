#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    '''残差块，包含一个全连接层和批量归一化层'''

    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.fc(x)))
        out += residual
        return out


class PolicyNet(nn.Module):
    '''策略网络'''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # 创建 7 个残差块，总共 8 层
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(10)])

        self.fc_out = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound 是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        for res_block in self.res_blocks:
            x = res_block(x)
        x = torch.tanh(self.fc_out(x)) * self.action_bound
        return x
class QValueNet(nn.Module):
    '''价值网络'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        # Q1 网络
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.res_blocks1 = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(10)])
        self.fc_out1 = nn.Linear(hidden_dim, 1)

        # Q2 网络
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.res_blocks2 = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(10)])
        self.fc_out2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        cat = torch.cat([state, action], dim=1)  # 拼接状态和动作

        # 计算 Q1
        x1 = F.relu(self.bn1(self.fc1(cat)))
        for res_block in self.res_blocks1:
            x1 = res_block(x1)
        q1 = self.fc_out1(x1)

        # 计算 Q2
        x2 = F.relu(self.bn4(self.fc4(cat)))
        for res_block in self.res_blocks2:
            x2 = res_block(x2)
        q2 = self.fc_out2(x2)

        return q1, q2

    def Q1(self, state, action):
        cat = torch.cat([state, action], dim=1)
        x1 = F.relu(self.bn1(self.fc1(cat)))
        for res_block in self.res_blocks1:
            x1 = res_block(x1)
        q1 = self.fc_out1(x1)
        return q1