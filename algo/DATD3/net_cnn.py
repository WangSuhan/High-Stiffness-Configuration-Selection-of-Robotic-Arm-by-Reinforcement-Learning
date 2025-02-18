#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class PolicyNet_CNN(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(PolicyNet_CNN, self).__init__()
        self.l1 = nn.Conv2d(in_channels=state_dim[0], out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.l2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.l3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.l4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.fc1 = nn.Linear(in_features=1152, out_features=512, bias=True)
        self.fc_out1 = nn.Linear(in_features=512, out_features=action_dim, bias=True)

        self.max_action = max_action

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a))
        a = torch.tanh(self.l4(a))
        a = a.view(a.size(0), 32 * 6 * 6)
        a = torch.tanh(self.fc1(a))
        a = torch.tanh(self.fc_out1(a))
        return self.max_action * a


class QValueNet_CNN(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QValueNet_CNN, self).__init__()
        # Q1 architecture
        self.l1 = nn.Conv2d(in_channels=state_dim[0], out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.l2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.l3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.l4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.fc1 = nn.Linear(in_features=1152+3, out_features=512, bias=True)
        self.fc_out1 = nn.Linear(in_features=512, out_features=1, bias=True)

        # Q2 architecture
        self.l5 = nn.Conv2d(in_channels=state_dim[0], out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.l6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.l7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.l8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.fc2 = nn.Linear(in_features=1152+3, out_features=512, bias=True)
        self.fc_out2 = nn.Linear(in_features=512, out_features=1, bias=True)

    def forward(self, state, action):
        q1 = torch.tanh(self.l1(state))
        q1 = torch.tanh(self.l2(q1))
        q1 = torch.tanh(self.l3(q1))
        q1 = torch.tanh(self.l4(q1))
        q1 = q1.view(q1.size(0), 32 * 6 * 6)
        action = action.view(action.size(0), 3)
        q1 = torch.cat([q1, action], 1)
        q1 = torch.tanh(self.fc1(q1))
        q1 = self.fc_out1(q1)

        q2 = torch.tanh(self.l5(state))
        q2 = torch.tanh(self.l6(q2))
        q2 = torch.tanh(self.l7(q2))
        q2 = torch.tanh(self.l8(q2))
        q2 = q2.view(q2.size(0), 32 * 6 * 6)
        action = action.view(action.size(0), 3)
        q2 = torch.cat([q2, action], 1)
        q2 = torch.tanh(self.fc2(q2))
        q2 = self.fc_out2(q2)
        return q1, q2

    def Q1(self, state, action):
        q1 = torch.tanh(self.l1(state))
        q1 = torch.tanh(self.l2(q1))
        q1 = torch.tanh(self.l3(q1))
        q1 = torch.tanh(self.l4(q1))
        q1 = q1.view(q1.size(0), 32 * 6 * 6)
        action = action.view(action.size(0), 3)
        q1 = torch.cat([q1, action], 1)
        q1 = torch.tanh(self.fc1(q1))
        q1 = self.fc_out1(q1)
        return q1