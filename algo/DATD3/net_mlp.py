import torch
import torch.nn.functional as F
import torch.nn as nn

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.hidden_dim = hidden_dim * 2
        self.action_bound = action_bound

        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        # 逆运动学网络
        self.ik_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Lambda(lambda x: torch.sin(x)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 7)
        )

        # 刚度网络
        self.stiffness_net = nn.Sequential(
            nn.Linear(7, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Lambda(lambda x: x + 0.5 * x ** 2 + 0.1 * x ** 3),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # 动作生成网络
        self.action_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_dim)
        )

        # 刚度评估器
        self.stiffness_evaluator = nn.Sequential(
            nn.Linear(7, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, evaluate_stiffness=False):
        # 基础特征提取
        basic_features = self.feature_net(state)

        # 生成关节角配置
        joint_angles = self.ik_net(basic_features)

        # 计算刚度特征
        stiffness_features = self.stiffness_net(joint_angles)

        # 特征融合
        combined_features = torch.cat([basic_features, stiffness_features], dim=1)

        # 生成动作
        action = self.action_net(combined_features)
        action = torch.tanh(action) * self.action_bound

        if evaluate_stiffness:
            current_stiffness = self.stiffness_evaluator(joint_angles)
            return action, current_stiffness
        return action


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.hidden_dim = hidden_dim * 2

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, hidden_dim)
        )

        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, hidden_dim)
        )

        # Q值预测器
        self.q_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(action)

        combined = torch.cat([state_features, action_features], dim=1)
        q_value = self.q_predictor(combined)

        return q_value


class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)