#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import torch as t

class DefaultConfig(object):
    # global parameter
    env = 'RLReachEnv'   # env name, need to be the same as envs/__init__.py
    """Choose from RLReachEnv / RLPushEnv / RLPickEnv / RLCamReachEnv"""
    algo = 'TD3_MLP'  # algo name, need to be the same as algo/__init__.py中的名字一致
    """Choose from DDPG_MLP / TD3_MLP / DADDPG_MLP / DATD3_MLP / DARC_MLP / DDPG_CNN / TD3_CNN / DADDPG_CNN / DATD3_CNN / DARC_CNN"""
    vis_name = 'Reach_DADDPG'  # visdom env
    vis_port = '8097'      # visdom port
    jsonfile = "visdata/push/updata_TD3/TD3.json"     # json file dir
    csvname = "visdata/push/updata_TD3/updata_TD3_"   # data save dir

    # reach env parameter
    reach_ctr = 1e-2     # to control the robot arm moving rate every step
    reach_dis = 1e-3     # to control the target distance
    threshold_ratio=1
    balance=0.3
    timeStep = 1.0 / 1e7

    # train parameter
    use_gpu = True       # user GPU or not
    device = t.device('cuda') if use_gpu else t.device('cpu')
    random_seed = 0
    num_episodes = 500   # number of training episodes
    n_train = 40         # number of network updates per episodes
    minimal_episodes = 5  # Minimum number of start rounds for the experience replay buffer
    max_steps_one_episode = 500  # Maximum number of simulation steps per round

    # net parameter
    actor_lr = 1e-4      # actor net learning rate
    critic_lr = 1e-3     # critic net learning rate
    hidden_dim = 256     # mlp hidden size
    batch_size = 256     # batch size
    hidden_layers = 7    #value and policy network layers

    # public algo parameter
    sigma = 0.01          # Standard Deviation of Gaussian Noise
    tau = 0.005          # Target network soft update parameters
    gamma = 0.99         # discount
    buffer_size = 1000000   # buffer size

    # DQN algo only
    epsilon = 0.01
    target_update = 10

    # TD3, DATD3 algo only
    policy_noise = 0.1   # policy noise
    noise_clip = 0.3     # noise clip
    policy_freq = 4      # Delay update frequency

    # DARC algo only
    q_weight = 0.2
    regularization_weight = 0.005

    # HER algo only
    her_ratio = 0.8      # her rate per batch

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        # 打印配置信息
        print('-------------------------------------------------------------------')
        print('==> Printing user config..')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                km = '[' + str(k) + ']'
                print('{:<25}{:<20}'.format(str(km), str(getattr(self, k))))  # {:<30d}含义是 左对齐，且占用30个字符位
        print('-------------------------------------------------------------------')

    def _parsehelp(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        # 打印配置信息
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                km = '[' + str(k) + ']'
                print('            {:<18}--- {:<20}'.format(str(km), str(getattr(self, k))))  # {:<30d}含义是 左对齐，且占用30个字符位


opt = DefaultConfig()