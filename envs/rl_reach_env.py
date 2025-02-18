#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pybullet as p
import pybullet_data
import os
import gym
from gym import spaces
from gym.utils import seeding
import random
import time
import math
from config import opt
import Jacobian as JACOBI


class RLReachEnv(gym.Env):
    """创建强化学习机械臂reach任务仿真环境"""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self, is_render=False, is_good_view=True):
        """
        用于初始化reach环境中的各项参数，

        Args:
            is_render (bool):       是否创建场景可视化
            is_good_view (bool):    是否创建更优视角

        Returns:
            None
        """
        self.is_render = is_render
        self.is_good_view = is_good_view
        self.max_steps_one_episode = opt.max_steps_one_episode

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setTimeStep(opt.timeStep)
        # p.setPhysicsEngineParameter(contactBreakingThreshold=0.0001)
        # 机械臂移动范围限制
        self.x_low_obs = 0.2
        self.x_high_obs = 0.6
        self.y_low_obs = -0.25
        self.y_high_obs = 0.25
        self.z_low_obs = 0
        self.z_high_obs = 0.55
        self.euler_low=-math.pi
        self.euler_high = math.pi

        # 机械臂动作范围限制
        self.x_low_action = 0.2
        self.x_high_action = 0.6
        self.y_low_action = -0.25
        self.y_high_action = 0.25
        self.z_low_action = 0
        self.z_high_action = 0.55

        # 设置相机
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([self.x_low_action, self.y_low_action, self.z_low_action, self.euler_low, self.euler_low, self.euler_low]),
            high=np.array([self.x_high_action, self.y_high_action, self.z_high_action, self.euler_high, self.euler_high, self.euler_high]),
            dtype=np.float32)

        # 状态空间
        self.observation_space = spaces.Box(
            low=np.array([self.x_low_obs, self.y_low_obs, self.z_low_obs, self.euler_low, self.euler_low, self.euler_low]),
            high=np.array([self.x_high_obs, self.y_high_obs, self.z_high_obs, self.euler_high, self.euler_high, self.euler_high]),
            dtype=np.float32)

        # 时间步计数器
        self.step_counter = 0

        self.urdf_root_path = pybullet_data.getDataPath()
        print(self.urdf_root_path)
        # # lower limits for null space
        # self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # # upper limits for null space
        # self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # # joint ranges for null space
        # self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # # restposes for null space
        # self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # # joint damping coefficents
        # self.joint_damping = [
        #     0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        # ]
        #
        # # 初始关节角度
        # self.init_joint_positions = [
        #     0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
        #     -0.006539
        # ]
        # lower limits for null space
        self.lower_limits = [-math.pi, -math.pi/180*215, -math.pi/180*50, -math.pi, -math.pi/180*130, -math.pi]
        # upper limits for null space
        self.upper_limits = [math.pi, math.pi/180*35, math.pi/180*230, math.pi, math.pi/180*130, math.pi]
        # joint ranges for null space
        self.joint_ranges = [2*math.pi, math.pi/180*250, math.pi/180*280, 2*math.pi, math.pi/180*260, 2*math.pi]
        # restposes for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66]
        # joint damping coefficents
        self.joint_damping = [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ]

        # 初始关节角度
        # self.init_joint_positions = [
        #     0, -math.pi/2, math.pi, 0, math.pi/2, 0
        # ]
        self.init_joint_positions = [
            0, -math.pi/2, math.pi, 0, math.pi/2, 0
        ]

        # alpha=random.uniform(-math.pi, math.pi)
        # beta=random.uniform(-math.pi, math.pi)
        # gamma=random.uniform(-math.pi, math.pi)
        self.orientation = p.getQuaternionFromEuler(
             [0, 0, -math.pi])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        """随机种子"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """环境reset，获得初始state"""
        # alpha=random.uniform(-math.pi, math.pi)
        # beta=random.uniform(-math.pi, math.pi)
        # gamma=random.uniform(-math.pi, math.pi)
        # self.orientation = p.getQuaternionFromEuler(
        #     [alpha, beta, gamma])
        # 初始化时间步计数器
        self.step_counter = 0

        p.resetSimulation()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # 初始化重力以及运行结束标志
        self.terminated = False
        p.setGravity(0, 0, -10)

        # 状态空间的限制空间可视化，以白线标识
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        # 载入平面
        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"), basePosition=[0, 0, -0.65])
        # 载入机械臂
        # self.kuka_id = p.loadURDF(os.path.join(self.urdf_root_path, "kuka_iiwa/model.urdf"), useFixedBase=True)
        self.kuka_id = p.loadURDF(os.path.join("D:/DRL_Diana_robot_arm/hsr_description/urdf", "hsr_co603.urdf"), useFixedBase=True)
        # 载入桌子
        p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        # p.loadURDF(os.path.join(self.urdf_root_path, "tray/traybox.urdf"),basePosition=[0.55,0,0])
        # object_id=p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=[0.53,0,0.02])

        xpos = random.uniform(self.x_low_obs, self.x_high_obs)
        ypos = random.uniform(self.y_low_obs, self.y_high_obs)
        zpos = random.uniform(self.z_low_obs, self.z_high_obs) # TODO 原z=0.01
        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        # 载入物体
        self.object_id = p.loadURDF("../models/cube_small_target_push.urdf",
                                    basePosition=[xpos, ypos, zpos],
                                    baseOrientation=[orn[0], orn[1], orn[2], orn[3]],
                                    useFixedBase=1)
        # 关节角初始化
        self.num_joints = p.getNumJoints(self.kuka_id)
        # print(self.num_joints)

        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )
        # for i in range(self.num_joints):
        #      print(p.getJointInfo(self.kuka_id, i))

        self.robot_pos_obs = p.getLinkState(self.kuka_id,
                                            self.num_joints - 1)[4]
        # print(self.robot_pos_obs)
        # logging.debug("init_pos={}\n".format(p.getLinkState(self.kuka_id,self.num_joints-1)))
        p.stepSimulation()
        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        self.object_pos = self.robot_pos_obs

        goal = [random.uniform(self.x_low_obs, self.x_high_obs),
                random.uniform(self.y_low_obs, self.y_high_obs),
                random.uniform(self.z_low_obs, self.z_high_obs)]
        self.object_state = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
            np.float32)
        # return np.array(self.object_pos).astype(np.float32), self.object_state
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.euler_angles = p.getEulerFromQuaternion(p.getLinkState(self.kuka_id, self.num_joints - 1)[5])
        return np.hstack((np.array(self.robot_state).astype(np.float32), self.euler_angles, self.object_state))
        # return np.hstack((np.array(self.robot_state).astype(np.float32), self.object_state))

    def step(self, action):
        """根据action获取下一步环境的state、reward、done"""
        limit_x = [0.2, 0.7]
        limit_y = [-0.3, 0.3]
        limit_z = [0, 0.55]
        limit_ang=[-math.pi,math.pi]

        def clip_val(val, limit):
            if val < limit[0]:
                return limit[0]
            if val > limit[1]:
                return limit[1]
            return val
        # if np.isnan(action[0]):
        if False:
            x=1
        #     action=np.array([0.9,0.8,0.9,0.5,0.6,0.7])
        #     joint_positions = [random.uniform(lower, upper) for lower, upper in zip(self.lower_limits, self.upper_limits)]
        #     for i, joint_position in enumerate(joint_positions):
        #         p.setJointMotorControl2(self.kuka_id, i, p.POSITION_CONTROL, targetPosition=joint_position)
        #     p.stepSimulation()
        else:
            dv = opt.reach_ctr
            dx = action[0] * dv
            dy = action[1] * dv
            dz = action[2] * dv
            dalpha = action[3] * dv
            dbeta = action[4] * dv
            dgamma = action[5] * dv
            # print(action)

            # 获取当前机械臂末端坐标
            self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
            self.current_ori =p.getEulerFromQuaternion(p.getLinkState(self.kuka_id, self.num_joints - 1)[5])
            # print(self.current_pos,self.current_ori,action)
            # 计算下一步的机械臂末端坐标
            self.new_robot_pos = [
                clip_val(self.current_pos[0] + dx, limit_x), clip_val(self.current_pos[1] + dy, limit_y),
                clip_val(self.current_pos[2] + dz, limit_z), clip_val(self.current_ori[0] + dalpha, limit_ang),
                clip_val(self.current_ori[1] + dbeta, limit_ang), clip_val(self.current_ori[2] + dgamma, limit_ang)
            ]
            self.quaternion = p.getQuaternionFromEuler([self.new_robot_pos[3], self.new_robot_pos[4], self.new_robot_pos[5]])

            # 通过逆运动学计算机械臂移动到新位置的关节角度
            self.robot_joint_positions = p.calculateInverseKinematics(
                bodyUniqueId=self.kuka_id,
                endEffectorLinkIndex=self.num_joints - 1,
                targetPosition=[self.new_robot_pos[0], self.new_robot_pos[1], self.new_robot_pos[2]],
                targetOrientation=self.quaternion,
                jointDamping=self.joint_damping,
                # solver=p.IK_SDLS,
                # maxNumIterations=100,  # 最大迭代次数
                # residualThreshold=1e-5  # 残差阈值
            )
            # print(self.new_robot_pos,self.robot_joint_positions)
            # 使机械臂移动到新位置
            for i in range(self.num_joints):
                p.resetJointState(
                    bodyUniqueId=self.kuka_id,
                    jointIndex=i,
                    targetValue=self.robot_joint_positions[i],
                )

            p.stepSimulation()



        # 在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1
        return self._reward()

    def _reward(self):
        """根据state计算当前的reward"""
        # 获取机械臂当前的末端坐标
        # 一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.euler_angles = p.getEulerFromQuaternion(p.getLinkState(self.kuka_id, self.num_joints - 1)[5])
        # self.result = p.getLinkState(self.kuka_id, self.num_joints - 1)
        # link_trn, link_rot, self.com_trn, com_rot, frame_pos, frame_rot=self.result
        # #计算雅可比矩阵jac
        # self.joint_positions = [state[0] for state in self.result]
        # zero_vec=[0,0,0,0,0,0]
        # zero_acc=[0]*len(self.joint_positions)
        # jac_t, jac_r=p.calculateJacobian(self.kuka_id,self.num_joints - 1,self.com_trn,self.joint_positions,zero_vec,zero_acc)
        # jac_t=np.array(jac_t)*1000
        # jac_r=np.array(jac_r)
        # self.jac=np.vstack((jac_t,jac_r))
        # print(self.jac)
        self.joint_states = p.getJointStates(self.kuka_id, range(self.num_joints))
        self.theta=np.array([item[0] for item in self.joint_states])
        self.angle_velocity_sum=np.sum(np.array([item[1] for item in self.joint_states]))
        # print(self.angle_velocity_sum)
        # print(self.theta)
        m = np.array([321.5545, 300.7647, 300.4592, 205])
        self.Jf, self.Tf, self.Jm = JACOBI.compute_jacobi(self.theta.copy(), m)
        #计算顺应矩阵C和刚度指标ks
        K=np.array([10993208.05, 20007447.52, 14528020.43, 6669204.284,	4593508.441, 3769342.804])
        self.C=self.Jf*np.linalg.inv(np.diag(K))*np.transpose(self.Jf)
        self.Ctt=self.C[:3,:3]
        # self.cs=np.linalg.det(self.Ctt)**(1/3)
        eigenvalues, eigenvectors = np.linalg.eig(self.Ctt)
        self.cs=np.linalg.norm(self.Ctt*eigenvectors,'fro')
        # print(self.cs)
        # print(self.theta,self.Jf)

        # self.object_state=p.getBasePositionAndOrientation(self.object_id)
        # self.object_state=np.array(self.object_state).astype(np.float32)

        # 获取物体当前的位置坐标
        self.object_state = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
                np.float32)

        # 用机械臂末端和物体的距离作为奖励函数的依据
        self.distance = np.linalg.norm(self.robot_state - self.object_state, axis=-1)
        # print(self.distance)

        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]

        # 如果机械比末端超过了obs的空间，也视为done，而且会给予一定的惩罚
        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)

        # if terminated:
        #     reward = -50.0
        #     self.terminated = True
        self.is_success = False
        self.cs_history=[]
        # 如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        #距离的量级在0.3左右
        # if self.step_counter > self.max_steps_one_episode:
        #     reward = -self.distance*10
        #     self.terminated = True
        #
        # elif self.distance < opt.reach_dis:
        #     reward=0
        #     self.terminated = True
        #     self.is_success = True
        # else:
        #     reward = - self.distance * 10  # -0.1
        #     self.terminated = False

        if self.step_counter > self.max_steps_one_episode :
            reward = -self.distance
            self.terminated = True
        elif self.distance < opt.reach_dis:
            reward = 100*self.cs
            self.terminated = True
            self.is_success = True
        else:
            reward = -self.distance
            self.terminated = False
        info = {'distance:', self.distance}
        # self.observation=self.robot_state
        self.observation = self.robot_state
        # self.observation = [p.getLinkState(self.kuka_id, self.num_joints - 1)[4],p.getEulerFromQuaternion(p.getLinkState(self.kuka_id, self.num_joints - 1)[5])]
        # self.observation = np.hstack([np.array(vec) for vec in self.observation])

        goal = [random.uniform(self.x_low_obs,self.x_high_obs),
                random.uniform(self.y_low_obs,self.y_high_obs),
                random.uniform(self.z_low_obs, self.z_high_obs)]
        return np.hstack((np.array(self.observation).astype(np.float32), self.euler_angles, self.object_state)), reward, self.terminated, self.is_success
        # return np.hstack((np.array(self.observation).astype(np.float32), self.object_state)), reward, self.terminated, self.is_success

    def close(self):
        p.disconnect()



if __name__ == '__main__':
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数
    env = RLReachEnv(is_good_view=True, is_render=True)
    print('env={}'.format(env))
    print(env.observation_space.shape)
    # print(env.observation_space.sample())
    # print(env.action_space.sample())
    print(env.action_space.shape)
    obs = env.reset()
    # print(Fore.RED + 'obs={}'.format(obs))
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print('obs={},reward={},done={}'.format(obs, reward, done))

    sum_reward = 0
    success_times = 0
    for i in range(100):
        env.reset()
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print('reward={},done={}'.format(reward, done))
            sum_reward += reward
            if reward == 1:
                success_times += 1
            if done:
                break
        # time.sleep(0.1)
    print()
    print('sum_reward={}'.format(sum_reward))
    print('success rate={}'.format(success_times / 50))