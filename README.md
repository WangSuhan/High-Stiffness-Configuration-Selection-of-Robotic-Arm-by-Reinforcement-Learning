# High-Stiffness-Configuration-Selection-of-Robotic-Arm-by-Reinforcement-Learning

## Introduction

<div style="text-align: justify">
This project is part of my graduate thesis.
Industrial robots provide flexibility and cost advantages for machining complex structural components, but their inherently low and posture-dependent stiffness leads to significant deformations under cutting forces, which compromises machining accuracy. While deformation compensation can improve precision, it suffers from several drawbacks: it requires complex modeling, extensive experimental data collection, and introduces computational overhead during operation. Additionally, compensation models often struggle with real-time adaptation to varying cutting conditions and wear-induced changes. In contrast, selecting high-stiffness robot postures offers a more direct and reliable approach, as it addresses the root cause of inaccuracy by minimizing deformation from the outset rather than correcting it afterward. This preventative strategy simplifies the machining process, reduces the need for sophisticated compensation algorithms, and provides more consistent results across different machining operations and conditions.
</div>
<br>
<div style="text-align: justify">  
From a robotics kinematics perspective, selecting high-stiffness postures for robotic arms constitutes a complex multi-objective optimization problem that requires balancing stiffness performance with other machining requirements under workspace constraints. Traditional optimization approaches include gradient descent, genetic algorithms, particle swarm optimization, and simulated annealing, but these methods often require precise stiffness models and may converge to local optima in high-dimensional spaces. Reinforcement learning offers distinct advantages in addressing this challenge: it eliminates the need for explicit mathematical stiffness models, learning stiffness-posture mappings through environmental interaction; effectively handles nonlinearities and uncertainties in robot dynamics; demonstrates superior adaptability to real-time changes during machining processes; and simultaneously optimizes multiple objectives including stiffness, precision, and energy consumption. This data-driven approach is particularly well-suited for complex, variable machining tasks in modern manufacturing environments, providing a more flexible and efficient pathway to enhancing robotic machining precision.  
</div>

## Method

### Problem Formulation
<div style="text-align: justify">
In robotic machining tasks, we face a multi-objective optimization problem: the robotic arm needs to accurately reach specified positions in Cartesian space while maintaining high-stiffness configurations to resist external forces during machining. Reinforcement learning provides an effective solution to this problem without requiring explicit complex stiffness mathematical models.
</div>

### Stiffness Metrics
<div style="text-align: justify">
The stiffness characteristics of a robotic arm in a specific configuration can be measured using various metrics, such as the minimum eigenvalue of the stiffness matrix, condition number of the stiffness ellipsoid, or determinant of the stiffness matrix. However, these conventional metrics can create training difficulties due to discontinuities and abrupt changes in certain workspace regions.
</div>
<br>
<div style="text-align: justify">
To address this challenge, this work proposes a novel training-friendly stiffness metric that provides smoother gradients and better numerical stability during reinforcement learning. Our metric combines stiffness performance evaluation with mathematical properties conducive to stable optimization, avoiding the extreme value fluctuations that can interrupt the learning process near singularities or workspace boundaries.
</div>

### Reinforcement Learning Framework Design
#### State Space
1. Current joint angles of the robotic arm
2. Target position and orientation
3. Stiffness metrics of the current configuration
#### Action Space
Incremental changes to joint angles, allowing the robotic arm to explore different configurations.
### PyBullet Simulation Environment
please to this guide [https://github.com/WangSuhan/High-Stiffness-Configuration-Selection-of-Robotic-Arm-by-Reinforcement-Learning/blob/main/pybullet20quickstart20guide.479068914.pdf].
### Algorithm Selection
TD3 and DATD3 are used in this project.

## RUN
python -m visdom.server
python main.py run --env=RLReachEnv --algo=DATD3_MLP --vis_name=Reach_DATD3_MLP
