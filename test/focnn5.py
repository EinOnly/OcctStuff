import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import math
import random

# ==========================================
# 1. 配置 MPS
# ==========================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Running on: {DEVICE}")

# ==========================================
# 2. 升级版环境: 带噪声 + 随机目标
# ==========================================
class RobustTorqueEnv(gym.Env):
    def __init__(self):
        super(RobustTorqueEnv, self).__init__()
        # 动作: [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # 观察: [误差, 目标值, 上一步动作]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.max_steps = 200
        self.min_torque = 0.0
        self.max_torque = 20.0
        
        # === 噪声配置 ===
        self.sensor_noise_std = 0.5  # 传感器噪声标准差 (观测噪声)
        self.actuator_noise_std = 0.2 # 执行器噪声标准差 (动作噪声)

    def reset(self, seed=None):
        self.time_step = 0
        
        # === 域随机化 (Domain Randomization) ===
        # 每一轮开始时，随机生成不同的波形，防止死记硬背
        self.target_amp = np.random.uniform(4.0, 7.0)   # 振幅在 [4, 7] 之间随机
        self.target_freq = np.random.uniform(0.08, 0.15) # 频率在 [0.08, 0.15] 之间随机
        self.target_bias = np.random.uniform(10.0, 14.0) # 偏置在 [10, 14] 之间随机
        
        target = self._get_target(0)
        
        # 初始状态也要加噪声
        noisy_target = target + np.random.normal(0, self.sensor_noise_std)
        self.state = np.array([0.0, noisy_target, 0.0], dtype=np.float32)
        
        return self.state, {}

    def _get_target(self, t):
        # 使用随机化后的参数生成目标
        return self.target_bias + self.target_amp * math.sin(self.target_freq * t)

    def step(self, action):
        # 1. 动作缩放
        raw_torque = self.min_torque + (action[0] + 1.0) * 0.5 * (self.max_torque - self.min_torque)
        
        # === 加入执行器噪声 (Actuator Noise) ===
        # 即使 Agent 想输出准确的值，硬件也会有抖动
        actual_torque = raw_torque + np.random.normal(0, self.actuator_noise_std)
        actual_torque = np.clip(actual_torque, self.min_torque, self.max_torque) # 物理限制
        
        # 2. 获取真实目标
        true_target = self._get_target(self.time_step)
        
        # 3. 计算真实误差 (用于奖励，训练它即使在噪声中也要找准方向)
        true_error = true_target - actual_torque
        
        # 4. 状态更新
        self.time_step += 1
        next_true_target = self._get_target(self.time_step)
        
        # === 加入观测噪声 (Sensor Noise) ===
        # Agent 看到的"目标"和"误差"都是带噪声的
        observed_target = next_true_target + np.random.normal(0, self.sensor_noise_std)
        observed_error = true_error + np.random.normal(0, self.sensor_noise_std)
        
        self.state = np.array([observed_error, observed_target, actual_torque], dtype=np.float32)
        
        # 5. 奖励函数
        # 即使有噪声，我们希望 Agent 尽可能接近真实值，所以奖励基于 true_error
        reward = -(true_error ** 2) - 0.05 * (action[0] ** 2)
        
        done = self.time_step >= self.max_steps
        
        info = {
            "target": true_target,
            "output": actual_torque
        }
        
        return self.state, reward, done, False, info

# ==========================================
# 3. Agent (保持不变)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist, value

# ==========================================
# 4. 实时可视化类 (修改版: 滑动窗口放大细节)
# ==========================================
class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.set_title("Real-time Scrolling Scope (Window=50)")
        self.ax.set_ylabel("Torque (Nm)")
        self.ax.grid(True)
        
        # 定义窗口宽度
        self.window_size = 50  
        
        # 初始化数据容器
        self.target_buffer = []
        self.output_buffer = []
        self.time_buffer = [] 

        # 初始化线条
        self.line_target, = self.ax.plot([], [], 'g--', label='Target', linewidth=2)
        self.line_output, = self.ax.plot([], [], 'b-', label='RL Output', alpha=0.7)
        self.ax.legend(loc='upper right')

    def update(self, target, output, t):
        # 1. 追加新数据
        self.target_buffer.append(target)
        self.output_buffer.append(output)
        self.time_buffer.append(t)
        
        # 2. 移除旧数据 (保持窗口固定长度)
        if len(self.time_buffer) > self.window_size:
            self.target_buffer.pop(0)
            self.output_buffer.pop(0)
            self.time_buffer.pop(0)

        # 3. 更新线条数据
        self.line_target.set_data(self.time_buffer, self.target_buffer)
        self.line_output.set_data(self.time_buffer, self.output_buffer)
        
        # 4. 【关键修复】动态移动 X 轴视窗，实现"滚动"效果
        # 如果数据点还不够填满窗口，就显示 0 到 window_size
        # 如果数据点溢出，就显示 (当前时间-窗口) 到 (当前时间)
        left_bound = max(0, t - self.window_size)
        right_bound = max(self.window_size, t)
        
        self.ax.set_xlim(left_bound, right_bound)
        self.ax.set_ylim(-5, 25) # 固定 Y 轴范围，防止跳动

        # 5. 绘图刷新
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ==========================================
# 5. 主训练循环 (无限滚动版)
# ==========================================
def train():
    env = RobustTorqueEnv() # 确保你用的是带噪声的环境类
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 初始化绘图器
    plotter = LivePlotter() 

    episodes = 500
    
    # === 关键修改 1: 定义全局时间步 ===
    global_step = 0 
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # === 关键修改 2: 全局计数器累加 ===
            global_step += 1
            
            # --- 常规 RL 流程开始 ---
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            dist, value = model(state_tensor)
            action = dist.sample()
            
            next_state, reward, done, _, info = env.step(action.cpu().numpy()[0])
            
            log_prob = dist.log_prob(action)
            loss = -log_prob * reward + 0.5 * (value - reward).pow(2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            total_reward += reward
            # --- 常规 RL 流程结束 ---

            # === 关键修改 3: 使用 global_step 绘图 ===
            # 这样 X 轴会一直增加 (0 -> 200 -> 400 ...)，不会回头
            if global_step % 2 == 0: 
                plotter.update(info['target'], info['output'], global_step)

        # 打印日志 (可选)
        print(f"Episode {episode+1}/{episodes}, Global Step: {global_step}, Total Reward: {total_reward:.2f}")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()