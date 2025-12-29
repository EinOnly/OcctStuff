import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import math
from collections import deque
import random

# ==========================================
# 1. 配置 MPS
# ==========================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Running on: {DEVICE}")

# ==========================================
# 2. 真实物理噪声模拟器
# ==========================================
def get_realistic_noise(t, base_freq=0.1):
    """
    模拟真实的电机噪声，包含:
    1. White Noise: 传感器热噪声 (高频随机)
    2. Ripple/Cogging: 机械齿槽转矩 (高频周期性)
    3. Low Freq Drift: 低频漂移
    """
    # 1. 白噪声 (底噪)
    white = np.random.normal(0, 0.3)
    
    # 2. 机械纹波 (Ripple) - 模拟电机高速旋转时的震动
    # 频率通常是基波的几十倍 (比如减速比或极对数)
    ripple_freq = base_freq * 30.0 
    ripple = 0.8 * math.sin(ripple_freq * t) * math.cos(ripple_freq * 1.5 * t)
    
    # 3. 尖峰噪声 (偶尔出现的跳变)
    spike = 0.0
    if random.random() > 0.95: # 5% 的概率出现跳变
        spike = np.random.uniform(-1.5, 1.5)
        
    return white + ripple + spike

# ==========================================
# 3. 核心升级: 支持时序堆叠的环境
# ==========================================
class SequenceTorqueEnv(gym.Env):
    def __init__(self, history_len=10):
        super(SequenceTorqueEnv, self).__init__()
        
        self.history_len = history_len # 过去 10 帧
        self.feature_dim = 3           # [Error, Target, Last_Action]
        
        # === 关键修改: 状态空间变大了 ===
        # 输入维度 = 10 * 3 = 30 维
        flat_dim = self.history_len * self.feature_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.max_steps = 300
        self.min_torque = 0.0
        self.max_torque = 20.0
        
        # 历史缓冲区
        self.state_buffer = deque(maxlen=self.history_len)

    def reset(self, seed=None):
        self.time_step = 0
        
        # 域随机化
        self.target_amp = np.random.uniform(4.0, 7.0)
        self.target_freq = np.random.uniform(0.05, 0.12)
        self.target_bias = np.random.uniform(10.0, 14.0)
        
        # 初始化 buffer: 用第一帧填满 10 次，避免冷启动为空
        initial_target = self._get_target(0)
        initial_feat = np.array([0.0, initial_target, 0.0], dtype=np.float32)
        
        self.state_buffer.clear()
        for _ in range(self.history_len):
            self.state_buffer.append(initial_feat)
            
        return self._get_stacked_state(), {}

    def _get_target(self, t):
        return self.target_bias + self.target_amp * math.sin(self.target_freq * t)

    def _get_stacked_state(self):
        # 将 deque 中的 10 帧数据拼成一个 30 维的长向量
        # [frame1, frame2, ... frame10]
        return np.concatenate(self.state_buffer)

    def step(self, action):
        # 1. 动作映射
        raw_torque = self.min_torque + (action[0] + 1.0) * 0.5 * (self.max_torque - self.min_torque)
        
        # === 关键修改: 加入高频真实噪声 ===
        # 电机输出会有纹波
        actuator_noise = get_realistic_noise(self.time_step, self.target_freq) * 0.5
        actual_torque = np.clip(raw_torque + actuator_noise, self.min_torque, self.max_torque)
        
        # 2. 物理演化
        target = self._get_target(self.time_step)
        true_error = target - actual_torque
        
        # 3. 推进时间
        self.time_step += 1
        next_target = self._get_target(self.time_step)
        
        # === 关键修改: 观测噪声 ===
        # 传感器读数也有高频噪声
        sensor_noise = get_realistic_noise(self.time_step, self.target_freq) * 0.8
        obs_error = true_error + sensor_noise
        obs_target = next_target + sensor_noise
        
        # 4. 更新历史 Buffer
        new_feat = np.array([obs_error, obs_target, actual_torque], dtype=np.float32)
        self.state_buffer.append(new_feat)
        
        # 5. 奖励 (即使在重噪声下也要尽可能稳)
        reward = -(true_error ** 2) - 0.1 * (action[0] ** 2)
        
        done = self.time_step >= self.max_steps
        info = {"target": target, "output": actual_torque}
        
        return self._get_stacked_state(), reward, done, False, info

# ==========================================
# 4. 网络结构适配 (Input Dim = 30)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 因为输入变大了 (3->30)，第一层要做宽一点
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),  # 30 -> 128
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),  # 30 -> 128
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist, value

# ==========================================
# 5. 绘图类 (保持无限滚动)
# ==========================================
class LivePlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.set_title("10-Frame Sequence Input & High-Freq Torque Ripple")
        self.ax.set_ylabel("Torque (Nm)")
        self.ax.grid(True, alpha=0.3)
        self.window_size = 60 # 视野窗口
        
        self.target_buffer, self.output_buffer, self.time_buffer = [], [], []
        self.line_target, = self.ax.plot([], [], 'g--', label='Target (Smooth)', linewidth=2)
        self.line_output, = self.ax.plot([], [], 'b-', label='RL Output (Real Noise)', alpha=0.8, linewidth=1)
        self.ax.legend(loc='upper right')

    def update(self, target, output, t):
        self.target_buffer.append(target)
        self.output_buffer.append(output)
        self.time_buffer.append(t)
        
        if len(self.time_buffer) > self.window_size:
            self.target_buffer.pop(0)
            self.output_buffer.pop(0)
            self.time_buffer.pop(0)

        self.line_target.set_data(self.time_buffer, self.target_buffer)
        self.line_output.set_data(self.time_buffer, self.output_buffer)
        
        left = max(0, t - self.window_size)
        right = max(self.window_size, t)
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(-2, 25)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ==========================================
# 6. 主程序
# ==========================================
def train():
    # 初始化环境 (History=10)
    env = SequenceTorqueEnv(history_len=10)
    
    # 自动计算维度: 30 -> 1
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=3e-4) # 稍微调低 LR 以适应复杂噪声
    plotter = LivePlotter()
    
    episodes = 500
    global_step = 0
    best_score = -float('inf')

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            global_step += 1
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            
            dist, value = model(state_tensor)
            action = dist.sample()
            
            next_state, reward, done, _, info = env.step(action.cpu().numpy()[0])
            
            # Loss Calculation
            log_prob = dist.log_prob(action)
            loss = -log_prob * reward + 0.5 * (value - reward).pow(2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            total_reward += reward
            
            # 绘图: 每 2 步刷新，观察高频噪声
            if global_step % 2 == 0:
                plotter.update(info['target'], info['output'], global_step)

        # 简单的进度条
        print(f"Ep {episode+1}, Reward: {total_reward:.1f}")
        
        if total_reward > best_score:
            best_score = total_reward
            # torch.save(model.state_dict(), "best_seq_model.pth") 

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()