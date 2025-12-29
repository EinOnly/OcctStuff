import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import math

# ==========================================
# 1. 配置 MPS (Metal Performance Shaders)
# ==========================================
def get_device():
    if torch.backends.mps.is_available():
        print("✅ 使用 MPS (Apple Silicon) 加速训练")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ 使用 CUDA 加速训练")
        return torch.device("cuda")
    else:
        print("⚠️ 未检测到 GPU，使用 CPU")
        return torch.device("cpu")

DEVICE = get_device()

# ==========================================
# 2. 自定义环境: 模拟你的正弦波追踪任务
# ==========================================
class TorqueTrackingEnv(gym.Env):
    def __init__(self):
        super(TorqueTrackingEnv, self).__init__()
        # 动作空间: RL输出通常在 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # 观察空间: [当前误差, 当前目标值, 上一步动作]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.state = None
        self.time_step = 0
        self.max_steps = 200 # 每个Episode的长度
        
        # 定义物理范围 (关键修复点)
        self.min_torque = 0.0
        self.max_torque = 20.0 # 对应图中 target 的最大值大约是 18

    def reset(self, seed=None):
        self.time_step = 0
        target = self._get_target(0)
        self.state = np.array([0.0, target, 0.0], dtype=np.float32)
        return self.state, {}

    def _get_target(self, t):
        # 模拟图中的绿色虚线: 12 + 6 * sin(...)
        return 12.0 + 6.0 * math.sin(0.1 * t)

    def step(self, action):
        # 1. 动作缩放: 将 [-1, 1] 映射到 [0, 20]
        # 这是你之前失败的主要原因，RL够不到目标值
        scaled_action = self.min_torque + (action[0] + 1.0) * 0.5 * (self.max_torque - self.min_torque)
        
        target = self._get_target(self.time_step)
        
        # 2. 计算误差
        error = target - scaled_action
        
        # 3. 状态更新
        self.time_step += 1
        next_target = self._get_target(self.time_step)
        self.state = np.array([error, next_target, scaled_action], dtype=np.float32)
        
        # 4. 奖励函数 (使用 MSE 的负数，并增加平滑惩罚)
        reward = -(error ** 2) - 0.1 * (action[0] ** 2)
        
        done = self.time_step >= self.max_steps
        
        info = {
            "target": target,
            "output": scaled_action
        }
        
        return self.state, reward, done, False, info

# ==========================================
# 3. 简单的 PPO Agent (Actor-Critic)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor: 决定动作
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh() # 输出 [-1, 1]
        )
        # Critic: 评价状态价值
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
# 4. 实时可视化类
# ==========================================
class LivePlotter:
    def __init__(self):
        plt.ion() # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.set_title("Real-time Training Demo (MPS Accelerated)")
        self.ax.set_ylim(-2, 22)
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Torque (Nm)")
        self.ax.grid(True)
        
        self.line_target, = self.ax.plot([], [], 'g--', label='Target', linewidth=2)
        self.line_output, = self.ax.plot([], [], 'b-', label='RL Output', alpha=0.8)
        self.ax.legend()
        
        self.target_buffer = []
        self.output_buffer = []

    def update(self, target, output):
        self.target_buffer.append(target)
        self.output_buffer.append(output)
        
        # 只显示最近 200 个点
        if len(self.target_buffer) > 200:
            self.target_buffer.pop(0)
            self.output_buffer.pop(0)

        self.line_target.set_data(range(len(self.target_buffer)), self.target_buffer)
        self.line_output.set_data(range(len(self.output_buffer)), self.output_buffer)
        
        self.ax.set_xlim(0, max(200, len(self.target_buffer)))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ==========================================
# 5. 主训练循环
# ==========================================
def train():
    env = TorqueTrackingEnv()
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    plotter = LivePlotter()

    episodes = 200
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        # 临时存储用于可视化
        targets_ep = []
        outputs_ep = []
        
        while not done:
            # 转为 Tensor 并移至 MPS
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            
            # 获取动作
            dist, value = model(state_tensor)
            action = dist.sample()
            
            # 与环境交互 (动作需转回 CPU numpy)
            next_state, reward, done, _, info = env.step(action.cpu().numpy()[0])
            
            # 简单的 Policy Gradient 更新 (简化版 PPO)
            # 注意：正式项目建议使用标准 PPO Buffer 收集一批数据再更新，
            # 这里为了演示实时性，每步都做简单反向传播
            log_prob = dist.log_prob(action)
            loss = -log_prob * reward + 0.5 * (value - reward).pow(2).mean() # 极简 Loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            total_reward += reward
            
            # 收集数据用于绘图
            targets_ep.append(info['target'])
            outputs_ep.append(info['output'])

            # 为了演示效果，每 10 步刷新一次图表，避免太慢
            if env.time_step % 10 == 0:
                plotter.update(info['target'], info['output'])

        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()