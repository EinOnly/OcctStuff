import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time

# ==========================================
# 1. 物理环境模拟 (The Plant) - 保持不变
# ==========================================
class MotorSystemEnv:
    def __init__(self):
        self.time_step = 0
        self.dt = 0.02 # 稍微增加时间步长，让动画更流畅
        
    def f_t(self, t):
        # 基础效率
        base_efficiency = 0.85 
        # 谐波干扰 (高频波动)
        harmonic_ripple = 0.10 * np.sin(2 * np.pi * 1.5 * t) + \
                          0.05 * np.cos(2 * np.pi * 3.0 * t)
        # 随机噪声
        noise = np.random.normal(0, 0.008)
        
        return base_efficiency + harmonic_ripple + noise

    def step(self, ta0_command):
        current_t = self.time_step * self.dt
        fluctuation = self.f_t(current_t)
        ta2_actual = ta0_command * fluctuation
        self.time_step += 1
        return ta2_actual, current_t

# ==========================================
# 2. 神经网络代理 (FOCNN) - 保持不变
# ==========================================
class FOCNN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super(FOCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 动态实时仿真主循环
# ==========================================
def run_dynamic_simulation():
    # --- A. 初始化系统 ---
    env = MotorSystemEnv()
    # 输入特征: [目标转矩, 相位sin, 相位cos]
    input_dim = 3 
    agent = FOCNN(input_dim=input_dim)
    # 使用较大的学习率，以便在演示中更快看到适应过程
    optimizer = optim.Adam(agent.parameters(), lr=0.015) 
    
    # 用于存储历史数据以供绘图
    time_data, target_data, ta0_data, ta2_data, loss_data = [], [], [], [], []
    
    # --- B. 初始化动态图表 ---
    plt.ion() # 开启交互模式 (Interactive Mode)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 图1: 转矩跟踪 (核心观察区)
    line_target, = ax1.plot([], [], 'g--', label='Target (Ideal)', linewidth=2.5)
    line_ta2, = ax1.plot([], [], 'b-', label='TA2 (Actual Output)', linewidth=1.5, alpha=0.8)
    ax1.set_title('Real-time Torque Tracking (Watch change after 5s!)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Torque (Nm)')
    ax1.set_ylim(4, 16) # 设置合适的Y轴范围
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # 图2: FOCNN 控制输出
    line_ta0, = ax2.plot([], [], 'r-', label='TA0 (FOCNN Command)')
    ax2.set_title('FOCNN Adaptive Control Action')
    ax2.set_ylabel('Command Torque')
    ax2.set_ylim(4, 20)
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # 图3: 学习误差
    line_loss, = ax3.plot([], [], 'k-', label='Learning Loss')
    ax3.set_title('Online Learning Convergence')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Loss (log scale)')
    ax3.set_yscale('log')
    ax3.set_ylim(1e-5, 1.0)
    ax3.grid(True)
    
    plt.tight_layout()
    
    # 仿真参数
    total_time = 15.0 # 总时长 15 秒
    total_steps = int(total_time / env.dt)
    
    print("准备就绪。仿真即将开始...")
    print("-> 0-5s: 恒定目标热身")
    print("-> 5s后: 目标改变，观察模型的泛化跟随能力！")
    time.sleep(2) 
    
    # --- C. 主循环 ---
    for step in range(total_steps):
        t = step * env.dt
        
        # ========= 关键点：改变目标信号 =========
        if t < 5.0:
            # 前5秒：恒定 10Nm
            target_torque = 10.0
        else:
            # 5秒后：引入一个新的慢速正弦波目标 (8Nm - 12Nm)
            # 这是模型之前没见过的工况
            target_torque = 10.0 + 2.0 * np.sin(2 * np.pi * 0.3 * (t - 5.0))
        # =====================================

        # 1. 构建状态并预测
        state = torch.FloatTensor([
            target_torque, 
            np.sin(2 * np.pi * 1.5 * t), 
            np.cos(2 * np.pi * 1.5 * t)
        ]).unsqueeze(0)
        
        compensation_factor = agent(state) 
        ta0_cmd = target_torque * compensation_factor
        
        # 2. 环境执行
        ta0_val = ta0_cmd.item()
        ta2_actual, _ = env.step(ta0_val)
        
        # 3. 在线学习 (计算理想补偿系数并反向传播)
        # 防止除零的小保护
        actual_ratio = ta2_actual / (ta0_val + 1e-6)
        ideal_factor = 1.0 / (actual_ratio + 1e-6)
        
        target_label = torch.FloatTensor([[ideal_factor]])
        # 增加一点正则化项防止过拟合
        loss_fn = nn.MSELoss()
        real_loss = loss_fn(compensation_factor, target_label)
        
        optimizer.zero_grad()
        real_loss.backward()
        optimizer.step()
        
        # 4. 记录数据
        time_data.append(t)
        target_data.append(target_torque)
        ta0_data.append(ta0_val)
        ta2_data.append(ta2_actual)
        loss_data.append(real_loss.item())
        
        # --- D. 动态绘图更新 ---
        # 每隔几步更新一次，避免绘图过于频繁卡顿
        if step % 3 == 0:
            # 设置一个滚动的X轴窗口 (显示最近5秒的数据)
            window_start = max(0, t - 5.0)
            ax1.set_xlim(window_start, t + 0.5)
            ax2.set_xlim(window_start, t + 0.5)
            ax3.set_xlim(window_start, t + 0.5)
            
            # 更新线条数据
            line_target.set_data(time_data, target_data)
            line_ta2.set_data(time_data, ta2_data)
            line_ta0.set_data(time_data, ta0_data)
            line_loss.set_data(time_data, loss_data)
            
            # 刷新图表
            fig.canvas.draw()
            fig.canvas.flush_events()
            
    plt.ioff() # 关闭交互模式
    print("仿真结束。")
    plt.show() # 保持最后一帧图像

if __name__ == "__main__":
    run_dynamic_simulation()