import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. 物理环境模拟 (The Plant)
# ==========================================
class MotorSystemEnv:
    def __init__(self):
        self.time_step = 0
        self.dt = 0.01
        
    def f_t(self, t):
        """
        模拟波动函数 f(t)。
        包含：基准传动效率 + 周期性谐波干扰(章动) + 随机噪声
        """
        # 基础效率 0.85
        base_efficiency = 0.85 
        # 谐波干扰 (模拟减速机特有的周期性误差)
        harmonic_ripple = 0.10 * np.sin(2 * np.pi * 1.5 * t) + \
                          0.05 * np.cos(2 * np.pi * 3.0 * t)
        # 随机噪声
        noise = np.random.normal(0, 0.01)
        
        return base_efficiency + harmonic_ripple + noise

    def step(self, ta0_command):
        """
        系统执行一步
        Input: TA0 (FOC指令)
        Output: TA2 (传感器读数)
        """
        current_t = self.time_step * self.dt
        fluctuation = self.f_t(current_t)
        
        # 物理关系: TA2 = f(t) * TA0
        ta2_actual = ta0_command * fluctuation
        
        self.time_step += 1
        return ta2_actual, current_t

# ==========================================
# 2. 强化学习/神经网络代理 (FOCNN)
# ==========================================
class FOCNN(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super(FOCNN, self).__init__()
        # 输入: 过去一段时间的观测值 [Target, Error_History...]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出: 修正系数 k, 使得 TA0 = Target * k
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 训练与推理主循环
# ==========================================

def run_simulation():
    # 初始化
    env = MotorSystemEnv()
    # 输入特征: [目标转矩, 正弦相位, 余弦相位] (引入相位是为了让NN更容易拟合周期性干扰)
    input_dim = 3 
    agent = FOCNN(input_dim=input_dim)
    optimizer = optim.Adam(agent.parameters(), lr=0.005)
    
    # 记录数据用于绘图
    history = {
        'time': [],
        'target': [],
        'ta0_cmd': [],
        'ta2_actual': [],
        'loss': []
    }
    
    # 设定测试参数
    total_steps = 1000
    target_torque = 10.0 # 我们希望输出恒定 10Nm
    
    print("开始系统仿真与在线训练...")
    
    # --- 模拟时间流 ---
    for step in range(total_steps):
        t = step * env.dt
        
        # --- A. 状态构建 (Input) ---
        # 我们输入目标值和当前的时间相位信息(因为干扰是周期性的)
        # 实际工程中，这里会输入Encoder的角度位置
        state = torch.FloatTensor([
            target_torque, 
            np.sin(2 * np.pi * 1.5 * t), 
            np.cos(2 * np.pi * 1.5 * t)
        ]).unsqueeze(0)
        
        # --- B. 代理决策 (Predict) ---
        # Agent 输出一个补偿系数 k
        compensation_factor = agent(state) 
        
        # 实际上 TA0 = Target * Compensation
        # 如果系统完美，k应该是 1/efficiency。Agent需要学会这个值。
        ta0_cmd = target_torque * compensation_factor
        
        # --- C. 环境反馈 (Step) ---
        ta0_val = ta0_cmd.item()
        ta2_actual, _ = env.step(ta0_val)
        
        # --- D. 计算误差与反向传播 (Learn) ---
        # 我们的目标是让 ta2_actual 接近 target_torque
        # Loss = (TA2 - Target)^2
        # 注意: 这里用了一种简化的梯度估计，即假设 PID/物理梯度方向已知
        # 在纯黑盒RL中需要用Policy Gradient，但这里为了演示收敛性，
        # 我们利用 ta2 = ta0 * f(t) 的线性性质来构建Loss。
        
        # 将 ta2 转化回 tensor 用于求导 (模拟 differentiable physics)
        # 实际上 agent 输出的是 compensation_factor
        # pred_ta2 = target * compensation * f(t) (f(t) 未知)
        # 我们简化为：我们要训练 agent 输出的 factor 能够抵消掉刚才观测到的衰减
        
        # 真实的 Reward/Loss 计算:
        loss = nn.MSELoss()(torch.tensor([ta2_actual], dtype=torch.float32, requires_grad=True), 
                            torch.tensor([target_torque], dtype=torch.float32))
        
        # 这里的关键 Trick: 
        # 这是一个在线控制问题。我们用当前的误差去更新网络，让它下次预测得更准。
        # 目标系数 target_k = Target / (TA2 / Factor_old) = Target / (f(t)*Target) * Factor_old ???
        # 简化版监督学习：如果刚才 TA2 小了，说明 Factor 要变大。
        
        actual_ratio = ta2_actual / ta0_val # 这是当前的 f(t)
        ideal_factor = 1.0 / actual_ratio   # 如果我们知道 f(t)，我们就该乘这个
        
        # 训练网络去逼近这个 ideal_factor
        target_label = torch.FloatTensor([[ideal_factor]])
        real_loss = nn.MSELoss()(compensation_factor, target_label)
        
        optimizer.zero_grad()
        real_loss.backward()
        optimizer.step()
        
        # --- E. 记录 ---
        history['time'].append(t)
        history['target'].append(target_torque)
        history['ta0_cmd'].append(ta0_val)
        history['ta2_actual'].append(ta2_actual)
        history['loss'].append(real_loss.item())

    # ==========================================
    # 4. 可视化结果
    # ==========================================
    plt.figure(figsize=(12, 10))
    
    # 图1: 转矩跟踪效果
    plt.subplot(3, 1, 1)
    plt.plot(history['time'], history['target'], 'g--', label='Target (Ideal)', linewidth=2)
    plt.plot(history['time'], history['ta2_actual'], 'b-', label='TA2 (Actual Output)', alpha=0.7)
    plt.title('Torque Tracking Performance (Online Learning)')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid(True)
    
    # 图2: FOCNN 输出的补偿指令
    plt.subplot(3, 1, 2)
    plt.plot(history['time'], history['ta0_cmd'], 'r-', label='TA0 (FOC Command)')
    plt.title('FOCNN Control Action (Adaptive Compensation)')
    plt.ylabel('Command Torque')
    plt.legend()
    plt.grid(True)
    
    # 图3: 训练误差收敛
    plt.subplot(3, 1, 3)
    plt.plot(history['time'], history['loss'], 'k-', label='Prediction Loss')
    plt.title('Learning Convergence')
    plt.xlabel('Time (s)')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()