import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ==========================================
# 1. 模拟环境：带有延迟的电机系统 (The Plant)
# ==========================================
class DelayedMotorSystem:
    def __init__(self, delay_step=20, inertia=0.95):
        self.delay_step = delay_step
        self.inertia = inertia
        # 这是一个 FIFO 队列，用来模拟物理传输延迟
        self.delay_buffer = deque([0.0] * delay_step, maxlen=delay_step)
        self.current_val = 0.0

    def step(self, command_torque):
        # 1. 指令进入延迟管道
        self.delay_buffer.append(command_torque)
        
        # 2. 从延迟管道取出 N 步之前的指令（模拟 20ms 后的生效）
        delayed_command = self.delay_buffer[0]
        
        # 3. 简单的物理响应 (一阶惯性环节)
        # y[t] = a * y[t-1] + (1-a) * u[t-delay]
        self.current_val = self.inertia * self.current_val + (1 - self.inertia) * delayed_command
        return self.current_val

# ==========================================
# 2. 控制器：FOCNN (模拟在线自适应)
# ==========================================
class AdaptiveController:
    def __init__(self, learning_rate=0.01, delay_step=20, delta=0.5, beta=1.5):
        self.lr = learning_rate
        self.weights = 1.0 # 模拟神经网络的权重 (这里简化为一个增益)
        
        # --- 关键点 1: 延迟对齐 Buffer ---
        # 我们需要存储发出的指令，以便在 20ms 后拿出来计算梯度
        self.command_memory = deque([0.0] * delay_step, maxlen=delay_step)
        self.delay_step = delay_step
        
        # --- 关键点 2: 阈值参数 ---
        self.delta = delta # 小阈值 (进入休眠)
        self.beta = beta   # 大阈值 (激活学习)
        self.is_active = False # 当前状态
        
        # 用于记录 debug 信息
        self.loss_history = []
        self.active_history = []

    def _check_hysteresis(self, error):
        abs_err = abs(error)
        
        # 状态机逻辑
        if not self.is_active:
            # 如果当前是休眠，只有误差大于 Beta 才激活
            if abs_err > self.beta:
                self.is_active = True
        else:
            # 如果当前是激活，只有误差小于 Delta 才休眠
            if abs_err < self.delta:
                self.is_active = False
                
        return self.is_active

    def update(self, target, actual, current_cmd):
        error = target - actual
        
        # 1. 将当前的指令存入记忆，供未来训练使用
        self.command_memory.append(current_cmd)
        
        # 2. 检查是否需要学习 (阈值逻辑)
        should_learn = self._check_hysteresis(error)
        self.active_history.append(1 if should_learn else 0)

        # 3. 在线学习过程 (核心修正)
        if should_learn and len(self.command_memory) >= self.delay_step:
            # !!!!!!!! 重点 !!!!!!!!
            # 我们不使用 current_cmd 来计算梯度
            # 我们使用 delay_step 之前的那个指令 (past_cmd)
            # 因为是 past_cmd 导致了现在的 actual
            past_cmd = self.command_memory[0] 
            
            # 简单的梯度下降更新权重 (模拟 NN 反向传播)
            # 目标是：output = weight * input -> 调整 weight 使得 output 接近 target
            # Loss = 0.5 * (target - actual)^2
            # Grad = -(target - actual) * input_signal (简化版)
            
            # 注意：这里仅仅是演示更新逻辑，实际 NN 会更复杂
            grad = -error * 0.1 
            self.weights -= self.lr * grad 
            
            self.loss_history.append(error**2)
        else:
            self.loss_history.append(0)

    def predict(self, target_val):
        # 前馈控制 + 自适应权重
        # 这里的策略：主要靠前馈 (target)，权重用来补偿系统增益不足
        return target_val * self.weights

# ==========================================
# 3. 主循环仿真
# ==========================================
def run_simulation():
    # 初始化
    sim_steps = 1000
    delay_steps = 20 # 假设 20ms
    
    # 创建对象
    plant = DelayedMotorSystem(delay_step=delay_steps)
    controller = AdaptiveController(
        learning_rate=0.05, 
        delay_step=delay_steps, 
        delta=0.5, # 误差 < 0.5 停止修正
        beta=2.0   # 误差 > 2.0 开始修正
    )

    # 数据记录
    targets = []
    actuals = []
    commands = []
    
    # 生成目标信号 (正弦波)
    t = np.linspace(0, 20, sim_steps)
    target_signal = 10 + 5 * np.sin(t) # 类似于你的波形

    print("开始仿真...")
    
    for i in range(sim_steps):
        target = target_signal[i]
        
        # 1. 控制器计算输出 (Predict)
        cmd = controller.predict(target)
        
        # 2. 物理系统响应 (Step)
        actual = plant.step(cmd)
        
        # 3. 控制器在线学习 (Learn) - 包含延迟对齐和阈值判断
        controller.update(target, actual, cmd)
        
        # 记录
        targets.append(target)
        actuals.append(actual)
        commands.append(cmd)

    # ==========================================
    # 4. 绘图展示
    # ==========================================
    plt.figure(figsize=(12, 10))

    # 子图1: 跟踪效果
    plt.subplot(3, 1, 1)
    plt.title("Real-time Tracking with Delay Compensation")
    plt.plot(targets, 'g--', label='Target (Ideal)', linewidth=2)
    plt.plot(actuals, 'b-', label='Actual Output (Delayed Plant)', alpha=0.8)
    plt.ylabel("Torque")
    plt.legend()
    plt.grid(True)

    # 子图2: 激活状态 (阈值演示)
    plt.subplot(3, 1, 2)
    plt.title(f"Hysteresis Activation (Delta={controller.delta}, Beta={controller.beta})")
    plt.plot(np.array(targets) - np.array(actuals), 'k-', alpha=0.3, label="Error")
    
    # 画出阈值线
    plt.axhline(controller.beta, color='r', linestyle=':', label='Beta (Trigger)')
    plt.axhline(-controller.beta, color='r', linestyle=':')
    plt.axhline(controller.delta, color='orange', linestyle='--', label='Delta (Relax)')
    plt.axhline(-controller.delta, color='orange', linestyle='--')
    
    # 画出激活区域
    is_active_curve = np.array(controller.active_history) * (max(targets)-min(targets)) # 放大以便观看
    plt.fill_between(range(len(is_active_curve)), 0, 15, where=np.array(controller.active_history)==1, color='red', alpha=0.1, label="Learning Active")
    plt.legend(loc='upper right')
    plt.ylabel("Error / Active State")
    plt.grid(True)

    # 子图3: 权重变化 (自适应过程)
    plt.subplot(3, 1, 3)
    plt.title("Online Learning Convergence")
    plt.plot(controller.loss_history, 'k-', label='Loss (Squared Error)')
    plt.yscale('log')
    plt.ylabel("Loss (Log Scale)")
    plt.xlabel("Time Step (ms)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()