"""
实时可视化演示 - 异步版本
支持并发agent推理和异步状态更新
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import asyncio
import concurrent.futures
import time
from typing import List

from core.world import SimpleWorld
from core.llm_agent_v2 import LLMAgentV2
from config import Config


class LiveVisualizationAsync:
    """异步实时可视化系统"""
    
    def __init__(self, num_agents=5, world_size=20):
        self.num_agents = num_agents
        self.world_size = world_size
        
        # 先初始化共享模型（在创建agents之前）
        print("Initializing shared base model...")
        from core.model_loader import SharedModelPool
        self.model_pool = SharedModelPool()
        self.shared_model = self.model_pool.get_base_model()
        
        # 创建世界和agents
        self.world = SimpleWorld(world_size)
        self.agents = []
        self._init_agents()
        
        # 统计
        self.generation = 0
        self.total_steps = 0
        self.generation_food_delivered = 0
        self.all_time_food_delivered = 0
        
        # 异步控制
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_agents)
        self.pending_actions = {}  # agent_id -> (action, future)
        
        # 设置matplotlib
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.suptitle('LLM Agent Evolution - Live Demo (Async)', fontsize=16, fontweight='bold')
        
        # 颜色映射
        self.agent_colors = plt.cm.rainbow(np.linspace(0, 1, num_agents))
        
    def _init_agents(self):
        """初始化agents"""
        self.agents = []
        for i in range(self.num_agents):
            while True:
                pos = (np.random.randint(1, self.world_size-1), 
                       np.random.randint(1, self.world_size-1))
                if self.world.can_move(*pos):
                    break
            agent = LLMAgentV2(agent_id=i, position=pos, energy=200)
            self.agents.append(agent)
    
    def _draw_world(self, ax):
        """绘制世界地图（增强：显示探索区域和记忆）"""
        ax.clear()
        
        # 🧠 绘制所有agent的探索区域（热力图）
        exploration_map = np.zeros((self.world.size, self.world.size))
        for agent in self.agents:
            if agent.alive:
                for (ex, ey) in agent.explored_cells:
                    if 0 <= ex < self.world.size and 0 <= ey < self.world.size:
                        exploration_map[ey, ex] += 0.2  # 更高的探索热度
        
        # 绘制探索热力图（浅蓝色背景）
        if exploration_map.max() > 0:
            ax.imshow(exploration_map, cmap='Blues', alpha=0.4, origin='lower', vmin=0, vmax=1.0)
        
        # 绘制墙壁
        for (wx, wy) in self.world.wall_locations:
            ax.add_patch(plt.Rectangle((wx-0.5, wy-0.5), 1, 1, 
                                       facecolor='#444444', edgecolor='#666666'))
        
        # 绘制home（红色大方块）
        hx, hy = self.world.home_pos
        ax.add_patch(plt.Rectangle((hx-0.5, hy-0.5), 1, 1, 
                                   facecolor='#ff4444', edgecolor='#ff0000', linewidth=2))
        ax.text(hx, hy, 'HOME', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
        # 绘制食物（绿色小圆）
        for (fx, fy) in self.world.food_locations:
            ax.plot(fx, fy, 'go', markersize=10, markerfacecolor='#44ff44', 
                   markeredgecolor='#00ff00', markeredgewidth=2)
        
        # 🧠 绘制所有agent的食物记忆（淡黄色圆圈）
        for agent in self.agents:
            if agent.alive and len(agent.food_memory) > 0:
                for (mx, my) in agent.food_memory:
                    ax.plot(mx, my, 'o', markersize=8, markerfacecolor='none', 
                           markeredgecolor='#ffcc00', markeredgewidth=1.5, alpha=0.5)
        
        # 绘制agents
        for agent in self.agents:
            if agent.alive:
                x, y = agent.position
                
                # Agent颜色根据状态
                if agent.carrying_food:
                    color = '#ffaa00'  # 橙色 = 携带食物
                    marker = 's'
                    size = 15
                else:
                    color = '#00aaff'  # 蓝色 = 正常
                    marker = 'o'
                    size = 12
                
                # 绘制agent
                ax.plot(x, y, marker, markersize=size, markerfacecolor=color, 
                       markeredgecolor='white', markeredgewidth=2, alpha=0.9)
                
                # Agent ID
                ax.text(x, y-0.7, f'{agent.agent_id}', ha='center', va='top',
                       fontsize=8, color='white', fontweight='bold')
                
                # 🧠 绘制路径轨迹（最近10步）
                if len(agent.path_history) > 1:
                    path = agent.path_history[-10:]
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    ax.plot(path_x, path_y, '-', color=color, alpha=0.4, linewidth=2)
                
                # Energy bar (在agent上方)
                energy_ratio = agent.energy / agent.max_energy
                bar_width = 0.8
                bar_height = 0.15
                ax.add_patch(plt.Rectangle((x - bar_width/2, y + 0.7), 
                                          bar_width * energy_ratio, bar_height,
                                          facecolor='#00ff00' if energy_ratio > 0.5 else '#ff0000',
                                          edgecolor='white', linewidth=1))
        
        ax.set_xlim(-1, self.world.size)
        ax.set_ylim(-1, self.world.size)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a1a')
        ax.grid(True, alpha=0.2, color='#444444')
        ax.set_title(f"Gen {self.generation} | Step {self.total_steps} | "
                    f"Alive: {sum(1 for a in self.agents if a.alive)}/{len(self.agents)}", 
                    color='white', fontsize=14, pad=10, fontweight='bold')
    
    def _draw_stats(self, ax):
        """绘制统计信息（增强：显示探索和记忆统计）"""
        ax.clear()
        ax.axis('off')
        
        # 标题
        title_text = f"Stats - Generation {self.generation}\n{'='*35}\n"
        ax.text(0.5, 0.95, title_text, ha='center', va='top', 
               fontsize=12, color='#00ffff', fontweight='bold',
               transform=ax.transAxes)
        
        # 获取存活agents并排序
        alive_agents = [a for a in self.agents if a.alive]
        alive_agents.sort(key=lambda a: a.total_reward, reverse=True)
        
        # 显示所有agents
        y_pos = 0.85
        stats_text = "Agent Rankings:\n"
        
        for i, agent in enumerate(alive_agents):
            reward = agent.total_reward
            stats_text += (f"  #{i+1} Agent-{agent.agent_id}: "
                          f"R={reward:.0f} | "
                          f"E={agent.energy:3d} | "
                          f"Food={agent.food_delivered} | "
                          f"Explore={len(agent.explored_cells):3d} | "
                          f"Mem={len(agent.food_memory):2d}\n")
        
        # 🧠 新增：整体探索和记忆统计
        total_explored = sum(len(a.explored_cells) for a in alive_agents)
        avg_explored = total_explored / len(alive_agents) if alive_agents else 0
        total_memory = sum(len(a.food_memory) for a in alive_agents)
        avg_memory = total_memory / len(alive_agents) if alive_agents else 0
        
        stats_text += f"\nPopulation Stats:\n"
        stats_text += f"  Avg Exploration: {avg_explored:.1f} cells\n"
        stats_text += f"  Avg Memory: {avg_memory:.1f} foods\n"
        stats_text += f"  Gen Food: {self.generation_food_delivered}\n"
        stats_text += f"  Total Food: {self.all_time_food_delivered}\n"
        stats_text += f"  Alive: {len(alive_agents)}/{len(self.agents)}\n"
        
        ax.text(0.05, y_pos, stats_text, ha='left', va='top',
               fontsize=10, color='white', family='monospace',
               transform=ax.transAxes)
        
        # 底部显示世界信息
        world_text = f"\nWorld: {self.world.size}x{self.world.size} | "
        world_text += f"Food: {len(self.world.food_locations)} | "
        world_text += f"Steps: {self.total_steps}"
        
        ax.text(0.5, 0.02, world_text, ha='center', va='bottom',
               fontsize=10, color='#aaaaaa',
               transform=ax.transAxes)
    
    def _draw_energy_chart(self, ax):
        """绘制能量分布图"""
        ax.clear()
        
        # 收集所有agent的能量
        energies = [agent.energy for agent in self.agents if agent.alive]
        
        if energies:
            ax.hist(energies, bins=5, color='#00aaff', edgecolor='white', alpha=0.7, linewidth=2)
            ax.axvline(np.mean(energies), color='#ffaa00', linestyle='--', 
                      linewidth=3, label=f'Mean: {np.mean(energies):.1f}')
            ax.legend(fontsize=10)
        
        ax.set_xlabel('Energy', color='white', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', color='white', fontsize=11, fontweight='bold')
        ax.set_title('Energy Distribution', color='white', fontsize=12, pad=10, fontweight='bold')
        ax.set_facecolor('#1a1a1a')
        ax.grid(True, alpha=0.2, color='#444444')
    
    def _agent_step_async(self, agent: LLMAgentV2) -> dict:
        """单个agent执行一步（在线程池中运行）"""
        if not agent.alive:
            return None
        return agent.step(self.world)
    
    def step_async(self):
        """异步执行一步 - 所有agents并行推理"""
        # 1. 提交所有agent的推理任务
        futures = {}
        for agent in self.agents:
            if agent.alive:
                future = self.executor.submit(self._agent_step_async, agent)
                futures[agent.agent_id] = future
        
        # 2. 等待所有推理完成（非阻塞可视化）
        results = {}
        for agent_id, future in futures.items():
            try:
                result = future.result(timeout=5.0)  # 5秒超时
                if result:
                    results[agent_id] = result
            except Exception as e:
                print(f"Agent {agent_id} error: {e}")
        
        # 3. 更新统计
        self.total_steps += 1
        
        # 4. 检查是否需要重生食物
        if len(self.world.food_locations) < 3:
            self.world.spawn_food()
        
        # 5. 检查所有agents是否都死了
        alive_count = sum(1 for a in self.agents if a.alive)
        if alive_count == 0:
            self.evolve()
    
    def evolve(self):
        """进化到下一代"""
        print(f"\n{'='*60}")
        print(f"Generation {self.generation} Complete!")
        print(f"  Food delivered: {self.generation_food_delivered}")
        print(f"  Total steps: {self.total_steps}")
        
        # 统计
        self.all_time_food_delivered += self.generation_food_delivered
        
        # 简单重置（实际进化逻辑可以更复杂）
        self.generation += 1
        self.generation_food_delivered = 0
        
        # 重新初始化agents
        self._init_agents()
        
        # 重置世界
        self.world = SimpleWorld(self.world_size)
        
        print(f"✓ Generation {self.generation} ready!")
        print(f"{'='*60}\n")
    
    def animate(self, frame):
        """动画更新函数"""
        # 执行多步
        for _ in range(3):  # 每帧执行3步
            self.step_async()
        
        # 更新可视化
        self._draw_world(self.axes[0])
        self._draw_stats(self.axes[1])
        self._draw_energy_chart(self.axes[2])
        
        plt.tight_layout()
    
    def run(self):
        """运行可视化"""
        print("="*60)
        print("LLM Agent Evolution - Live Visualization (Async)")
        print("="*60)
        print(f"Agents: {self.num_agents}")
        print(f"World Size: {self.world_size}x{self.world_size}")
        print(f"Model: {Config.MODEL_TYPE}")
        print("\nStarting visualization...")
        print("Close the window to stop.")
        print("="*60)
        
        # 创建动画
        anim = FuncAnimation(self.fig, self.animate, 
                           interval=200,  # 200ms更新一次
                           blit=False,
                           cache_frame_data=False)
        
        plt.show()
        
        # 清理线程池
        self.executor.shutdown(wait=True)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🎮 LLM Agent Evolution - Async Live Demo")
    print("="*60)
    
    # 创建可视化（5个agents）
    viz = LiveVisualizationAsync(num_agents=5, world_size=20)
    
    # 运行
    viz.run()


if __name__ == "__main__":
    main()
