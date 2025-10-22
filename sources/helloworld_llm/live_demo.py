"""
实时可视化演示 - 无限循环观察agent行为
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import time

from core.world import SimpleWorld
from core.llm_agent_v2 import LLMAgentV2
from config import Config


class LiveVisualization:
    """实时可视化系统"""
    
    def __init__(self, num_agents=20, world_size=20):
        self.num_agents = num_agents
        self.world_size = world_size
        
        # 创建世界和agents
        self.world = SimpleWorld(world_size)
        self.agents = []
        self._init_agents()
        
        # 统计
        self.generation = 0
        self.total_steps = 0
        self.generation_food_delivered = 0
        self.all_time_food_delivered = 0
        
        # 设置matplotlib
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.suptitle('LLM Agent Evolution - Live Demo', fontsize=16, fontweight='bold')
        
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
                        exploration_map[ey, ex] += 0.1  # 累加探索热度
        
        # 绘制探索热力图（浅蓝色背景）
        if exploration_map.max() > 0:
            ax.imshow(exploration_map, cmap='Blues', alpha=0.3, origin='lower', vmin=0, vmax=1.0)
        
        # 绘制墙壁
        for (wx, wy) in self.world.wall_locations:
            ax.add_patch(plt.Rectangle((wx-0.5, wy-0.5), 1, 1, 
                                       facecolor='#444444', edgecolor='#666666'))
        
        # 绘制home（红色大方块）
        hx, hy = self.world.home_pos
        ax.add_patch(plt.Rectangle((hx-0.5, hy-0.5), 1, 1, 
                                   facecolor='#ff4444', edgecolor='#ff0000', linewidth=2))
        ax.text(hx, hy, '🏠', ha='center', va='center', fontsize=16)
        
        # 绘制食物（绿色小圆）
        for (fx, fy) in self.world.food_locations:
            ax.plot(fx, fy, 'go', markersize=8, markerfacecolor='#44ff44', 
                   markeredgecolor='#00ff00', markeredgewidth=1.5)
        
        # 🧠 绘制所有agent的食物记忆（淡黄色圆圈）
        for agent in self.agents:
            if agent.alive and len(agent.food_memory) > 0:
                for (mx, my) in agent.food_memory:
                    ax.plot(mx, my, 'o', markersize=6, markerfacecolor='none', 
                           markeredgecolor='#ffcc00', markeredgewidth=1, alpha=0.4)
        
        # 绘制agents
        for agent in self.agents:
            if agent.alive:
                x, y = agent.position
                
                # Agent颜色根据状态
                if agent.carrying_food:
                    color = '#ffaa00'  # 橙色 = 携带食物
                    marker = 's'
                else:
                    color = '#00aaff'  # 蓝色 = 正常
                    marker = 'o'
                
                # 绘制agent
                ax.plot(x, y, marker, markersize=10, markerfacecolor=color, 
                       markeredgecolor='white', markeredgewidth=1.5, alpha=0.8)
                
                # 🧠 绘制路径轨迹（最近5步）
                if len(agent.path_history) > 1:
                    path = agent.path_history[-5:]
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    ax.plot(path_x, path_y, '-', color=color, alpha=0.3, linewidth=1)
                
                # Energy bar (在agent上方)
                energy_ratio = agent.energy / agent.max_energy
                bar_width = 0.6
                bar_height = 0.1
                ax.add_patch(plt.Rectangle((x - bar_width/2, y + 0.6), 
                                          bar_width * energy_ratio, bar_height,
                                          facecolor='#00ff00' if energy_ratio > 0.5 else '#ff0000',
                                          edgecolor='white', linewidth=0.5))
        
        ax.set_xlim(-1, self.world.size)
        ax.set_ylim(-1, self.world.size)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a1a')
        ax.grid(True, alpha=0.2, color='#444444')
        ax.set_title(f"Generation {self.generation} - Total Steps {self.total_steps} | "
                    f"Alive: {sum(1 for a in self.agents if a.alive)}/{len(self.agents)}", 
                    color='white', fontsize=12, pad=10)
    
    def _draw_stats(self, ax):
        """绘制统计信息（增强：显示探索和记忆统计）"""
        ax.clear()
        ax.axis('off')
        
        # 标题
        title_text = f"📊 Generation {self.generation} Stats\n{'='*40}\n"
        ax.text(0.5, 0.95, title_text, ha='center', va='top', 
               fontsize=14, color='#00ffff', fontweight='bold',
               transform=ax.transAxes)
        
        # 获取存活agents并排序
        alive_agents = [a for a in self.agents if a.alive]
        alive_agents.sort(key=lambda a: a.total_reward, reverse=True)
        
        # 显示Top 10 agents
        y_pos = 0.85
        stats_text = "🏆 Top 10 Agents:\n"
        
        for i, agent in enumerate(alive_agents[:10]):
            reward = agent.total_reward
            stats_text += (f"  #{i+1} Agent-{agent.agent_id:02d}: "
                          f"R={reward:.0f} | "
                          f"E={agent.energy:3d} | "
                          f"Food={agent.food_delivered}🏠 | "
                          f"Explore={len(agent.explored_cells):3d}📍 | "
                          f"Memory={len(agent.food_memory):2d}🧠\n")
        
        # 🧠 新增：整体探索和记忆统计
        total_explored = sum(len(a.explored_cells) for a in alive_agents)
        avg_explored = total_explored / len(alive_agents) if alive_agents else 0
        total_memory = sum(len(a.food_memory) for a in alive_agents)
        avg_memory = total_memory / len(alive_agents) if alive_agents else 0
        
        stats_text += f"\n📈 Population Stats:\n"
        stats_text += f"  Avg Exploration: {avg_explored:.1f} cells\n"
        stats_text += f"  Avg Memory: {avg_memory:.1f} foods\n"
        stats_text += f"  This Gen Food: {self.generation_food_delivered}\n"
        stats_text += f"  All Time Food: {self.all_time_food_delivered}\n"
        stats_text += f"  Alive: {len(alive_agents)}/{len(self.agents)}\n"
        
        ax.text(0.05, y_pos, stats_text, ha='left', va='top',
               fontsize=9, color='white', family='monospace',
               transform=ax.transAxes)
        
        # 底部显示世界信息
        world_text = f"\n🌍 World: {self.world.size}x{self.world.size} | "
        world_text += f"Food: {len(self.world.food_locations)} | "
        world_text += f"Total Steps: {self.total_steps}"
        
        ax.text(0.5, 0.02, world_text, ha='center', va='bottom',
               fontsize=10, color='#aaaaaa',
               transform=ax.transAxes)
    
    def _draw_energy_chart(self, ax):
        """绘制能量分布图"""
        ax.clear()
        
        # 收集所有agent的能量
        energies = [agent.energy for agent in self.agents if agent.alive]
        
        if energies:
            ax.hist(energies, bins=10, color='#00aaff', edgecolor='white', alpha=0.7)
            ax.axvline(np.mean(energies), color='#ffaa00', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(energies):.1f}')
            ax.legend()
        
        ax.set_xlabel('Energy', color='white')
        ax.set_ylabel('Count', color='white')
        ax.set_title('Energy Distribution', color='white', fontsize=12, pad=10)
        ax.set_facecolor('#1a1a1a')
        ax.grid(True, alpha=0.2, color='#444444')
    
    def step(self):
        """执行一步"""
        alive_agents = [a for a in self.agents if a.alive]
        
        if len(alive_agents) == 0:
            # 全部死亡，进化到下一代
            self.evolve()
            return True
        
        # 所有存活agents执行一步
        for agent in alive_agents:
            result = agent.step(self.world)
            
            # 统计食物
            if agent.food_delivered > self.generation_food_delivered:
                self.generation_food_delivered = agent.food_delivered
                self.all_time_food_delivered += 1
        
        # 定期生成食物
        if self.total_steps % 50 == 0:
            self.world.spawn_food()
        
        self.total_steps += 1
        return True
    
    def evolve(self):
        """进化到下一代"""
        print(f"\n{'='*60}")
        print(f"Generation {self.generation} Complete!")
        print(f"  Food delivered: {self.generation_food_delivered}")
        print(f"  Total steps: {self.total_steps}")
        
        # 计算fitness
        fitnesses = [(agent, agent.get_fitness()) for agent in self.agents]
        fitnesses.sort(key=lambda x: x[1], reverse=True)
        
        # 显示top 3
        print("\nTop 3 Agents:")
        for i, (agent, fitness) in enumerate(fitnesses[:3], 1):
            print(f"  #{i} Agent {agent.agent_id}: fitness={fitness:.1f}, "
                  f"food={agent.food_delivered}, steps={agent.steps_taken}")
        
        # 选择精英
        elite_count = max(2, int(self.num_agents * 0.2))
        elites = [agent for agent, _ in fitnesses[:elite_count]]
        
        # 重置世界
        self.world = SimpleWorld(self.world_size)
        
        # 创建新一代
        new_agents = []
        new_agent_id = self.num_agents * (self.generation + 1)
        
        # 保留并重置elites
        for elite in elites:
            elite.energy = 200
            elite.alive = True
            elite.carrying_food = False
            while True:
                pos = (np.random.randint(1, self.world_size-1), 
                       np.random.randint(1, self.world_size-1))
                if self.world.can_move(*pos):
                    break
            elite.position = pos
            new_agents.append(elite)
        
        # 繁殖填满
        parent_pool = [agent for agent, _ in fitnesses[:max(4, int(self.num_agents * 0.5))]]
        
        while len(new_agents) < self.num_agents:
            parent1, parent2 = np.random.choice(parent_pool, 2, replace=False)
            while True:
                pos = (np.random.randint(1, self.world_size-1), 
                       np.random.randint(1, self.world_size-1))
                if self.world.can_move(*pos):
                    break
            child = LLMAgentV2.breed(parent1, parent2, new_agent_id, pos)
            new_agents.append(child)
            new_agent_id += 1
        
        self.agents = new_agents
        self.generation += 1
        self.generation_food_delivered = 0
        self.total_steps = 0
        
        print(f"\n✓ Generation {self.generation} ready!")
        print(f"{'='*60}\n")
    
    def animate(self, frame):
        """动画更新函数"""
        # 执行若干步（加快速度）
        for _ in range(5):  # 每帧执行5步
            self.step()
        
        # 更新所有面板
        self._draw_world(self.axes[0])
        self._draw_stats(self.axes[1])
        self._draw_energy_chart(self.axes[2])
        
        plt.tight_layout()
    
    def run(self):
        """运行实时演示"""
        print("="*60)
        print("LLM Agent Evolution - Live Visualization")
        print("="*60)
        print(f"Agents: {self.num_agents}")
        print(f"World Size: {self.world_size}x{self.world_size}")
        print(f"Model: {Config.MODEL_TYPE}")
        print("\nStarting visualization...")
        print("Close the window to stop.")
        print("="*60)
        
        # 创建动画
        anim = FuncAnimation(self.fig, self.animate, 
                           interval=100,  # 100ms更新一次
                           blit=False,
                           cache_frame_data=False)
        
        plt.show()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🎮 LLM Agent Evolution - Live Demo")
    print("="*60)
    
    # 创建可视化
    viz = LiveVisualization(num_agents=20, world_size=20)
    
    # 运行
    viz.run()


if __name__ == "__main__":
    main()
