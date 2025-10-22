"""
主运行脚本 - 20个LLM Agent的进化系统
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List
from pathlib import Path

from core.world import SimpleWorld
from core.llm_agent_v2 import LLMAgentV2
from config import Config


class EvolutionSimulation:
    """进化模拟系统"""
    
    def __init__(self, num_agents: int = 20, world_size: int = 20):
        self.num_agents = num_agents
        self.world_size = world_size
        
        # 创建世界
        self.world = SimpleWorld(world_size)
        
        # 创建agents
        self.agents: List[LLMAgentV2] = []
        self._init_agents()
        
        # 统计
        self.generation = 0
        self.total_steps = 0
        self.generation_stats = []
        
    def _init_agents(self):
        """初始化agents"""
        print(f"Initializing {self.num_agents} agents...")
        self.agents = []
        
        for i in range(self.num_agents):
            # 随机位置（但不在墙上）
            while True:
                pos = (np.random.randint(1, self.world_size-1), 
                       np.random.randint(1, self.world_size-1))
                if self.world.can_move(*pos):
                    break
            
            agent = LLMAgentV2(agent_id=i, position=pos, energy=200)
            self.agents.append(agent)
        
        print(f"✓ {len(self.agents)} agents initialized")
    
    def step(self):
        """执行一步模拟"""
        alive_agents = [a for a in self.agents if a.alive]
        
        if len(alive_agents) == 0:
            return False  # 全部死亡
        
        # 所有存活agents执行一步
        for agent in alive_agents:
            agent.step(self.world)
        
        # 世界更新（食物重新生成等）
        if self.total_steps % 50 == 0:
            self.world.spawn_food()
        
        self.total_steps += 1
        return True
    
    def evolve(self):
        """进化：选择、繁殖、变异"""
        print(f"\n=== Generation {self.generation} Evolution ===")
        
        # 计算每个agent的fitness
        fitnesses = [(agent, agent.get_fitness()) for agent in self.agents]
        fitnesses.sort(key=lambda x: x[1], reverse=True)
        
        # 显示最好的3个
        print("Top 3 agents:")
        for i, (agent, fitness) in enumerate(fitnesses[:3]):
            stats = agent.get_stats()
            print(f"  #{i+1} Agent {agent.agent_id}: fitness={fitness:.1f}, "
                  f"food_delivered={stats['food_delivered']}, "
                  f"food_collected={stats['food_collected']}, "
                  f"steps={stats['steps']}")
        
        # 记录统计
        gen_stat = {
            "generation": self.generation,
            "max_fitness": fitnesses[0][1],
            "avg_fitness": np.mean([f for _, f in fitnesses]),
            "total_food_delivered": sum(a.food_delivered for a, _ in fitnesses),
            "alive_count": sum(1 for a, _ in fitnesses if a.alive)
        }
        self.generation_stats.append(gen_stat)
        
        # 选择elite（保留top 20%）
        elite_count = max(2, int(self.num_agents * Config.ELITE_RATIO))
        elites = [agent for agent, _ in fitnesses[:elite_count]]
        
        print(f"Keeping {elite_count} elites")
        
        # 从top 50%中选择parents进行繁殖
        parent_pool_size = max(4, int(self.num_agents * 0.5))
        parent_pool = [agent for agent, _ in fitnesses[:parent_pool_size]]
        
        # 生成offspring
        new_agents = []
        new_agent_id = self.num_agents
        
        # 保留elites
        for elite in elites:
            # 重置elite的状态
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
        
        # 繁殖填满剩余位置
        while len(new_agents) < self.num_agents:
            # 随机选择两个parents
            parent1, parent2 = np.random.choice(parent_pool, 2, replace=False)
            
            # 繁殖
            while True:
                pos = (np.random.randint(1, self.world_size-1), 
                       np.random.randint(1, self.world_size-1))
                if self.world.can_move(*pos):
                    break
            
            child = LLMAgentV2.breed(parent1, parent2, new_agent_id, pos)
            new_agents.append(child)
            new_agent_id += 1
        
        # 更新agents
        self.agents = new_agents
        self.generation += 1
        
        # 重置世界
        self.world = SimpleWorld(self.world_size)
        self.total_steps = 0
        
        print(f"✓ Generation {self.generation} ready with {len(self.agents)} agents\n")
    
    def run(self, steps_per_generation: int = 200, num_generations: int = 5):
        """运行模拟"""
        print(f"\n{'='*60}")
        print(f"Starting Evolution Simulation")
        print(f"  Agents: {self.num_agents}")
        print(f"  Steps per generation: {steps_per_generation}")
        print(f"  Generations: {num_generations}")
        print(f"  Online learning: {Config.ENABLE_ONLINE_LEARNING}")
        print(f"{'='*60}\n")
        
        for gen in range(num_generations):
            print(f"\n--- Generation {self.generation} ---")
            
            # 运行一代
            for step in range(steps_per_generation):
                if not self.step():
                    print(f"All agents died at step {step}")
                    break
                
                # 每100步显示进度
                if step % 100 == 0 and step > 0:
                    alive = sum(1 for a in self.agents if a.alive)
                    total_food = sum(a.food_delivered for a in self.agents)
                    print(f"  Step {step}/{steps_per_generation}: "
                          f"{alive}/{self.num_agents} alive, "
                          f"{total_food} food delivered")
            
            # 显示这一代的总结
            alive = sum(1 for a in self.agents if a.alive)
            total_food = sum(a.food_delivered for a in self.agents)
            total_collected = sum(a.food_collected for a in self.agents)
            
            print(f"\nGeneration {self.generation} Summary:")
            print(f"  Final: {alive}/{self.num_agents} alive")
            print(f"  Total food delivered: {total_food}")
            print(f"  Total food collected: {total_collected}")
            print(f"  Efficiency: {total_food/total_collected*100 if total_collected > 0 else 0:.1f}%")
            
            # 进化到下一代
            if gen < num_generations - 1:
                self.evolve()
                time.sleep(1)
        
        # 显示最终统计
        self.show_stats()
    
    def show_stats(self):
        """显示统计图表"""
        if len(self.generation_stats) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        generations = [s["generation"] for s in self.generation_stats]
        max_fitness = [s["max_fitness"] for s in self.generation_stats]
        avg_fitness = [s["avg_fitness"] for s in self.generation_stats]
        total_food = [s["total_food_delivered"] for s in self.generation_stats]
        
        # Max fitness
        axes[0, 0].plot(generations, max_fitness, 'b-', label='Max Fitness')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title('Max Fitness Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Avg fitness
        axes[0, 1].plot(generations, avg_fitness, 'g-', label='Avg Fitness')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Fitness')
        axes[0, 1].set_title('Average Fitness Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Total food delivered
        axes[1, 0].bar(generations, total_food)
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Food Delivered')
        axes[1, 0].set_title('Total Food Delivered per Generation')
        axes[1, 0].grid(True)
        
        # Agent decision times
        alive_agents = [a for a in self.agents if len(a.decision_times) > 0]
        if len(alive_agents) > 0:
            avg_times = [np.mean(a.decision_times) * 1000 for a in alive_agents[:10]]
            axes[1, 1].bar(range(len(avg_times)), avg_times)
            axes[1, 1].set_xlabel('Agent ID')
            axes[1, 1].set_ylabel('Avg Decision Time (ms)')
            axes[1, 1].set_title('Agent Decision Times (Top 10)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('evolution_stats.png', dpi=150)
        print("\n✓ Stats saved to evolution_stats.png")
        plt.show()


def main():
    """主函数"""
    # 检查模型是否存在
    if Config.MODEL_TYPE != "mock":
        model_path = Path(Config.MODEL_PATH)
        if not model_path.exists():
            print("=" * 60)
            print("ERROR: Model file not found!")
            print(f"Expected: {model_path}")
            print("\nPlease download a model first:")
            print("  See DOWNLOAD_MODEL.md for instructions")
            print("=" * 60)
            return
    else:
        print("=" * 60)
        print("Using Mock Model (no download needed)")
        print("=" * 60)
    
    # 创建并运行模拟
    sim = EvolutionSimulation(num_agents=20, world_size=20)
    sim.run(steps_per_generation=500, num_generations=10)


if __name__ == "__main__":
    main()
