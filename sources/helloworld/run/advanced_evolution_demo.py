import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple
import random

from core.expandable_world import ExpandableWorld, WorldConf
from core.advanced_agent import AdvancedAgent, EliteGenome, Dir
from policy.policy_advanced import policy_advanced
from viz.advanced_render import render_advanced_world, create_elegant_info_panel


class AdvancedEvolutionDemo:
    def __init__(self, n_agents: int = 20, init_size: int = 64, max_size: int = 256):
        self.n_agents = n_agents
        self.init_size = init_size
        self.max_size = max_size
        
        conf = WorldConf(
            H=init_size,
            W=init_size,
            max_H=max_size,
            max_W=max_size,
            expand_threshold=20,
        )
        self.world = ExpandableWorld(conf)
        
        hx = init_size // 2
        hy = init_size // 2
        self.world.place_home(hx, hy, radius=3)
        self._seed_base_environment(hx, hy)
        
        # 适量放置初始食物块，避免早期资源泛滥
        for _ in range(2):
            fx = random.randint(10, init_size - 11)
            fy = random.randint(10, init_size - 11)
            self.world.place_food_patch(fx, fy, radius=4, amount=140)
        
        self.agents: List[AdvancedAgent] = []
        self.next_agent_id = 0
        for _ in range(n_agents):
            genome = EliteGenome()
            agent = self._spawn_agent(genome)
            self.agents.append(agent)
        
        self.step_count = 0
        self.generation = 0
        self.reproduce_interval = 280  # 调整为280步，更频繁地进行繁殖
        
        self.history_pop = []
        self.history_avg_fitness = []
        self.history_world_size = []
        self.history_total_buildings = []
        
        # 优化布局，减少拥挤
        self.fig = plt.figure(figsize=(20, 11))
        gs = self.fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35, 
                                   left=0.05, right=0.98, top=0.96, bottom=0.05)
        
        self.ax_world = self.fig.add_subplot(gs[:, :2])
        self.ax_info = self.fig.add_subplot(gs[0, 2])
        self.ax_genome = self.fig.add_subplot(gs[0, 3])
        self.ax_pop = self.fig.add_subplot(gs[1, 2])
        self.ax_fitness = self.fig.add_subplot(gs[1, 3])
        self.ax_worldsize = self.fig.add_subplot(gs[2, 2])
        self.ax_buildings = self.fig.add_subplot(gs[2, 3])
        
        self.elite_genome = None
    
    def step(self):
        alive_agents = [a for a in self.agents if a.alive]
        
        # Update all living agents
        for agent in alive_agents:
            action = policy_advanced(agent, self.world)
            agent.act(action, self.world)
        
        # Check for world expansion
        for agent in alive_agents:
            new_x, new_y = self.world.check_and_expand(agent.x, agent.y)
            if (new_x, new_y) != (agent.x, agent.y):
                agent.x, agent.y = new_x, new_y
        
        self.world.step()
        
        self.step_count += 1
        
        # Natural selection: remove dead agents from list
        previously_alive = len(self.agents)
        self.agents = [a for a in self.agents if a.alive]
        deaths = previously_alive - len(self.agents)
        if deaths > 0:
            print(f"💀 {deaths} agent(s) died from starvation. Remaining: {len(self.agents)}")
        
        # Reproduce if interval reached and we have living agents
        if self.step_count % self.reproduce_interval == 0 and len(self.agents) > 0:
            self.reproduce()
        # Emergency reproduction if population gets too low
        elif len(self.agents) < self.n_agents // 4 and len(self.agents) > 0:
            print(f"⚠️  Population critically low ({len(self.agents)}), triggering emergency reproduction!")
            self.reproduce()
    
    def reproduce(self):
        """Breeding system: offspring are born from living parents, not replacement"""
        alive = [a for a in self.agents if a.alive]
        if len(alive) == 0:
            print(f"⚠️  All agents died! Restarting with new population...")
            self.agents = [self._spawn_agent(EliteGenome()) for _ in range(self.n_agents)]
            if self.world.home_positions:
                hx, hy = self.world.home_positions[0]
                self._seed_base_environment(hx, hy)
            self.generation = 0
            return
        
        alive.sort(key=lambda a: a.fitness(), reverse=True)
        
        elite = alive[0]
        self.elite_genome = copy.deepcopy(elite.genome)
        
        # Select breeding parents - top 40% can reproduce
        parent_pool_size = max(2, int(len(alive) * 0.4))
        parents = alive[:parent_pool_size]
        
        # Keep the best parents alive (top 25%)
        survivors = alive[:max(2, int(len(alive) * 0.25))]
        
        # Start with surviving parents
        new_agents: List[AdvancedAgent] = survivors.copy()
        
        # Calculate how many offspring to create
        offspring_count = self.n_agents - len(new_agents)
        
        # Breed offspring from parent pool
        for _ in range(offspring_count):
            parent_a = random.choice(parents)
            parent_b = random.choice(parents)
            
            # Try to select different parents
            if len(parents) > 1:
                retry = 0
                while parent_b.id == parent_a.id and retry < 3:
                    parent_b = random.choice(parents)
                    retry += 1
            
            # Create offspring through crossover
            genome = EliteGenome.crossover(parent_a.genome, parent_b.genome, mutation_rate=0.08)
            child_generation = max(parent_a.generation, parent_b.generation) + 1
            offspring = self._spawn_agent(genome, generation=child_generation, parent_id=parent_a.id)
            new_agents.append(offspring)
        
        self.agents = new_agents[:self.n_agents]
        self.generation += 1
        
        avg_fitness = np.mean([a.fitness() for a in alive])
        print(f"Gen {self.generation}: Elite fitness={elite.fitness():.1f}, "
              f"Avg fitness={avg_fitness:.1f}, Survivors={len(survivors)}, "
              f"Offspring={offspring_count}, Pop={len(self.agents)}, World={self.world.H}x{self.world.W}")
    
    def get_stats(self):
        alive = [a for a in self.agents if a.alive]
        if len(alive) == 0:
            return {
                "alive": 0,
                "avg_energy": 0,
                "avg_fitness": 0,
                "max_fitness": 0,
                "avg_generation": 0,
                "explorers": 0,
                "gatherers": 0,
                "builders": 0,
            }
        
        # 统计角色
        explorers = sum(1 for a in alive if a.role == "Explorer")
        gatherers = sum(1 for a in alive if a.role == "Gatherer")
        builders = sum(1 for a in alive if a.role == "Builder")
        
        return {
            "alive": len(alive),
            "avg_energy": np.mean([a.energy for a in alive]),
            "avg_fitness": np.mean([a.fitness() for a in alive]),
            "max_fitness": max(a.fitness() for a in alive),
            "avg_generation": np.mean([a.generation for a in alive]),
            "explorers": explorers,
            "gatherers": gatherers,
            "builders": builders,
        }
    
    def get_building_stats(self):
        nest_count = np.sum(self.world.layers["NEST"] > 0)
        storage_count = np.sum(self.world.layers["STORAGE"] > 0)
        trail_count = np.sum(self.world.layers["TRAIL"] > 0)
        return {
            "nests": nest_count,
            "storage": storage_count,
            "trails": trail_count,
            "total": nest_count + storage_count + trail_count,
        }
    
    def update_charts(self):
        stats = self.get_stats()
        building_stats = self.get_building_stats()
        
        self.history_pop.append(stats["alive"])
        self.history_avg_fitness.append(stats["avg_fitness"])
        self.history_world_size.append(self.world.H * self.world.W)
        self.history_total_buildings.append(building_stats["total"])
        
        self.ax_pop.clear()
        self.ax_pop.plot(self.history_pop, color='blue', linewidth=2)
        self.ax_pop.set_title("Population Over Time", fontsize=10, weight='bold')
        self.ax_pop.set_xlabel("Generations", fontsize=8)
        self.ax_pop.set_ylabel("Alive Agents", fontsize=8)
        self.ax_pop.grid(True, alpha=0.3)
        
        self.ax_fitness.clear()
        self.ax_fitness.plot(self.history_avg_fitness, color='green', linewidth=2)
        self.ax_fitness.set_title("Average Fitness", fontsize=10, weight='bold')
        self.ax_fitness.set_xlabel("Generations", fontsize=8)
        self.ax_fitness.set_ylabel("Fitness", fontsize=8)
        self.ax_fitness.grid(True, alpha=0.3)
        
        self.ax_worldsize.clear()
        self.ax_worldsize.plot(self.history_world_size, color='purple', linewidth=2)
        self.ax_worldsize.set_title("World Size", fontsize=10, weight='bold')
        self.ax_worldsize.set_xlabel("Generations", fontsize=8)
        self.ax_worldsize.set_ylabel("Total Pixels", fontsize=8)
        self.ax_worldsize.grid(True, alpha=0.3)
        
        self.ax_buildings.clear()
        self.ax_buildings.plot(self.history_total_buildings, color='orange', linewidth=2)
        self.ax_buildings.set_title("Total Buildings", fontsize=10, weight='bold')
        self.ax_buildings.set_xlabel("Generations", fontsize=8)
        self.ax_buildings.set_ylabel("Count", fontsize=8)
        self.ax_buildings.grid(True, alpha=0.3)
        
        stats = self.get_stats()
        building_stats = self.get_building_stats()
        
        info_data = {
            "__section__1": "═══ Simulation ═══",
            "Step": self.step_count,
            "Generation": self.generation,
            "World Size": f"{self.world.H}x{self.world.W}",
            "Expansions": self.world.expansion_count,
            "__section__2": "═══ Population ═══",
            "Alive": stats["alive"],
            "Avg Energy": f"{stats['avg_energy']:.1f}",
            "Avg Fitness": f"{stats['avg_fitness']:.1f}",
            "Max Fitness": f"{stats['max_fitness']:.1f}",
            "Avg Gen": f"{stats['avg_generation']:.1f}",
            "__section__3": "═══ Roles ═══",
            "🔍 Explorers": stats["explorers"],
            "🌾 Gatherers": stats["gatherers"],
            "🏗️  Builders": stats["builders"],
            "__section__4": "═══ Buildings ═══",
            "Nests": building_stats["nests"],
            "Storage": building_stats["storage"],
            "Trails": building_stats["trails"],
        }
        create_elegant_info_panel(self.ax_info, "📊 Status", info_data, bg_color='lightgreen')
        
        if self.elite_genome:
            genome_data = {
                "__section__1": "═══ Elite Genome ═══",
                "Exploration": f"{self.elite_genome.exploration_rate:.2f}",
                "Food Attr": f"{self.elite_genome.food_attraction:.2f}",
                "Home Attr": f"{self.elite_genome.home_attraction:.2f}",
                "Pher Sens": f"{self.elite_genome.pheromone_sensitivity:.2f}",
                "Energy Eff": f"{self.elite_genome.energy_efficiency:.2f}",
                "Build Tend": f"{self.elite_genome.building_tendency:.2f}",
                "Body Size": f"{self.elite_genome.body_size:.2f}",
            }
            create_elegant_info_panel(self.ax_genome, "🧬 Best Genes", genome_data, bg_color='lightyellow')
    
    def animate(self, frame):
        # 每次动画更新只执行2步（原来是5步）
        for _ in range(2):
            self.step()
        
        render_advanced_world(
            self.world,
            self.agents,
            ax=self.ax_world,
            title=f"🌍 Advanced Evolution World (Gen {self.generation})",
            show_grid=False,
        )
        
        # 更频繁地更新图表
        if self.step_count % 20 == 0:
            self.update_charts()
        
        return []
    
    def run(self):
        anim = FuncAnimation(
            self.fig,
            self.animate,
            interval=100,  # 增加到100ms（原来50ms），进一步降低速度
            blit=False,
            cache_frame_data=False,
        )
        
        # 移除tight_layout，使用gridspec的边距设置
        plt.show()

    def _random_spawn_position(self, radius: int = 5) -> Tuple[int, int]:
        if self.world.home_positions:
            center_x, center_y = self.world.home_positions[0]
        else:
            center_x = self.world.W // 2
            center_y = self.world.H // 2
        radius = max(2, min(radius, center_x - 1 if center_x > 1 else 2, center_y - 1 if center_y > 1 else 2))
        x_min = max(1, center_x - radius)
        x_max = min(self.world.W - 2, center_x + radius)
        y_min = max(1, center_y - radius)
        y_max = min(self.world.H - 2, center_y + radius)
        if x_min > x_max:
            x_min, x_max = center_x, center_x
        if y_min > y_max:
            y_min, y_max = center_y, center_y
        return random.randint(x_min, x_max), random.randint(y_min, y_max)

    def _spawn_agent(
        self,
        genome: EliteGenome,
        generation: int = 0,
        parent_id: Optional[int] = None,
    ) -> AdvancedAgent:
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        x, y = self._random_spawn_position()
        base_energy = 78.0 + genome.energy_efficiency * 9.0
        return AdvancedAgent.spawn(
            id=agent_id,
            x=x,
            y=y,
            genome=genome,
            generation=generation,
            parent_id=parent_id,
            base_energy=base_energy,
        )

    def _seed_base_environment(self, cx: int, cy: int) -> None:
        # 预先在巢穴周围铺设返家信息素和少量小径，帮助早期导航
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                x = cx + dx
                y = cy + dy
                if not self.world.in_bounds(x, y):
                    continue
                distance_sq = dx * dx + dy * dy
                if distance_sq <= 64:
                    decay = max(0.1, 1.0 - (distance_sq / 64.0))
                    current_home_pher = self.world.read_cell("PHER_HOME", x, y)
                    boosted = min(255, current_home_pher + int(180 * decay))
                    self.world.write_cell("PHER_HOME", x, y, boosted)
                    if distance_sq <= 9:
                        current_trail = self.world.read_cell("TRAIL", x, y)
                        self.world.write_cell("TRAIL", x, y, min(255, current_trail + 120))
        # 提供一些近巢食物维持产业
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                if dx * dx + dy * dy > 18:
                    continue
                if random.random() > 0.45:
                    continue
                x = cx + dx
                y = cy + dy
                if self.world.in_bounds(x, y) and self.world.read_cell("FOOD", x, y) == 0:
                    self.world.write_cell("FOOD", x, y, random.randint(18, 36))


if __name__ == "__main__":
    demo = AdvancedEvolutionDemo(n_agents=25, init_size=64, max_size=256)
    demo.run()
