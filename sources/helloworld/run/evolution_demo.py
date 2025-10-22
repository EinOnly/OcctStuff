import sys
import os
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.world import World, WorldConf
from core.agent_evolved import Agent, Dir, AgentGenome
from policy.tiny_policy_evolved import execute_action_evolved
from viz.render import render_world


class EvolutionDemo:
    
    def __init__(self, initial_agents: int = 10, world_size: int = 128, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        conf = WorldConf(H=world_size, W=world_size, pher_decay=0.95, diffuse_weight=0.2)
        self.world = World(conf)
        
        home_x, home_y = conf.W // 2, conf.H // 2
        self.world.place_home(home_x, home_y, radius=10)
        self.home_center = (home_x, home_y)
        
        self.world.scatter_food(self.rng, num_patches=20, patch_radius=5, amount=150)
        
        self.world.place_obstacle_rect(0, 0, conf.W, 3)
        self.world.place_obstacle_rect(0, 0, 3, conf.H)
        self.world.place_obstacle_rect(conf.W - 3, 0, conf.W, conf.H)
        self.world.place_obstacle_rect(0, conf.H - 3, conf.W, conf.H)
        
        self.agents = []
        self.next_id = 0
        
        for i in range(initial_agents):
            self._spawn_agent(generation=0)
        
        self.step_count = 0
        self.generation_count = 0
        self.total_births = initial_agents
        self.total_deaths = 0
        
        self.history_pop = []
        self.history_avg_fitness = []
        self.history_avg_energy = []
        
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[2, 1])
        
        self.ax_world = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_stats = self.fig.add_subplot(gs[0, 2])
        self.ax_pop = self.fig.add_subplot(gs[1, 0])
        self.ax_fitness = self.fig.add_subplot(gs[1, 1])
        self.ax_energy = self.fig.add_subplot(gs[1, 2])
        
        print(f"Evolution Demo Started:")
        print(f"  Initial population: {initial_agents}")
        print(f"  World size: {world_size}x{world_size}")
        print(f"  Seed: {seed}")
        print(f"  Press Ctrl+C to stop")
    
    def _spawn_agent(self, generation: int = 0, parent_genome: AgentGenome = None):
        home_x, home_y = self.home_center
        angle = random.random() * 2 * np.pi
        offset = random.randint(8, 15)
        x = home_x + int(offset * np.cos(angle))
        y = home_y + int(offset * np.sin(angle))
        d = Dir(random.randint(0, 3))
        
        if parent_genome:
            genome = parent_genome.mutate(mutation_rate=0.15)
        else:
            genome = AgentGenome()
        
        agent = Agent(
            id=self.next_id,
            x=x,
            y=y,
            d=d,
            genome=genome,
            generation=generation,
            energy=100.0
        )
        self.agents.append(agent)
        self.next_id += 1
        return agent
    
    def update_step(self):
        alive_agents = [a for a in self.agents if a.alive]
        
        for agent in alive_agents:
            if agent.energy > 0:
                obs = agent.obs_string(self.world)
                execute_action_evolved(agent, self.world, obs)
        
        self.world.step_fields()
        
        if self.step_count % 100 == 0 and self.step_count > 0:
            self.world.scatter_food(self.rng, num_patches=3, patch_radius=4, amount=100)
        
        dead_agents = [a for a in self.agents if not a.alive]
        for agent in dead_agents:
            self.agents.remove(agent)
            self.total_deaths += 1
        
        if self.step_count % 200 == 0 and self.step_count > 0:
            self._reproduction()
        
        if len(self.agents) == 0:
            print("Population extinct! Restarting...")
            for _ in range(5):
                self._spawn_agent(generation=self.generation_count)
            self.total_births += 5
        
        if len(self.agents) > 50:
            self.agents.sort(key=lambda a: a.fitness(), reverse=True)
            removed = self.agents[50:]
            self.agents = self.agents[:50]
            self.total_deaths += len(removed)
        
        self.step_count += 1
    
    def _reproduction(self):
        if len(self.agents) < 2:
            return
        
        self.agents.sort(key=lambda a: a.fitness(), reverse=True)
        
        top_performers = self.agents[:max(2, len(self.agents) // 3)]
        
        num_offspring = min(5, len(top_performers))
        
        self.generation_count += 1
        
        for _ in range(num_offspring):
            parent = random.choice(top_performers)
            if parent.food_delivered > 0:
                parent.energy -= 20
                if parent.energy > 0:
                    self._spawn_agent(generation=self.generation_count, parent_genome=parent.genome)
                    self.total_births += 1
    
    def render_frame(self, frame):
        self.update_step()
        
        for ax in [self.ax_world, self.ax_info, self.ax_stats, self.ax_pop, self.ax_fitness, self.ax_energy]:
            ax.clear()
        
        render_world(self.world, self.agents, ax=self.ax_world, 
                    title=f"Step {self.step_count} | Generation {self.generation_count} | Population {len(self.agents)}")
        
        self.ax_info.axis('off')
        
        info_text = f"EVOLUTION STATUS\n{'=' * 35}\n\n"
        info_text += f"Step: {self.step_count}\n"
        info_text += f"Generation: {self.generation_count}\n"
        info_text += f"Population: {len(self.agents)}\n"
        info_text += f"Total Births: {self.total_births}\n"
        info_text += f"Total Deaths: {self.total_deaths}\n\n"
        
        if self.agents:
            avg_energy = np.mean([a.energy for a in self.agents])
            avg_fitness = np.mean([a.fitness() for a in self.agents])
            avg_age = np.mean([a.age for a in self.agents])
            total_delivered = sum(a.food_delivered for a in self.agents)
            
            info_text += f"Avg Energy: {avg_energy:.1f}\n"
            info_text += f"Avg Fitness: {avg_fitness:.1f}\n"
            info_text += f"Avg Age: {avg_age:.1f}\n"
            info_text += f"Food Delivered: {total_delivered}\n\n"
            
            self.agents.sort(key=lambda a: a.fitness(), reverse=True)
            info_text += f"TOP 3 AGENTS\n{'-' * 35}\n"
            for i, agent in enumerate(self.agents[:3]):
                info_text += f"{i+1}. ID{agent.id} Gen{agent.generation}\n"
                info_text += f"   Fitness: {agent.fitness():.1f}\n"
                info_text += f"   Energy: {agent.energy:.1f}\n"
                info_text += f"   Delivered: {agent.food_delivered}\n"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        self.ax_stats.axis('off')
        if self.agents:
            stats_text = f"GENOME STATS\n{'=' * 35}\n\n"
            
            avg_explore = np.mean([a.genome.exploration_rate for a in self.agents])
            avg_food_attr = np.mean([a.genome.food_attraction for a in self.agents])
            avg_home_attr = np.mean([a.genome.home_attraction for a in self.agents])
            avg_pher_sens = np.mean([a.genome.pheromone_sensitivity for a in self.agents])
            avg_efficiency = np.mean([a.genome.energy_efficiency for a in self.agents])
            
            stats_text += f"Exploration: {avg_explore:.2f}\n"
            stats_text += f"Food Attraction: {avg_food_attr:.2f}\n"
            stats_text += f"Home Attraction: {avg_home_attr:.2f}\n"
            stats_text += f"Pher. Sensitivity: {avg_pher_sens:.2f}\n"
            stats_text += f"Energy Efficiency: {avg_efficiency:.2f}\n\n"
            
            stats_text += f"GENERATION DIVERSITY\n{'-' * 35}\n"
            gen_counts = {}
            for agent in self.agents:
                gen_counts[agent.generation] = gen_counts.get(agent.generation, 0) + 1
            
            for gen in sorted(gen_counts.keys(), reverse=True)[:5]:
                stats_text += f"Gen {gen}: {gen_counts[gen]} agents\n"
            
            self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                             fontsize=9, verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        self.history_pop.append(len(self.agents))
        if self.agents:
            self.history_avg_fitness.append(np.mean([a.fitness() for a in self.agents]))
            self.history_avg_energy.append(np.mean([a.energy for a in self.agents]))
        else:
            self.history_avg_fitness.append(0)
            self.history_avg_energy.append(0)
        
        window = 500
        
        self.ax_pop.plot(self.history_pop[-window:], 'b-', linewidth=2)
        self.ax_pop.set_title('Population Over Time')
        self.ax_pop.set_ylabel('Population')
        self.ax_pop.set_xlabel('Step')
        self.ax_pop.grid(True, alpha=0.3)
        
        self.ax_fitness.plot(self.history_avg_fitness[-window:], 'g-', linewidth=2)
        self.ax_fitness.set_title('Avg Fitness Over Time')
        self.ax_fitness.set_ylabel('Fitness')
        self.ax_fitness.set_xlabel('Step')
        self.ax_fitness.grid(True, alpha=0.3)
        
        self.ax_energy.plot(self.history_avg_energy[-window:], 'r-', linewidth=2)
        self.ax_energy.set_title('Avg Energy Over Time')
        self.ax_energy.set_ylabel('Energy')
        self.ax_energy.set_xlabel('Step')
        self.ax_energy.grid(True, alpha=0.3)
        
        return self.ax_world, self.ax_info, self.ax_stats
    
    def run(self, interval: int = 10):
        try:
            anim = FuncAnimation(self.fig, self.render_frame, interval=interval, 
                               blit=False, cache_frame_data=False)
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            print(f"\nEvolution demo stopped at step {self.step_count}")
            print(f"Final generation: {self.generation_count}")
            print(f"Final population: {len(self.agents)}")
            print(f"Total births: {self.total_births}")
            print(f"Total deaths: {self.total_deaths}")
            if self.agents:
                best = max(self.agents, key=lambda a: a.fitness())
                print(f"Best agent: ID{best.id}, Fitness={best.fitness():.1f}, Delivered={best.food_delivered}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evolution Demo - Agents with Survival, Energy, and Reproduction")
    parser.add_argument("--agents", type=int, default=10,
                        help="Initial number of agents (default: 10)")
    parser.add_argument("--world-size", type=int, default=128,
                        help="World size (default: 128)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--interval", type=int, default=10,
                        help="Animation interval in ms (default: 10)")
    
    args = parser.parse_args()
    
    demo = EvolutionDemo(initial_agents=args.agents, world_size=args.world_size, seed=args.seed)
    demo.run(interval=args.interval)


if __name__ == "__main__":
    main()
