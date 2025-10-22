import sys
import os
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.world import World, WorldConf
from core.agent import Agent, Dir
from policy.tiny_policy import execute_action
from viz.render import render_world


class LiveDemo:
    
    def __init__(self, num_agents: int = 5, world_size: int = 128, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        conf = WorldConf(H=world_size, W=world_size, pher_decay=0.95, diffuse_weight=0.2)
        self.world = World(conf)
        
        home_x, home_y = conf.W // 2, conf.H // 2
        self.world.place_home(home_x, home_y, radius=8)
        self.home_center = (home_x, home_y)
        
        self.world.scatter_food(self.rng, num_patches=15, patch_radius=5, amount=100)
        
        self.world.place_obstacle_rect(0, 0, conf.W, 3)
        self.world.place_obstacle_rect(0, 0, 3, conf.H)
        self.world.place_obstacle_rect(conf.W - 3, 0, conf.W, conf.H)
        self.world.place_obstacle_rect(0, conf.H - 3, conf.W, conf.H)
        
        self.agents = []
        for i in range(num_agents):
            angle = (2 * np.pi * i) / num_agents
            offset = 12
            x = home_x + int(offset * np.cos(angle))
            y = home_y + int(offset * np.sin(angle))
            d = Dir(self.rng.integers(0, 4))
            self.agents.append(Agent(id=i, x=x, y=y, d=d))
        
        self.step_count = 0
        self.food_collected = 0
        self.total_food_gathered = 0
        
        self.fig, (self.ax_main, self.ax_info) = plt.subplots(1, 2, figsize=(16, 8), 
                                                                gridspec_kw={'width_ratios': [3, 1]})
        
        print(f"Live Demo Started: {num_agents} agents, world size {world_size}x{world_size}, seed={seed}")
        print("Press Ctrl+C to stop")
    
    def update_step(self):
        for agent in self.agents:
            obs = agent.obs_string(self.world)
            execute_action(agent, self.world, obs)
        
        self.world.step_fields()
        
        if self.step_count % 100 == 0:
            self.world.scatter_food(self.rng, num_patches=2, patch_radius=4, amount=80)
        
        home_x, home_y = self.home_center
        current_food = 0
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                if dx * dx + dy * dy <= 64:
                    current_food += self.world.read_cell("FOOD", home_x + dx, home_y + dy)
        
        if current_food > self.food_collected:
            self.total_food_gathered += (current_food - self.food_collected)
        self.food_collected = current_food
        
        self.step_count += 1
    
    def render_frame(self, frame):
        self.update_step()
        
        self.ax_main.clear()
        self.ax_info.clear()
        
        render_world(self.world, self.agents, ax=self.ax_main, 
                    title=f"Step {self.step_count} - Live Foraging Simulation")
        
        self.ax_info.axis('off')
        
        info_text = f"SIMULATION STATUS\n"
        info_text += f"{'=' * 30}\n\n"
        info_text += f"Step: {self.step_count}\n\n"
        info_text += f"Agents: {len(self.agents)}\n\n"
        info_text += f"Food at Home: {self.food_collected}\n"
        info_text += f"Total Gathered: {self.total_food_gathered}\n\n"
        
        info_text += f"AGENT DETAILS\n"
        info_text += f"{'-' * 30}\n"
        for agent in self.agents:
            carry_status = "🟡 FOOD" if agent.carry_food > 0 else "⚪ EMPTY"
            info_text += f"Agent {agent.id}:\n"
            info_text += f"  Pos: ({agent.x}, {agent.y})\n"
            info_text += f"  Dir: {agent.d.name}\n"
            info_text += f"  Carry: {carry_status}\n\n"
        
        food_total = np.sum(self.world.layers["FOOD"])
        pher_food_total = np.sum(self.world.layers["PHER_FOOD"])
        pher_home_total = np.sum(self.world.layers["PHER_HOME"])
        
        info_text += f"WORLD STATE\n"
        info_text += f"{'-' * 30}\n"
        info_text += f"Total Food: {food_total}\n"
        info_text += f"Food Pheromone: {pher_food_total}\n"
        info_text += f"Home Pheromone: {pher_home_total}\n\n"
        
        info_text += f"LEGEND\n"
        info_text += f"{'-' * 30}\n"
        info_text += f"🟢 Green: Food\n"
        info_text += f"🔵 Blue: Home\n"
        info_text += f"⚪ Gray: Obstacles\n"
        info_text += f"🟡 Yellow: Agent (carrying)\n"
        info_text += f"⚪ White: Agent (empty)\n"
        info_text += f"🔴 Red tint: Pheromones\n"
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        return self.ax_main, self.ax_info
    
    def run(self, interval: int = 50):
        try:
            anim = FuncAnimation(self.fig, self.render_frame, interval=interval, 
                               blit=False, cache_frame_data=False)
            plt.tight_layout()
            plt.show()
        except KeyboardInterrupt:
            print(f"\nDemo stopped at step {self.step_count}")
            print(f"Total food gathered: {self.total_food_gathered}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Live 2D World Simulation - Infinite Loop Demo")
    parser.add_argument("--agents", type=int, default=5,
                        help="Number of agents (default: 5)")
    parser.add_argument("--world-size", type=int, default=128,
                        help="World size (default: 128)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--interval", type=int, default=50,
                        help="Animation interval in ms (default: 50)")
    
    args = parser.parse_args()
    
    demo = LiveDemo(num_agents=args.agents, world_size=args.world_size, seed=args.seed)
    demo.run(interval=args.interval)


if __name__ == "__main__":
    main()
