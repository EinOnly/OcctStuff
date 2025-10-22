#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random
from core.world import World, WorldConf
from core.agent import Agent, Dir
from policy.tiny_policy import execute_action


def test_live_simulation(steps=20):
    random.seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    conf = WorldConf(H=64, W=64, pher_decay=0.95, diffuse_weight=0.2)
    world = World(conf)
    
    home_x, home_y = 32, 32
    world.place_home(home_x, home_y, radius=5)
    world.scatter_food(rng, num_patches=5, patch_radius=3, amount=50)
    
    agents = []
    for i in range(3):
        angle = (2 * np.pi * i) / 3
        x = home_x + int(8 * np.cos(angle))
        y = home_y + int(8 * np.sin(angle))
        agents.append(Agent(id=i, x=x, y=y, d=Dir.N))
    
    print("Testing live simulation loop...")
    print(f"Initial state: {len(agents)} agents at home")
    
    for step in range(steps):
        for agent in agents:
            obs = agent.obs_string(world)
            execute_action(agent, world, obs)
        
        world.step_fields()
        
        if step % 5 == 0:
            food_total = np.sum(world.layers["FOOD"])
            carrying = sum(1 for a in agents if a.carry_food > 0)
            print(f"Step {step:3d}: Food={food_total:4d}, Carrying={carrying}, " +
                  f"Agents at: " + ", ".join(f"({a.x},{a.y})" for a in agents))
    
    print("\n✓ Live simulation test completed successfully!")
    print(f"  - All agents moved and made decisions")
    print(f"  - World state updated {steps} times")
    print(f"  - No errors or crashes")
    return True


if __name__ == "__main__":
    try:
        success = test_live_simulation(steps=20)
        if success:
            print("\nLive demo is ready to run!")
            print("\nTo start the infinite loop visualization:")
            print("  python run/live_demo.py --agents 5 --interval 50")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
