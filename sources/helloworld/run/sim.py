import sys
import os
import argparse
import numpy as np
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.world import World, WorldConf
from core.agent import Agent, Dir
from policy.tiny_policy import execute_action
from logic.bf_interpreter import BFInterpreter
from viz.render import render_world, animate_run


def run_foraging_demo(num_agents: int = 5, steps: int = 1000, seed: int = 42, render_interval: int = 50):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    conf = WorldConf(H=128, W=128, pher_decay=0.95, diffuse_weight=0.2)
    world = World(conf)
    
    home_x, home_y = conf.W // 2, conf.H // 2
    world.place_home(home_x, home_y, radius=8)
    
    world.scatter_food(rng, num_patches=15, patch_radius=5, amount=100)
    
    world.place_obstacle_rect(0, 0, conf.W, 3)
    world.place_obstacle_rect(0, 0, 3, conf.H)
    world.place_obstacle_rect(conf.W - 3, 0, conf.W, conf.H)
    world.place_obstacle_rect(0, conf.H - 3, conf.W, conf.H)
    
    agents = []
    for i in range(num_agents):
        angle = (2 * np.pi * i) / num_agents
        offset = 12
        x = home_x + int(offset * np.cos(angle))
        y = home_y + int(offset * np.sin(angle))
        d = Dir(rng.integers(0, 4))
        agents.append(Agent(id=i, x=x, y=y, d=d))
    
    print(f"Starting foraging demo: {num_agents} agents, {steps} steps, seed={seed}")
    
    frames = []
    
    for step in range(steps):
        for agent in agents:
            obs = agent.obs_string(world)
            execute_action(agent, world, obs)
        
        world.step_fields()
        
        if step % render_interval == 0:
            print(f"Step {step}/{steps}")
            import copy
            frames.append({
                'world': copy.deepcopy(world),
                'agents': copy.deepcopy(agents),
                'step': step
            })
    
    print(f"Foraging demo complete. Collected {len(frames)} snapshots.")
    
    if frames:
        animate_run(frames[:20])
    
    food_at_home = 0
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            if dx * dx + dy * dy <= 64:
                food_at_home += world.read_cell("FOOD", home_x + dx, home_y + dy)
    
    print(f"Final food at home area: {food_at_home}")


def run_bf_demo(steps: int = 500, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    conf = WorldConf(H=64, W=128, pher_decay=0.99, diffuse_weight=0.0)
    world = World(conf)
    
    bf_program = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++."
    
    interpreter = BFInterpreter(program_marks_layer="MARK", tape_layer="TAPE")
    interpreter.load_program(world, bf_program, start_x=5, start_y=5)
    
    dummy_agent = Agent(id=0, x=5, y=5, d=Dir.E)
    
    print(f"Starting BF demo: '{bf_program[:50]}...' ({len(bf_program)} chars), {steps} steps, seed={seed}")
    
    frames = []
    snapshot_interval = max(1, steps // 20)
    
    for step in range(steps):
        if interpreter.halted:
            print(f"BF program halted at step {step}")
            break
        
        interpreter.step(dummy_agent, world)
        
        if step % snapshot_interval == 0:
            print(f"Step {step}/{steps}, PC={interpreter.pc}, Output so far: {interpreter.get_output()}")
            import copy
            frames.append({
                'world': copy.deepcopy(world),
                'agents': [copy.deepcopy(dummy_agent)],
                'step': step
            })
    
    output = interpreter.get_output()
    print(f"\nBF program complete!")
    print(f"Output: {output}")
    print(f"Output (hex): {' '.join(f'{ord(c):02x}' for c in output)}")
    
    if frames:
        animate_run(frames[:10])


def main():
    parser = argparse.ArgumentParser(description="2D Finite Pixel World Simulation")
    parser.add_argument("--demo", type=str, choices=["foraging", "bf"], required=True,
                        help="Demo to run: foraging or bf")
    parser.add_argument("--agents", type=int, default=5,
                        help="Number of agents for foraging demo")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--render-interval", type=int, default=50,
                        help="Render snapshot every N steps")
    
    args = parser.parse_args()
    
    if args.demo == "foraging":
        steps = args.steps if args.steps is not None else 1000
        run_foraging_demo(
            num_agents=args.agents,
            steps=steps,
            seed=args.seed,
            render_interval=args.render_interval
        )
    elif args.demo == "bf":
        steps = args.steps if args.steps is not None else 500
        run_bf_demo(steps=steps, seed=args.seed)


if __name__ == "__main__":
    main()
