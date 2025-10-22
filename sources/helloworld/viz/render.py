import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.world import World
    from core.agent import Agent, Dir


def render_world(world: 'World', agents: List['Agent'], ax=None, title: str = "World") -> None:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        ax.clear()
    
    H, W = world.H, world.W
    img = np.zeros((H, W, 3), dtype=np.float32)
    
    solid = world.layers["SOLID"].astype(np.float32) / 255.0
    img[:, :, 0] += solid * 0.3
    img[:, :, 1] += solid * 0.3
    img[:, :, 2] += solid * 0.3
    
    food = world.layers["FOOD"].astype(np.float32) / 255.0
    img[:, :, 1] += food * 0.7
    
    home = world.layers["HOME"].astype(np.float32) / 255.0
    img[:, :, 2] += home * 0.5
    
    pher_food = world.layers["PHER_FOOD"].astype(np.float32) / 255.0
    img[:, :, 0] += pher_food * 0.3
    img[:, :, 1] += pher_food * 0.3
    
    pher_home = world.layers["PHER_HOME"].astype(np.float32) / 255.0
    img[:, :, 2] += pher_home * 0.2
    
    tape = world.layers["TAPE"].astype(np.float32) / 255.0
    img[:, :, 0] += tape * 0.15
    img[:, :, 2] += tape * 0.15
    
    mark = world.layers["MARK"].astype(np.float32) / 255.0
    img[:, :, 0] += mark * 0.2
    img[:, :, 1] += mark * 0.1
    
    img = np.clip(img, 0, 1)
    
    ax.imshow(img, origin='upper', interpolation='nearest')
    
    for agent in agents:
        x, y = agent.x, agent.y
        d = agent.d
        
        size = 2.0
        if d.name == 'N':
            tri = [[x, y - size], [x - size * 0.6, y + size * 0.6], [x + size * 0.6, y + size * 0.6]]
        elif d.name == 'E':
            tri = [[x + size, y], [x - size * 0.6, y - size * 0.6], [x - size * 0.6, y + size * 0.6]]
        elif d.name == 'S':
            tri = [[x, y + size], [x - size * 0.6, y - size * 0.6], [x + size * 0.6, y - size * 0.6]]
        else:
            tri = [[x - size, y], [x + size * 0.6, y - size * 0.6], [x + size * 0.6, y + size * 0.6]]
        
        color = 'yellow' if agent.carry_food > 0 else 'white'
        polygon = Polygon(tri, closed=True, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(polygon)
    
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')


def animate_run(frames: List[dict], save_path: Optional[str] = None) -> None:
    if not frames:
        return
    
    try:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for i, frame in enumerate(frames):
            world = frame['world']
            agents = frame['agents']
            step = frame.get('step', i)
            
            render_world(world, agents, ax=ax, title=f"Step {step}")
            
            if save_path and i == len(frames) - 1:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
            
            plt.pause(0.01)
        
        plt.ioff()
        plt.show()
        
    except Exception as e:
        print(f"Visualization error (running headless?): {e}")
        print("Continuing without visualization...")
