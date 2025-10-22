import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Polygon, FancyBboxPatch
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.expandable_world import ExpandableWorld
    from core.advanced_agent import AdvancedAgent


def render_advanced_world(world: 'ExpandableWorld', agents: List['AdvancedAgent'], 
                          ax=None, title: str = "World", show_grid: bool = False) -> None:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        ax.clear()
    
    H, W = world.H, world.W
    img = np.zeros((H, W, 3), dtype=np.float32)
    
    # 先渲染SOLID作为基础层（白色）
    solid = world.layers["SOLID"].astype(np.float32)
    # 但是如果有建筑物，就不显示白色SOLID
    nest = world.layers["NEST"].astype(np.float32)
    storage = world.layers["STORAGE"].astype(np.float32)
    has_building = (nest > 0) | (storage > 0)
    
    # 只在没有建筑物的地方显示白色SOLID
    solid_display = solid * (~has_building)
    img[:, :, 0] += solid_display * 1.0  # 白色
    img[:, :, 1] += solid_display * 1.0
    img[:, :, 2] += solid_display * 1.0
    
    food = world.layers["FOOD"].astype(np.float32) / 255.0
    img[:, :, 1] += food * 0.9
    img[:, :, 0] += food * 0.3
    
    home = world.layers["HOME"].astype(np.float32) / 255.0
    img[:, :, 2] += home * 0.8
    img[:, :, 0] += home * 0.2
    
    # 建筑物使用更明显的颜色（覆盖SOLID）
    nest_norm = nest / 255.0
    img[:, :, 0] += nest_norm * 0.9  # 增强红色
    img[:, :, 1] += nest_norm * 0.4
    img[:, :, 2] += nest_norm * 0.1
    
    storage_norm = storage / 255.0
    img[:, :, 0] += storage_norm * 0.2
    img[:, :, 1] += storage_norm * 0.8  # 增强绿色
    img[:, :, 2] += storage_norm * 0.6
    
    trail = world.layers["TRAIL"].astype(np.float32) / 255.0
    img[:, :, 0] += trail * 0.5
    img[:, :, 1] += trail * 0.4
    img[:, :, 2] += trail * 0.3
    
    # Enhanced pheromone visualization - make them much more visible
    pher_food = world.layers["PHER_FOOD"].astype(np.float32) / 255.0
    img[:, :, 0] += pher_food * 0.6  # Increased from 0.25
    img[:, :, 1] += pher_food * 0.5  # Increased from 0.2
    img[:, :, 2] += pher_food * 0.1  # Add slight blue for better contrast
    
    pher_home = world.layers["PHER_HOME"].astype(np.float32) / 255.0
    img[:, :, 2] += pher_home * 0.5  # Increased from 0.2
    img[:, :, 1] += pher_home * 0.3  # Increased from 0.1
    img[:, :, 0] += pher_home * 0.1  # Add slight red for better visibility
    
    img = np.clip(img, 0, 1)
    
    ax.imshow(img, origin='upper', interpolation='nearest', aspect='equal')
    
    for agent in agents:
        if not agent.alive:
            continue
        
        x, y = agent.x, agent.y
        d = agent.d
        size = agent.get_visual_size()
        color = agent.get_color()
        
        if d.name == 'N':
            body = Circle((x, y), size * 0.4, facecolor=color, edgecolor='black', linewidth=1.2, zorder=10)
            ax.add_patch(body)
            
            head = Circle((x, y - size * 0.5), size * 0.25, facecolor=color, edgecolor='black', linewidth=1, zorder=11)
            ax.add_patch(head)
            
        elif d.name == 'E':
            body = Circle((x, y), size * 0.4, facecolor=color, edgecolor='black', linewidth=1.2, zorder=10)
            ax.add_patch(body)
            
            head = Circle((x + size * 0.5, y), size * 0.25, facecolor=color, edgecolor='black', linewidth=1, zorder=11)
            ax.add_patch(head)
            
        elif d.name == 'S':
            body = Circle((x, y), size * 0.4, facecolor=color, edgecolor='black', linewidth=1.2, zorder=10)
            ax.add_patch(body)
            
            head = Circle((x, y + size * 0.5), size * 0.25, facecolor=color, edgecolor='black', linewidth=1, zorder=11)
            ax.add_patch(head)
            
        else:
            body = Circle((x, y), size * 0.4, facecolor=color, edgecolor='black', linewidth=1.2, zorder=10)
            ax.add_patch(body)
            
            head = Circle((x - size * 0.5, y), size * 0.25, facecolor=color, edgecolor='black', linewidth=1, zorder=11)
            ax.add_patch(head)
        
        # 显示代理角色标记
        role_symbol = ""
        if agent.role == "Explorer":
            role_symbol = "🔍"
        elif agent.role == "Gatherer":
            role_symbol = "🌾"
        elif agent.role == "Builder":
            role_symbol = "🏗"
        
        if role_symbol:
            role_text = ax.text(x + size * 0.6, y - size * 0.6, role_symbol,
                              fontsize=7, ha='left', va='top', zorder=12)
        
        if agent.generation > 0:
            gen_text = ax.text(x, y - size * 0.8, f"G{agent.generation}", 
                             fontsize=6, ha='center', va='top', 
                             color='white', weight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
            gen_text.set_zorder(12)
    
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, weight='bold', pad=10)
    
    if show_grid and W <= 64:
        ax.grid(True, alpha=0.2, linewidth=0.5)
    else:
        ax.axis('off')
    
    legend_elements = [
        mpatches.Patch(facecolor=(0.3, 0.9, 0.3), label='Food (食物)', edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor=(0.2, 0.1, 0.8), label='Home (家园)', edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor=(1.0, 1.0, 1.0), label='Solid (固体墙)', edgecolor='black', linewidth=1),
        mpatches.Patch(facecolor=(0.9, 0.4, 0.1), label='Nest (巢穴)', edgecolor='black', linewidth=1),
        mpatches.Patch(facecolor=(0.2, 0.8, 0.6), label='Storage (储藏)', edgecolor='black', linewidth=1),
        mpatches.Patch(facecolor=(0.5, 0.4, 0.3), label='Trail (路径)', edgecolor='black', linewidth=0.5),
        mpatches.Patch(facecolor=(0.6, 0.5, 0.1), label='Food Pher (食物信息素)', edgecolor='orange', linewidth=0.5),
        mpatches.Patch(facecolor=(0.1, 0.3, 0.5), label='Home Pher (返家信息素)', edgecolor='blue', linewidth=0.5),
        mpatches.Patch(facecolor=(0.3, 0.6, 1.0), label='🔍 Explorer (探索者)', edgecolor='blue', linewidth=1),
        mpatches.Patch(facecolor=(0.4, 0.9, 0.3), label='🌾 Gatherer (采集者)', edgecolor='green', linewidth=1),
        mpatches.Patch(facecolor=(1.0, 0.5, 0.2), label='🏗 Builder (建造者)', edgecolor='orange', linewidth=1),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9, 
              edgecolor='black', fancybox=True, shadow=True)


def create_elegant_info_panel(ax, title: str, data: dict, bg_color='lightblue'):
    ax.clear()
    ax.axis('off')
    
    y_pos = 0.95
    line_height = 0.05  # 增加行高（原来0.04）
    
    ax.text(0.5, y_pos, title, transform=ax.transAxes,
           fontsize=13, weight='bold', ha='center', va='top',
           bbox=dict(boxstyle='round,pad=0.6', facecolor=bg_color, alpha=0.8, edgecolor='black', linewidth=2))
    
    y_pos -= 0.10  # 增加标题后的间距
    
    for key, value in data.items():
        if key.startswith('__section__'):
            y_pos -= line_height * 0.4
            section_name = value
            ax.text(0.05, y_pos, section_name, transform=ax.transAxes,
                   fontsize=11, weight='bold', va='top', style='italic',
                   color='darkblue')
            y_pos -= line_height * 1.3
            
            ax.plot([0.05, 0.95], [y_pos + line_height * 0.3, y_pos + line_height * 0.3], 
                   transform=ax.transAxes, color='gray', linewidth=1, alpha=0.5)
            y_pos -= line_height * 0.4
        else:
            ax.text(0.08, y_pos, f"{key}:", transform=ax.transAxes,
                   fontsize=10, va='top', weight='bold')
            ax.text(0.55, y_pos, str(value), transform=ax.transAxes,
                   fontsize=10, va='top', family='monospace')
            y_pos -= line_height
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
