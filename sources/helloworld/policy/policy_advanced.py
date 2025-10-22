from typing import TYPE_CHECKING, Optional
import random

if TYPE_CHECKING:
    from core.expandable_world import ExpandableWorld
    from core.advanced_agent import AdvancedAgent

from logic.tools_advanced import (
    tool_hunt_food_advanced,
    tool_go_home_advanced,
    tool_find_nest_location,
    tool_find_storage_location,
    tool_should_build_nest,
    tool_should_build_storage,
    tool_should_leave_trail,
)
from logic.role_tools import (
    tool_explore_unknown,
    tool_follow_path_back,
)


def policy_advanced(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> str:
    if not agent.alive:
        return "NOOP"
    
    x, y = agent.x, agent.y
    H, W = world.H, world.W
    
    current_food = world.layers["FOOD"][y, x]
    current_home = world.layers["HOME"][y, x]
    current_storage = world.layers["STORAGE"][y, x]
    carrying = agent.carry_food > 0
    home_vec = world.nearest_home_vector(x, y)
    if home_vec is not None:
        home_distance = abs(home_vec[0]) + abs(home_vec[1])
    else:
        home_distance = None
    
    # 优先吃当前位置的食物，放宽能量阈值
    if current_food > 0 and not carrying and agent.energy < 45:
        return "EAT"
    
    # 能量低时回家
    if agent.energy < 30:
        action = tool_go_home_advanced(agent, world)
        if current_home > 0:
            return "EAT"
        return action
    
    # ===== 角色专用策略 =====
    
    # 探索者：专注于探索未知区域
    if agent.role == "Explorer":
        return policy_explorer(agent, world, carrying, current_food, current_home, current_storage, home_distance)
    
    # 建造者：专注于建造结构
    elif agent.role == "Builder":
        return policy_builder(agent, world, carrying, current_food, current_home, current_storage, home_distance)
    
    # 采集者：专注于采集和运输食物（默认策略）
    else:
        return policy_gatherer(agent, world, carrying, current_food, current_home, current_storage, home_distance)
    
    # 携带食物时优先返回
    if carrying:
        # 偶尔留下路径标记（降低概率）
        if agent.energy > 20 and tool_should_leave_trail(agent, world):
            if random.random() < 0.05:  # 降低到5%
                return "BUILD_TRAIL"

        if home_distance is not None and home_distance > 18 and random.random() < 0.3:
            return tool_go_home_advanced(agent, world)
        
        action = tool_go_home_advanced(agent, world)
        if current_home > 0 or current_storage > 0:
            return "DROP"
        return action
    
    if home_distance is not None and home_distance > max(W, H) * 0.38:
        if random.random() < 0.6:
            return tool_go_home_advanced(agent, world)
    
    # 没有食物时，寻找食物（优先级提高）
    if not carrying:
        if current_food > 0:
            if agent.energy < 70 and random.random() < 0.5:
                return "EAT"
            return "PICK"
        
        # 计算周围食物
        nearby_food = 0
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                check_x = x + dx
                check_y = y + dy
                if world.in_bounds(check_x, check_y):
                    nearby_food += int(world.layers["FOOD"][check_y, check_x])
        
        # 如果周围食物很多，降低探索倾向，多留在这里
        if nearby_food > 300:
            if random.random() < 0.3:
                return random.choice(["TL", "TR", "NOOP"])
        
        # 寻找食物优先级最高
        action = tool_hunt_food_advanced(agent, world)
        
        # 建造逻辑移到最后，降低优先级（只在非常富余时才建造）
        # 必须有大量食物和高能量才考虑建造
        if nearby_food > 500 and agent.energy > 70 and agent.genome.building_tendency > 0.5:
            if tool_should_build_nest(agent, world):
                if random.random() < 0.1:  # 大幅降低概率到10%
                    return "BUILD_NEST"
        
        if nearby_food > 400 and agent.energy > 60 and agent.genome.building_tendency > 0.6:
            if tool_should_build_storage(agent, world):
                if random.random() < 0.08:  # 大幅降低概率到8%
                    return "BUILD_STORAGE"
        
        # 偶尔留下路径（最低优先级）
        if agent.energy > 30 and tool_should_leave_trail(agent, world):
            if random.random() < 0.03:  # 降低到3%
                return "BUILD_TRAIL"
        
        return action
    
    # 默认：小幅度随机转向
    if random.random() < 0.1:
        return random.choice(["TL", "TR", "NOOP"])
    
    return "F"


def policy_explorer(agent: 'AdvancedAgent', world: 'ExpandableWorld', 
                   carrying: bool, current_food: int, current_home: int, 
                   current_storage: int, home_distance: Optional[int]) -> str:
    """探索者策略：优先探索未知区域，发现食物后标记"""
    x, y = agent.x, agent.y
    
    # 如果携带食物，快速返回
    if carrying:
        # 尝试使用记忆路径返回
        path_action = tool_follow_path_back(agent, world)
        if path_action:
            return path_action
        
        action = tool_go_home_advanced(agent, world)
        if current_home > 0 or current_storage > 0:
            return "DROP"
        return action
    
    # 发现食物，留下强烈信息素标记
    if current_food > 0:
        # 偶尔采集一些
        if agent.energy < 60 and random.random() < 0.3:
            return "PICK"
        # 主要是标记
        current = world.read_cell("PHER_FOOD", x, y)
        world.write_cell("PHER_FOOD", x, y, min(255, current + 50))
    
    # 远离家太远，返回
    if home_distance and home_distance > 30:
        if random.random() < 0.7:
            return tool_go_home_advanced(agent, world)
    
    # 主要工作：探索未知区域
    action = tool_explore_unknown(agent, world)
    return action


def policy_gatherer(agent: 'AdvancedAgent', world: 'ExpandableWorld',
                   carrying: bool, current_food: int, current_home: int,
                   current_storage: int, home_distance: Optional[int]) -> str:
    """采集者策略：优先采集和运输食物，使用最短路径"""
    x, y = agent.x, agent.y
    H, W = world.H, world.W
    
    # 携带食物时优先返回
    if carrying:
        # 尝试使用记忆的路径返回（最短路径）
        path_action = tool_follow_path_back(agent, world)
        if path_action:
            return path_action
        
        # 如果路径失效，使用标准寻路
        action = tool_go_home_advanced(agent, world)
        if current_home > 0 or current_storage > 0:
            return "DROP"
        return action
    
    if home_distance is not None and home_distance > max(W, H) * 0.38:
        if random.random() < 0.6:
            return tool_go_home_advanced(agent, world)
    
    # 没有食物时，寻找食物（优先级提高）
    if not carrying:
        if current_food > 0:
            if agent.energy < 70 and random.random() < 0.5:
                return "EAT"
            return "PICK"
        
        # 计算周围食物
        nearby_food = 0
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                check_x = x + dx
                check_y = y + dy
                if world.in_bounds(check_x, check_y):
                    nearby_food += int(world.layers["FOOD"][check_y, check_x])
        
        # 如果周围食物很多，降低探索倾向，多留在这里
        if nearby_food > 300:
            if random.random() < 0.3:
                return random.choice(["TL", "TR", "NOOP"])
        
        # 寻找食物
        action = tool_hunt_food_advanced(agent, world)
        return action
    
    return "F"


def policy_builder(agent: 'AdvancedAgent', world: 'ExpandableWorld',
                  carrying: bool, current_food: int, current_home: int,
                  current_storage: int, home_distance: Optional[int]) -> str:
    """建造者策略：专注于建造结构，需要时采集食物"""
    x, y = agent.x, agent.y
    
    # 如果携带食物，返回
    if carrying:
        action = tool_go_home_advanced(agent, world)
        if current_home > 0 or current_storage > 0:
            return "DROP"
        return action
    
    # 计算周围食物
    nearby_food = 0
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            check_x = x + dx
            check_y = y + dy
            if world.in_bounds(check_x, check_y):
                nearby_food += int(world.layers["FOOD"][check_y, check_x])
    
    # 能量充足且食物丰富，优先建造
    if agent.energy > 60 and nearby_food > 400:
        # 建造巢穴
        if tool_should_build_nest(agent, world):
            if random.random() < 0.4:  # 建造者有更高概率建造
                return "BUILD_NEST"
        
        # 建造储藏室
        if tool_should_build_storage(agent, world):
            if random.random() < 0.35:
                return "BUILD_STORAGE"
        
        # 留下路径
        if tool_should_leave_trail(agent, world):
            if random.random() < 0.15:
                return "BUILD_TRAIL"
    
    # 需要食物时去采集
    if current_food > 0:
        if agent.energy < 80:
            return "EAT"
        if random.random() < 0.5:
            return "PICK"
    
    # 寻找食物丰富的地方建造
    action = tool_hunt_food_advanced(agent, world)
    return action
