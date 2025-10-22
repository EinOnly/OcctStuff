from typing import TYPE_CHECKING, Tuple, Optional
import random

if TYPE_CHECKING:
    from core.expandable_world import ExpandableWorld
    from core.advanced_agent import AdvancedAgent
from core.advanced_agent import Dir


def tool_hunt_food_advanced(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> str:
    x, y = agent.x, agent.y
    d = agent.d
    H, W = world.H, world.W
    
    food_attraction = agent.genome.food_attraction
    exploration_rate = agent.genome.exploration_rate
    pher_sensitivity = agent.genome.pheromone_sensitivity
    
    candidates = {
        "F": d,
        "TL": d.turn_left(),
        "TR": d.turn_right(),
    }
    
    best_action = "F"
    best_score = -1.0
    
    for action, heading in candidates.items():
        dx, dy = heading.forward_delta()
        nx = x + dx
        ny = y + dy
        # Don't wrap around - check bounds instead
        if not world.in_bounds(nx, ny) or world.layers["SOLID"][ny, nx] > 0:
            score = -999.0
            if score > best_score:
                best_score = score
                best_action = action
            continue
        food_val = float(world.layers["FOOD"][ny, nx])
        pher_val = float(world.layers["PHER_FOOD"][ny, nx])
        trail_val = float(world.layers["TRAIL"][ny, nx])
        score = (
            food_val * food_attraction * 2.2 +
            pher_val * pher_sensitivity * 1.1 +
            trail_val * 0.12
        )
        if score > best_score:
            best_score = score
            best_action = action
    
    if best_score > 20 and not _would_hit_wall(agent, world, best_action):
        return best_action
    
    pher_current = world.layers["PHER_FOOD"][y, x]
    if pher_current > 25 and random.random() < pher_sensitivity:
        return "F"
    
    if best_score < 1 and random.random() < 0.6:
        return _random_turn(agent, world)
    
    if random.random() < exploration_rate * 0.7:
        choice = random.choice(["F", "TL", "TR"])
        if _would_hit_wall(agent, world, choice):
            return _random_turn(agent, world)
        return choice
    
    if _would_hit_wall(agent, world, best_action):
        return _random_turn(agent, world)
    return best_action


def tool_go_home_advanced(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> str:
    x, y = agent.x, agent.y
    d = agent.d
    H, W = world.H, world.W
    
    home_attraction = agent.genome.home_attraction
    pher_sensitivity = agent.genome.pheromone_sensitivity
    
    candidates = {
        "F": d,
        "TL": d.turn_left(),
        "TR": d.turn_right(),
    }
    
    best_action = "F"
    best_score = -1.0
    
    for action, heading in candidates.items():
        dx, dy = heading.forward_delta()
        nx = x + dx
        ny = y + dy
        # Don't wrap around - check bounds instead
        if not world.in_bounds(nx, ny) or world.layers["SOLID"][ny, nx] > 0:
            score = -999.0
            if score > best_score:
                best_score = score
                best_action = action
            continue
        home_val = float(world.layers["HOME"][ny, nx])
        pher_val = float(world.layers["PHER_HOME"][ny, nx])
        trail_val = float(world.layers["TRAIL"][ny, nx])
        score = (
            home_val * home_attraction * 2.5 +
            pher_val * pher_sensitivity * 1.3 +
            trail_val * 0.15
        )
        if score > best_score:
            best_score = score
            best_action = action
    
    if best_score > 10 and not _would_hit_wall(agent, world, best_action):
        return best_action
    
    pher_current = world.layers["PHER_HOME"][y, x]
    if pher_current > 20 and random.random() < pher_sensitivity:
        return "F"
    
    home_vec = world.nearest_home_vector(x, y)
    if home_vec is not None:
        dx, dy = home_vec
        if abs(dx) + abs(dy) > 2:
            desired_dir = _vector_to_dir(dx, dy)
            if desired_dir is not None:
                turn_action = _turn_towards(agent.d, desired_dir)
                if not _would_hit_wall(agent, world, turn_action):
                    return turn_action
    
    if random.random() < 0.4:
        return _random_turn(agent, world)
    
    if _would_hit_wall(agent, world, best_action):
        return _random_turn(agent, world)
    return best_action


def tool_find_nest_location(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> Optional[Tuple[int, int]]:
    x, y = agent.x, agent.y
    H, W = world.H, world.W
    
    best_score = -1
    best_loc = None
    
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            nx = x + dx
            ny = y + dy
            
            if not world.in_bounds(nx, ny):
                continue
            if world.layers["SOLID"][ny, nx] > 0:
                continue
            if world.layers["NEST"][ny, nx] > 0:
                continue
            
            food_nearby = 0
            home_nearby = 0
            for fx in range(-2, 3):
                for fy in range(-2, 3):
                    check_x = nx + fx
                    check_y = ny + fy
                    if world.in_bounds(check_x, check_y):
                        food_nearby += world.layers["FOOD"][check_y, check_x]
                        home_nearby += world.layers["HOME"][check_y, check_x]
            
            score = food_nearby * 0.5 + home_nearby * 0.3
            
            if score > best_score:
                best_score = score
                best_loc = (nx, ny)
    
    return best_loc if best_score > 100 else None


def tool_find_storage_location(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> Optional[Tuple[int, int]]:
    x, y = agent.x, agent.y
    H, W = world.H, world.W
    
    best_score = -1
    best_loc = None
    
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            nx = x + dx
            ny = y + dy
            
            if not world.in_bounds(nx, ny):
                continue
            if world.layers["SOLID"][ny, nx] > 0:
                continue
            if world.layers["STORAGE"][ny, nx] > 0:
                continue
            
            nest_nearby = 0
            trail_nearby = 0
            for fx in range(-1, 2):
                for fy in range(-1, 2):
                    check_x = nx + fx
                    check_y = ny + fy
                    if world.in_bounds(check_x, check_y):
                        nest_nearby += world.layers["NEST"][check_y, check_x]
                        trail_nearby += world.layers["TRAIL"][check_y, check_x]
            
            score = nest_nearby * 0.6 + trail_nearby * 0.2
            
            if score > best_score:
                best_score = score
                best_loc = (nx, ny)
    
    return best_loc if best_score > 50 else None


def tool_should_build_nest(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> bool:
    # 提高能量要求
    if agent.energy < 60:
        return False
    
    if random.random() > agent.genome.building_tendency:
        return False
    
    x, y = agent.x, agent.y
    H, W = world.H, world.W
    
    # 检查周围必须有足够的食物才能建造
    nearby_food = 0
    nest_count = 0
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            check_x = x + dx
            check_y = y + dy
            if world.in_bounds(check_x, check_y):
                nearby_food += world.layers["FOOD"][check_y, check_x]
                nest_count += world.layers["NEST"][check_y, check_x]
    
    # 必须周围有大量食物才能建造
    if nearby_food < 500:
        return False
    
    if nest_count > 50:
        return False
    
    return True


def tool_should_build_storage(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> bool:
    # 提高能量要求
    if agent.energy < 50:
        return False
    
    if random.random() > agent.genome.building_tendency * 0.8:
        return False
    
    x, y = agent.x, agent.y
    H, W = world.H, world.W
    
    # 检查周围必须有足够的食物才能建造
    nearby_food = 0
    storage_count = 0
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            check_x = x + dx
            check_y = y + dy
            if world.in_bounds(check_x, check_y):
                nearby_food += world.layers["FOOD"][check_y, check_x]
                storage_count += world.layers["STORAGE"][check_y, check_x]
    
    # 必须周围有足够食物才能建造
    if nearby_food < 300:
        return False
    
    if storage_count > 30:
        return False
    
    return True


def tool_should_leave_trail(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> bool:
    if agent.energy < 25:
        return False
    
    if random.random() > agent.genome.building_tendency * 1.2:
        return False
    
    x, y = agent.x, agent.y
    
    if world.layers["TRAIL"][y, x] > 100:
        return False
    
    return True


def _vector_to_dir(dx: int, dy: int) -> Optional[Dir]:
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        if dx > 0:
            return Dir.E
        elif dx < 0:
            return Dir.W
    if dy > 0:
        return Dir.S
    elif dy < 0:
        return Dir.N
    return None


def _turn_towards(current: Dir, target: Dir) -> str:
    if current == target:
        return "F"
    if current.turn_left() == target:
        return "TL"
    if current.turn_right() == target:
        return "TR"
    if current.turn_left().turn_left() == target:
        return random.choice(["TL", "TR"])
    return "F"


def _would_hit_wall(agent: 'AdvancedAgent', world: 'ExpandableWorld', action: str) -> bool:
    action = action or "F"
    if action not in ("F", "MOVE_FWD"):
        return False
    dx, dy = agent.d.forward_delta()
    nx = agent.x + dx
    ny = agent.y + dy
    # Check bounds and solid walls
    if not world.in_bounds(nx, ny):
        return True
    return world.layers["SOLID"][ny, nx] > 0


def _random_turn(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> str:
    options = ["TL", "TR", "NOOP"]
    random.shuffle(options)
    for option in options:
        if not _would_hit_wall(agent, world, option):
            return option
    return "NOOP"
