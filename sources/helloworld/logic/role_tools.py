"""
角色专用工具函数
"""
from typing import TYPE_CHECKING, Optional, Tuple
import random

if TYPE_CHECKING:
    from core.expandable_world import ExpandableWorld
    from core.advanced_agent import AdvancedAgent
from core.advanced_agent import Dir


def tool_explore_unknown(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> str:
    """探索者专用：寻找未探索的区域"""
    x, y = agent.x, agent.y
    d = agent.d
    
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
        
        # 检查是否能走
        if not world.in_bounds(nx, ny) or world.layers["SOLID"][ny, nx] > 0:
            continue
        
        # 优先去未探索的地方
        if (nx, ny) not in agent.explored_cells:
            score = 100.0  # 高分给未探索区域
        else:
            score = 10.0
        
        # 也要考虑周围的未探索密度
        unexplored_nearby = 0
        for dy2 in range(-2, 3):
            for dx2 in range(-2, 3):
                check_x = nx + dx2
                check_y = ny + dy2
                if world.in_bounds(check_x, check_y):
                    if (check_x, check_y) not in agent.explored_cells:
                        unexplored_nearby += 1
        
        score += unexplored_nearby * 2.0
        
        # 也留意食物，为团队提供信息
        food_val = float(world.layers["FOOD"][ny, nx])
        score += food_val * 0.3
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action


def tool_follow_path_back(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> Optional[str]:
    """沿着记录的路径返回（最短路径）"""
    if not agent.path_to_food:
        return None
    
    # 反向走回去
    if len(agent.path_to_food) > 0:
        target_x, target_y, target_dir = agent.path_to_food[-1]
        
        # 如果已经到达这个点，从路径中移除
        if agent.x == target_x and agent.y == target_y:
            agent.path_to_food.pop()
            if len(agent.path_to_food) == 0:
                return None
            target_x, target_y, target_dir = agent.path_to_food[-1]
        
        # 计算如何移动到目标点
        dx = target_x - agent.x
        dy = target_y - agent.y
        
        # 确定需要的方向
        desired_dir = None
        if abs(dx) >= abs(dy):
            if dx > 0:
                desired_dir = Dir.E
            elif dx < 0:
                desired_dir = Dir.W
        else:
            if dy > 0:
                desired_dir = Dir.S
            elif dy < 0:
                desired_dir = Dir.N
        
        if desired_dir is None:
            return None
        
        # 转向或前进
        if agent.d == desired_dir:
            return "F"
        elif agent.d.turn_left() == desired_dir:
            return "TL"
        elif agent.d.turn_right() == desired_dir:
            return "TR"
        else:
            # 需要转180度
            return random.choice(["TL", "TR"])
    
    return None


def count_role_in_area(world: 'ExpandableWorld', agents: list, role: str, cx: int, cy: int, radius: int) -> int:
    """统计某个区域内特定角色的数量"""
    count = 0
    for agent in agents:
        if not agent.alive or agent.role != role:
            continue
        dx = agent.x - cx
        dy = agent.y - cy
        if dx * dx + dy * dy <= radius * radius:
            count += 1
    return count


def get_nearest_unexplored_direction(agent: 'AdvancedAgent', world: 'ExpandableWorld') -> Optional[Dir]:
    """获取最近的未探索方向"""
    x, y = agent.x, agent.y
    best_dist = None
    best_dir = None
    
    # 在周围寻找未探索的单元格
    for radius in range(1, 20):
        for angle in range(0, 360, 30):
            import math
            dx = int(radius * math.cos(math.radians(angle)))
            dy = int(radius * math.sin(math.radians(angle)))
            check_x = x + dx
            check_y = y + dy
            
            if not world.in_bounds(check_x, check_y):
                continue
            
            if (check_x, check_y) not in agent.explored_cells:
                dist = abs(dx) + abs(dy)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    # 确定方向
                    if abs(dx) >= abs(dy):
                        best_dir = Dir.E if dx > 0 else Dir.W
                    else:
                        best_dir = Dir.S if dy > 0 else Dir.N
        
        if best_dir is not None:
            break
    
    return best_dir
