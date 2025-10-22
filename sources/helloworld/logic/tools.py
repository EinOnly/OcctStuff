import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent import Agent as AgentType
    from core.world import World as WorldType
else:
    AgentType = 'Agent'
    WorldType = 'World'

from core.agent import Agent, Dir
from core.world import World


def tool_hunt_food(agent: Agent, world: World) -> str:
    best_val = -1
    best_actions = []
    
    actions = ["MOVE_FWD", "TURN_L", "TURN_R"]
    
    for action in actions:
        test_agent = Agent(
            id=agent.id,
            x=agent.x,
            y=agent.y,
            d=agent.d,
            carry_food=agent.carry_food
        )
        test_agent.apply_action(world, action)
        
        food = world.read_cell("FOOD", test_agent.x, test_agent.y)
        pher_food = world.read_cell("PHER_FOOD", test_agent.x, test_agent.y)
        
        combined = food * 2 + pher_food
        
        if combined > best_val:
            best_val = combined
            best_actions = [action]
        elif combined == best_val:
            best_actions.append(action)
    
    if best_val > 0 and best_actions:
        return random.choice(best_actions)
    
    if random.random() < 0.3:
        return "MOVE_FWD"
    else:
        return random.choice(["TURN_L", "TURN_R"])


def tool_go_home(agent: Agent, world: World) -> str:
    best_val = -1
    best_actions = []
    
    actions = ["MOVE_FWD", "TURN_L", "TURN_R"]
    
    for action in actions:
        test_agent = Agent(
            id=agent.id,
            x=agent.x,
            y=agent.y,
            d=agent.d,
            carry_food=agent.carry_food
        )
        test_agent.apply_action(world, action)
        
        home = world.read_cell("HOME", test_agent.x, test_agent.y)
        pher_home = world.read_cell("PHER_HOME", test_agent.x, test_agent.y)
        
        combined = home * 2 + pher_home
        
        if combined > best_val:
            best_val = combined
            best_actions = [action]
        elif combined == best_val:
            best_actions.append(action)
    
    if best_val > 0 and best_actions:
        return random.choice(best_actions)
    
    if random.random() < 0.3:
        return "MOVE_FWD"
    else:
        return random.choice(["TURN_L", "TURN_R"])


def tool_drop_breadcrumb(agent: Agent, world: World) -> str:
    if agent.carry_food > 0:
        return "DEPOSIT:HOME:30"
    else:
        return "DEPOSIT:FOOD:30"
