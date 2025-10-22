import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent_evolved import Agent, AgentGenome
    from core.world import World


def tool_hunt_food_evolved(agent: 'Agent', world: 'World') -> str:
    from core.agent_evolved import Agent
    
    best_val = -1
    best_actions = []
    
    actions = ["MOVE_FWD", "TURN_L", "TURN_R"]
    
    for action in actions:
        test_agent = Agent(
            id=agent.id,
            x=agent.x,
            y=agent.y,
            d=agent.d,
            genome=agent.genome,
            carry_food=agent.carry_food
        )
        test_agent.apply_action(world, action)
        
        food = world.read_cell("FOOD", test_agent.x, test_agent.y)
        pher_food = world.read_cell("PHER_FOOD", test_agent.x, test_agent.y)
        
        combined = (
            food * agent.genome.food_attraction * 2 + 
            pher_food * agent.genome.pheromone_sensitivity
        )
        
        if combined > best_val:
            best_val = combined
            best_actions = [action]
        elif combined == best_val:
            best_actions.append(action)
    
    if best_val > 0 and best_actions and random.random() > agent.genome.exploration_rate:
        return random.choice(best_actions)
    
    if random.random() < agent.genome.exploration_rate:
        return "MOVE_FWD"
    else:
        return random.choice(["TURN_L", "TURN_R"])


def tool_go_home_evolved(agent: 'Agent', world: 'World') -> str:
    from core.agent_evolved import Agent
    
    best_val = -1
    best_actions = []
    
    actions = ["MOVE_FWD", "TURN_L", "TURN_R"]
    
    for action in actions:
        test_agent = Agent(
            id=agent.id,
            x=agent.x,
            y=agent.y,
            d=agent.d,
            genome=agent.genome,
            carry_food=agent.carry_food
        )
        test_agent.apply_action(world, action)
        
        home = world.read_cell("HOME", test_agent.x, test_agent.y)
        pher_home = world.read_cell("PHER_HOME", test_agent.x, test_agent.y)
        
        combined = (
            home * agent.genome.home_attraction * 2 + 
            pher_home * agent.genome.pheromone_sensitivity
        )
        
        if combined > best_val:
            best_val = combined
            best_actions = [action]
        elif combined == best_val:
            best_actions.append(action)
    
    if best_val > 0 and best_actions and random.random() > agent.genome.exploration_rate:
        return random.choice(best_actions)
    
    if random.random() < agent.genome.exploration_rate:
        return "MOVE_FWD"
    else:
        return random.choice(["TURN_L", "TURN_R"])


def tool_drop_breadcrumb_evolved(agent: 'Agent', world: 'World') -> str:
    amount = int(30 * agent.genome.pheromone_sensitivity)
    if agent.carry_food > 0:
        return f"DEPOSIT:HOME:{amount}"
    else:
        return f"DEPOSIT:FOOD:{amount}"
