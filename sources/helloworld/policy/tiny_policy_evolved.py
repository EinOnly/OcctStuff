import random
from logic.tools_evolved import tool_hunt_food_evolved, tool_go_home_evolved, tool_drop_breadcrumb_evolved


def decide_action_evolved(obs: str, agent) -> str:
    parts = obs.split(";")
    carry = 0
    energy = 100
    
    for part in parts:
        if part.startswith("CARRY="):
            carry = int(part.split("=")[1])
        elif part.startswith("ENERGY="):
            energy = int(part.split("=")[1])
    
    if energy < 10:
        return "TOOL:HUNT_FOOD"
    
    action_choice = random.random()
    
    if carry > 0:
        if action_choice < 0.6:
            return "TOOL:GO_HOME"
        elif action_choice < 0.75:
            return "TOOL:DROP_BREADCRUMB"
        elif action_choice < 0.95:
            return "MOVE_FWD"
        else:
            return "DROP_FOOD"
    else:
        if action_choice < 0.6:
            return "TOOL:HUNT_FOOD"
        elif action_choice < 0.75:
            return "TOOL:DROP_BREADCRUMB"
        elif action_choice < 0.95:
            return "MOVE_FWD"
        else:
            return "PICK_FOOD"


def execute_action_evolved(agent, world, obs: str) -> None:
    action = decide_action_evolved(obs, agent)
    
    if action.startswith("TOOL:"):
        tool_name = action[5:]
        
        if tool_name == "HUNT_FOOD":
            primitive = tool_hunt_food_evolved(agent, world)
        elif tool_name == "GO_HOME":
            primitive = tool_go_home_evolved(agent, world)
        elif tool_name == "DROP_BREADCRUMB":
            primitive = tool_drop_breadcrumb_evolved(agent, world)
        else:
            primitive = "NOP"
        
        agent.apply_action(world, primitive)
    else:
        agent.apply_action(world, action)
