import random
from logic.tools import tool_hunt_food, tool_go_home, tool_drop_breadcrumb


def decide_action(obs: str) -> str:
    parts = obs.split(";")
    carry = 0
    
    for part in parts:
        if part.startswith("CARRY="):
            carry = int(part.split("=")[1])
            break
    
    action_choice = random.random()
    
    if carry > 0:
        if action_choice < 0.7:
            return "TOOL:GO_HOME"
        elif action_choice < 0.85:
            return "TOOL:DROP_BREADCRUMB"
        else:
            return "DROP_FOOD"
    else:
        if action_choice < 0.7:
            return "TOOL:HUNT_FOOD"
        elif action_choice < 0.85:
            return "TOOL:DROP_BREADCRUMB"
        else:
            return "PICK_FOOD"


def execute_action(agent, world, obs: str) -> None:
    action = decide_action(obs)
    
    if action.startswith("TOOL:"):
        tool_name = action[5:]
        
        if tool_name == "HUNT_FOOD":
            primitive = tool_hunt_food(agent, world)
        elif tool_name == "GO_HOME":
            primitive = tool_go_home(agent, world)
        elif tool_name == "DROP_BREADCRUMB":
            primitive = tool_drop_breadcrumb(agent, world)
        else:
            primitive = "NOP"
        
        agent.apply_action(world, primitive)
    else:
        agent.apply_action(world, action)
