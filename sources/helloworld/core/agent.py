from enum import Enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import World


class Dir(Enum):
    N = 0
    E = 1
    S = 2
    W = 3
    
    def turn_left(self):
        return Dir((self.value - 1) % 4)
    
    def turn_right(self):
        return Dir((self.value + 1) % 4)
    
    def forward_delta(self):
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        return deltas[self.value]


@dataclass
class Agent:
    id: int
    x: int
    y: int
    d: Dir
    regs: list = field(default_factory=lambda: [0] * 8)
    stack: list = field(default_factory=list)
    carry_food: int = 0
    
    def obs_string(self, world: 'World') -> str:
        parts = []
        
        parts.append(f"ID={self.id}")
        parts.append(f"POS={self.x},{self.y}")
        parts.append(f"DIR={self.d.name}")
        parts.append(f"CARRY={self.carry_food}")
        
        nbr = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = self.x + dx, self.y + dy
                f = world.read_cell("FOOD", nx, ny)
                h = world.read_cell("HOME", nx, ny)
                pf = world.read_cell("PHER_FOOD", nx, ny)
                nbr.append(f"{f:02x}{h:02x}{pf:02x}")
        
        parts.append(f"NBR={''.join(nbr)}")
        
        tape_val = world.read_cell("TAPE", self.x, self.y)
        mark_val = world.read_cell("MARK", self.x, self.y)
        parts.append(f"TM={tape_val:02x}{mark_val:02x}")
        
        return ";".join(parts)
    
    def apply_action(self, world: 'World', action: str) -> None:
        action = action.strip()
        
        if action == "MOVE_FWD":
            dx, dy = self.d.forward_delta()
            nx, ny = self.x + dx, self.y + dy
            if world.in_bounds(nx, ny) and world.read_cell("SOLID", nx, ny) == 0:
                self.x, self.y = nx, ny
        
        elif action == "TURN_L":
            self.d = self.d.turn_left()
        
        elif action == "TURN_R":
            self.d = self.d.turn_right()
        
        elif action.startswith("READ:"):
            layer = action[5:]
            if layer in world.LAYERS:
                val = world.read_cell(layer, self.x, self.y)
                self.regs[0] = val
        
        elif action.startswith("WRITE:"):
            parts = action[6:].split(":")
            if len(parts) == 2:
                layer, val_str = parts
                try:
                    val = int(val_str)
                    if layer in world.LAYERS:
                        world.write_cell(layer, self.x, self.y, val)
                except ValueError:
                    pass
        
        elif action == "PICK_FOOD":
            food_val = world.read_cell("FOOD", self.x, self.y)
            if food_val > 0 and self.carry_food == 0:
                world.write_cell("FOOD", self.x, self.y, food_val - 1)
                self.carry_food = 1
        
        elif action == "DROP_FOOD":
            if self.carry_food > 0:
                food_val = world.read_cell("FOOD", self.x, self.y)
                world.write_cell("FOOD", self.x, self.y, min(255, food_val + 1))
                self.carry_food = 0
        
        elif action.startswith("DEPOSIT:"):
            parts = action[8:].split(":")
            if len(parts) == 2:
                pher_type, amount_str = parts
                try:
                    amount = int(amount_str)
                    layer_map = {"FOOD": "PHER_FOOD", "HOME": "PHER_HOME", "ALERT": "PHER_ALERT"}
                    if pher_type in layer_map:
                        layer = layer_map[pher_type]
                        current = world.read_cell(layer, self.x, self.y)
                        world.write_cell(layer, self.x, self.y, min(255, current + amount))
                except ValueError:
                    pass
        
        elif action == "NOP":
            pass
        
        elif action == "YIELD":
            pass
