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
class AgentGenome:
    exploration_rate: float = 0.3
    food_attraction: float = 1.0
    home_attraction: float = 1.0
    pheromone_sensitivity: float = 0.5
    energy_efficiency: float = 1.0
    
    def mutate(self, mutation_rate: float = 0.1):
        import random
        genome = AgentGenome(
            exploration_rate=max(0.05, min(0.95, self.exploration_rate + random.gauss(0, mutation_rate))),
            food_attraction=max(0.1, min(3.0, self.food_attraction + random.gauss(0, mutation_rate * 0.5))),
            home_attraction=max(0.1, min(3.0, self.home_attraction + random.gauss(0, mutation_rate * 0.5))),
            pheromone_sensitivity=max(0.1, min(2.0, self.pheromone_sensitivity + random.gauss(0, mutation_rate * 0.3))),
            energy_efficiency=max(0.5, min(2.0, self.energy_efficiency + random.gauss(0, mutation_rate * 0.2)))
        )
        return genome
    
    def crossover(self, other: 'AgentGenome'):
        import random
        return AgentGenome(
            exploration_rate=random.choice([self.exploration_rate, other.exploration_rate]),
            food_attraction=random.choice([self.food_attraction, other.food_attraction]),
            home_attraction=random.choice([self.home_attraction, other.home_attraction]),
            pheromone_sensitivity=random.choice([self.pheromone_sensitivity, other.pheromone_sensitivity]),
            energy_efficiency=random.choice([self.energy_efficiency, other.energy_efficiency])
        )


@dataclass
class Agent:
    id: int
    x: int
    y: int
    d: Dir
    genome: AgentGenome = field(default_factory=AgentGenome)
    regs: list = field(default_factory=lambda: [0] * 8)
    stack: list = field(default_factory=list)
    carry_food: int = 0
    energy: float = 100.0
    age: int = 0
    food_collected: int = 0
    food_delivered: int = 0
    distance_traveled: float = 0.0
    generation: int = 0
    alive: bool = True
    
    def fitness(self) -> float:
        return (
            self.food_delivered * 100 +
            self.food_collected * 50 +
            self.age * 0.1 -
            self.distance_traveled * 0.01
        )
    
    def obs_string(self, world: 'World') -> str:
        parts = []
        
        parts.append(f"ID={self.id}")
        parts.append(f"POS={self.x},{self.y}")
        parts.append(f"DIR={self.d.name}")
        parts.append(f"CARRY={self.carry_food}")
        parts.append(f"ENERGY={int(self.energy)}")
        
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
        
        old_x, old_y = self.x, self.y
        
        if action == "MOVE_FWD":
            dx, dy = self.d.forward_delta()
            nx, ny = self.x + dx, self.y + dy
            if world.in_bounds(nx, ny) and world.read_cell("SOLID", nx, ny) == 0:
                self.x, self.y = nx, ny
                self.energy -= 0.1 / self.genome.energy_efficiency
        
        elif action == "TURN_L":
            self.d = self.d.turn_left()
            self.energy -= 0.05 / self.genome.energy_efficiency
        
        elif action == "TURN_R":
            self.d = self.d.turn_right()
            self.energy -= 0.05 / self.genome.energy_efficiency
        
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
                self.food_collected += 1
                self.energy -= 0.2
        
        elif action == "DROP_FOOD":
            if self.carry_food > 0:
                home_val = world.read_cell("HOME", self.x, self.y)
                if home_val > 0:
                    self.food_delivered += 1
                    self.energy += 20.0
                else:
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
                        self.energy -= 0.1
                except ValueError:
                    pass
        
        elif action == "NOP":
            self.energy -= 0.01
        
        elif action == "YIELD":
            pass
        
        if (self.x, self.y) != (old_x, old_y):
            import math
            self.distance_traveled += math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2)
        
        self.age += 1
        self.energy -= 0.05
        
        if self.energy <= 0:
            self.alive = False
