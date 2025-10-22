from enum import Enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple
import colorsys
import random

if TYPE_CHECKING:
    from core.expandable_world import ExpandableWorld


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
class EliteGenome:
    exploration_rate: float = 0.3
    food_attraction: float = 1.0
    home_attraction: float = 1.0
    pheromone_sensitivity: float = 0.5
    energy_efficiency: float = 1.0
    building_tendency: float = 0.3
    body_size: float = 1.0
    # 角色倾向 (三者之和应该接近1.0)
    explorer_tendency: float = 0.33  # 探索倾向
    gatherer_tendency: float = 0.34  # 采集倾向
    builder_tendency_role: float = 0.33  # 建造倾向(角色)
    
    @classmethod
    def from_elite(cls, elite_genome: 'EliteGenome', mutation_rate: float = 0.08):
        import random
        # 角色倾向变异
        explorer_t = max(0.0, min(1.0, elite_genome.explorer_tendency + random.gauss(0, mutation_rate * 0.4)))
        gatherer_t = max(0.0, min(1.0, elite_genome.gatherer_tendency + random.gauss(0, mutation_rate * 0.4)))
        builder_t = max(0.0, min(1.0, elite_genome.builder_tendency_role + random.gauss(0, mutation_rate * 0.4)))
        # 归一化
        total = explorer_t + gatherer_t + builder_t
        if total > 0:
            explorer_t /= total
            gatherer_t /= total
            builder_t /= total
        
        return cls(
            exploration_rate=max(0.05, min(0.95, elite_genome.exploration_rate + random.gauss(0, mutation_rate))),
            food_attraction=max(0.1, min(3.0, elite_genome.food_attraction + random.gauss(0, mutation_rate * 0.5))),
            home_attraction=max(0.1, min(3.0, elite_genome.home_attraction + random.gauss(0, mutation_rate * 0.5))),
            pheromone_sensitivity=max(0.1, min(2.0, elite_genome.pheromone_sensitivity + random.gauss(0, mutation_rate * 0.3))),
            energy_efficiency=max(0.5, min(2.0, elite_genome.energy_efficiency + random.gauss(0, mutation_rate * 0.2))),
            building_tendency=max(0.0, min(1.0, elite_genome.building_tendency + random.gauss(0, mutation_rate * 0.3))),
            body_size=max(0.5, min(2.0, elite_genome.body_size + random.gauss(0, mutation_rate * 0.15))),
            explorer_tendency=explorer_t,
            gatherer_tendency=gatherer_t,
            builder_tendency_role=builder_t
        )

    @classmethod
    def crossover(cls, parent_a: 'EliteGenome', parent_b: 'EliteGenome', mutation_rate: float = 0.06):
        # 角色倾向的混合和归一化
        explorer_t = (parent_a.explorer_tendency + parent_b.explorer_tendency) * 0.5
        gatherer_t = (parent_a.gatherer_tendency + parent_b.gatherer_tendency) * 0.5
        builder_t = (parent_a.builder_tendency_role + parent_b.builder_tendency_role) * 0.5
        total = explorer_t + gatherer_t + builder_t
        if total > 0:
            explorer_t /= total
            gatherer_t /= total
            builder_t /= total
        
        base = cls(
            exploration_rate=(parent_a.exploration_rate + parent_b.exploration_rate) * 0.5,
            food_attraction=(parent_a.food_attraction + parent_b.food_attraction) * 0.5,
            home_attraction=(parent_a.home_attraction + parent_b.home_attraction) * 0.5,
            pheromone_sensitivity=(parent_a.pheromone_sensitivity + parent_b.pheromone_sensitivity) * 0.5,
            energy_efficiency=(parent_a.energy_efficiency + parent_b.energy_efficiency) * 0.5,
            building_tendency=(parent_a.building_tendency + parent_b.building_tendency) * 0.5,
            body_size=(parent_a.body_size + parent_b.body_size) * 0.5,
            explorer_tendency=explorer_t,
            gatherer_tendency=gatherer_t,
            builder_tendency_role=builder_t,
        )
        return cls.from_elite(base, mutation_rate=mutation_rate)
    
    def get_color(self) -> Tuple[float, float, float]:
        hue = (self.exploration_rate * 0.3 + 
               self.food_attraction * 0.15 + 
               self.pheromone_sensitivity * 0.2) % 1.0
        
        saturation = 0.5 + (self.energy_efficiency - 1.0) * 0.3
        saturation = max(0.3, min(1.0, saturation))
        
        value = 0.6 + (self.body_size - 1.0) * 0.2
        value = max(0.5, min(1.0, value))
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return rgb
    
    def fitness_contribution(self) -> float:
        return (
            self.energy_efficiency * 50 +
            self.food_attraction * 20 +
            self.home_attraction * 20 +
            (1.0 - self.exploration_rate) * 10
        )


@dataclass
class AdvancedAgent:
    id: int
    x: int
    y: int
    d: Dir
    genome: EliteGenome = field(default_factory=EliteGenome)
    regs: list = field(default_factory=lambda: [0] * 8)
    stack: list = field(default_factory=list)
    carry_food: int = 0
    has_food: bool = False
    energy: float = 100.0
    age: int = 0
    food_collected: int = 0
    food_delivered: int = 0
    distance_traveled: float = 0.0
    generation: int = 0
    alive: bool = True
    built_structures: int = 0
    parent_id: Optional[int] = None
    # 角色系统
    role: str = "Gatherer"  # Explorer, Gatherer, Builder
    # 路径记忆
    path_to_food: list = field(default_factory=list)  # 记录去食物的路径
    explored_cells: set = field(default_factory=set)  # 记录探索过的单元格
    
    def fitness(self) -> float:
        base_fitness = (
            self.food_delivered * 100 +
            self.food_collected * 50 +
            self.built_structures * 80 +
            self.age * 0.2 -
            self.distance_traveled * 0.005
        )
        return base_fitness + self.genome.fitness_contribution()
    
    def get_visual_size(self) -> float:
        return 2.0 + (self.genome.body_size - 1.0) * 1.5
    
    def get_color(self) -> Tuple[float, float, float]:
        if self.carry_food > 0:
            return (1.0, 0.9, 0.2)
        # 根据角色显示不同颜色
        if self.role == "Explorer":
            return (0.3, 0.6, 1.0)  # 蓝色 - 探索者
        elif self.role == "Builder":
            return (1.0, 0.5, 0.2)  # 橙色 - 建造者
        else:  # Gatherer
            return (0.4, 0.9, 0.3)  # 绿色 - 采集者
    
    def obs_string(self, world: 'ExpandableWorld') -> str:
        parts = []
        
        parts.append(f"ID={self.id}")
        parts.append(f"POS={self.x},{self.y}")
        parts.append(f"DIR={self.d.name}")
        parts.append(f"CARRY={self.carry_food}")
        parts.append(f"ENERGY={int(self.energy)}")
        parts.append(f"GEN={self.generation}")
        
        nbr = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = self.x + dx, self.y + dy
                f = world.read_cell("FOOD", nx, ny)
                h = world.read_cell("HOME", nx, ny)
                n = world.read_cell("NEST", nx, ny)
                nbr.append(f"{f:02x}{h:02x}{n:02x}")
        
        parts.append(f"NBR={''.join(nbr)}")
        
        return ";".join(parts)
    
    def act(self, action_code: str, world: 'ExpandableWorld') -> None:
        if action_code == "EAT":
            food_val = world.read_cell("FOOD", self.x, self.y)
            if food_val > 0:
                world.write_cell("FOOD", self.x, self.y, max(0, food_val - 10))
                self.energy = min(130, self.energy + 25)  # 增加到25（原来15）
                return
        
        action_map = {
            "F": "MOVE_FWD",
            "TL": "TURN_L",
            "TR": "TURN_R",
            "PICK": "PICK_FOOD",
            "DROP": "DROP_FOOD",
            "BUILD_NEST": "BUILD_NEST",
            "BUILD_STORAGE": "BUILD_STORAGE",
            "BUILD_TRAIL": "BUILD_TRAIL",
            "NOOP": "NOP",
        }
        
        full_action = action_map.get(action_code, "NOP")
        self.apply_action(world, full_action)
    
    def apply_action(self, world: 'ExpandableWorld', action: str) -> None:
        action = action.strip()
        
        old_x, old_y = self.x, self.y
        
        # 记录探索过的位置
        self.explored_cells.add((self.x, self.y))
        
        if action == "MOVE_FWD":
            dx, dy = self.d.forward_delta()
            nx, ny = self.x + dx, self.y + dy
            
            if world.in_bounds(nx, ny) and world.read_cell("SOLID", nx, ny) == 0:
                # 如果正在寻找食物且没有携带，记录路径
                if not self.has_food and len(self.path_to_food) < 100:  # 限制路径长度
                    self.path_to_food.append((self.x, self.y, self.d))
                
                self.x, self.y = nx, ny
                # 降低移动消耗
                move_cost = (0.15 + self.genome.body_size * 0.1) / self.genome.energy_efficiency
                self.energy -= move_cost
        
        elif action == "TURN_L":
            self.d = self.d.turn_left()
            self.energy -= 0.02 / self.genome.energy_efficiency
        
        elif action == "TURN_R":
            self.d = self.d.turn_right()
            self.energy -= 0.02 / self.genome.energy_efficiency
        
        elif action == "PICK_FOOD":
            food_val = world.read_cell("FOOD", self.x, self.y)
            if food_val > 0 and self.carry_food == 0:
                world.write_cell("FOOD", self.x, self.y, food_val - 1)
                self.carry_food = 1
                self.has_food = True
                self.food_collected += 1
                self.energy -= 0.2
                current = world.read_cell("PHER_FOOD", self.x, self.y)
                world.write_cell("PHER_FOOD", self.x, self.y, min(255, current + 25))
                # 找到食物，记录这个位置为食物源
                # （路径已经在移动时记录了）
        
        elif action == "DROP_FOOD":
            if self.carry_food > 0:
                home_val = world.read_cell("HOME", self.x, self.y)
                storage_val = world.read_cell("STORAGE", self.x, self.y)
                
                if home_val > 0 or storage_val > 0:
                    self.food_delivered += 1
                    self.energy = min(130.0, self.energy + 22.0)
                    current = world.read_cell("PHER_HOME", self.x, self.y)
                    world.write_cell("PHER_HOME", self.x, self.y, min(255, current + 30))
                    # 成功送达，准备下一次采集（清空路径以便重新记录）
                    self.path_to_food.clear()
                else:
                    food_val = world.read_cell("FOOD", self.x, self.y)
                    world.write_cell("FOOD", self.x, self.y, min(255, food_val + 1))
                self.carry_food = 0
                self.has_food = False
        
        elif action == "BUILD_NEST":
            if self.energy > 20 and world.read_cell("NEST", self.x, self.y) == 0:
                world.write_cell("NEST", self.x, self.y, 200)
                world.write_cell("SOLID", self.x, self.y, 1)  # SOLID只用0/1
                self.built_structures += 1
                self.energy -= 10  # 降低到10（原来15）
        
        elif action == "BUILD_STORAGE":
            if self.energy > 15 and world.read_cell("STORAGE", self.x, self.y) == 0:
                world.write_cell("STORAGE", self.x, self.y, 180)
                world.write_cell("SOLID", self.x, self.y, 1)  # SOLID只用0/1
                self.built_structures += 1
                self.energy -= 8  # 降低到8（原来12）
        
        elif action == "BUILD_TRAIL":
            if self.energy > 3:
                trail_val = world.read_cell("TRAIL", self.x, self.y)
                world.write_cell("TRAIL", self.x, self.y, min(255, trail_val + 50))
                self.energy -= 1  # 降低到1（原来2）
        
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
        
        if (self.x, self.y) != (old_x, old_y):
            import math
            self.distance_traveled += math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2)
        
        self.age += 1
        # 降低基础代谢消耗
        base_metabolism = 0.02 + self.genome.body_size * 0.01
        self.energy -= base_metabolism / self.genome.energy_efficiency
        self.has_food = self.carry_food > 0

        # 能量耗尽则死亡
        if self.energy <= 0:
            self.alive = False

    @classmethod
    def spawn(
        cls,
        id: int,
        x: int,
        y: int,
        genome: EliteGenome,
        generation: int = 0,
        parent_id: Optional[int] = None,
        heading: Optional[Dir] = None,
        base_energy: Optional[float] = None,
    ) -> 'AdvancedAgent':
        if heading is None:
            heading = random.choice(list(Dir))
        
        # 根据基因倾向选择角色
        role_weights = {
            "Explorer": genome.explorer_tendency,
            "Gatherer": genome.gatherer_tendency,
            "Builder": genome.builder_tendency_role,
        }
        # 选择最高倾向的角色
        role = max(role_weights, key=role_weights.get)
        
        agent = cls(
            id=id,
            x=x,
            y=y,
            d=heading,
            genome=genome,
            generation=generation,
            parent_id=parent_id,
            role=role,
        )
        if base_energy is None:
            base_energy = 80.0 + genome.energy_efficiency * 12.0
        agent.energy = min(130.0, base_energy)
        agent.carry_food = 0
        agent.has_food = False
        agent.stack = []
        agent.regs = [0] * 8
        agent.age = 0
        agent.food_collected = 0
        agent.food_delivered = 0
        agent.distance_traveled = 0.0
        agent.alive = True
        agent.built_structures = 0
        agent.path_to_food = []
        agent.explored_cells = set()
        return agent
