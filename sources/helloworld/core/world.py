import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class WorldConf:
    H: int = 128
    W: int = 128
    pher_decay: float = 0.95
    diffuse_weight: float = 0.2


class World:
    
    LAYERS = ["SOLID", "FOOD", "HOME", "PHER_FOOD", "PHER_HOME", "PHER_ALERT", "TAPE", "MARK"]
    
    def __init__(self, conf: WorldConf):
        self.conf = conf
        self.H = conf.H
        self.W = conf.W
        
        self.layers = {}
        for layer_name in self.LAYERS:
            self.layers[layer_name] = np.zeros((self.H, self.W), dtype=np.uint8)
    
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H
    
    def read_cell(self, layer: str, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            return 0
        return int(self.layers[layer][y, x])
    
    def write_cell(self, layer: str, x: int, y: int, val: int) -> None:
        if not self.in_bounds(x, y):
            return
        self.layers[layer][y, x] = np.clip(val, 0, 255)
    
    def step_fields(self) -> None:
        pher_layers = ["PHER_FOOD", "PHER_HOME", "PHER_ALERT"]
        
        for layer_name in pher_layers:
            field = self.layers[layer_name].astype(np.float32)
            
            diffused = np.zeros_like(field)
            diffused[1:-1, 1:-1] = (
                field[:-2, 1:-1] + 
                field[2:, 1:-1] + 
                field[1:-1, :-2] + 
                field[1:-1, 2:]
            ) * 0.25
            
            w = self.conf.diffuse_weight
            field = (1 - w) * field + w * diffused
            
            field *= self.conf.pher_decay
            
            self.layers[layer_name] = np.clip(field, 0, 255).astype(np.uint8)
    
    def place_home(self, cx: int, cy: int, radius: int) -> None:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x, y = cx + dx, cy + dy
                    if self.in_bounds(x, y):
                        self.layers["HOME"][y, x] = 255
    
    def place_food_patch(self, cx: int, cy: int, radius: int, amount: int) -> None:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x, y = cx + dx, cy + dy
                    if self.in_bounds(x, y):
                        self.layers["FOOD"][y, x] = min(255, self.layers["FOOD"][y, x] + amount)
    
    def place_obstacle_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        x0, x1 = max(0, x0), min(self.W, x1)
        y0, y1 = max(0, y0), min(self.H, y1)
        self.layers["SOLID"][y0:y1, x0:x1] = 255
    
    def scatter_food(self, rng: np.random.Generator, num_patches: int, patch_radius: int, amount: int) -> None:
        for _ in range(num_patches):
            cx = rng.integers(patch_radius, self.W - patch_radius)
            cy = rng.integers(patch_radius, self.H - patch_radius)
            self.place_food_patch(cx, cy, patch_radius, amount)
