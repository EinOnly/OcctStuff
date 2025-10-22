import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class WorldConf:
    H: int = 64
    W: int = 64
    max_H: int = 512
    max_W: int = 512
    pher_decay: float = 0.95
    diffuse_weight: float = 0.2
    expand_threshold: int = 5


class ExpandableWorld:
    
    LAYERS = ["SOLID", "FOOD", "HOME", "PHER_FOOD", "PHER_HOME", "PHER_ALERT", 
              "TAPE", "MARK", "NEST", "STORAGE", "TRAIL"]
    
    def __init__(self, conf: WorldConf):
        self.conf = conf
        self.H = conf.H
        self.W = conf.W
        self.max_H = conf.max_H
        self.max_W = conf.max_W
        
        self.layers = {}
        for layer_name in self.LAYERS:
            self.layers[layer_name] = np.zeros((self.H, self.W), dtype=np.uint8)
        
        self.expansion_count = 0
        self.edge_visit_count = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        self.home_positions: List[Tuple[int, int]] = []
        self._rng = np.random.default_rng()
    
    def check_and_expand(self, x: int, y: int) -> Tuple[int, int]:
        threshold = self.conf.expand_threshold
        expanded = False
        new_x, new_y = x, y
        
        if y < 3 and self.H < self.max_H:
            self.edge_visit_count["top"] += 1
            if self.edge_visit_count["top"] >= threshold:
                expanded = self._expand_top()
                new_y = y + 32
                self.edge_visit_count["top"] = 0
        
        elif y > self.H - 4 and self.H < self.max_H:
            self.edge_visit_count["bottom"] += 1
            if self.edge_visit_count["bottom"] >= threshold:
                expanded = self._expand_bottom()
                self.edge_visit_count["bottom"] = 0
        
        if x < 3 and self.W < self.max_W:
            self.edge_visit_count["left"] += 1
            if self.edge_visit_count["left"] >= threshold:
                expanded = self._expand_left()
                new_x = x + 32
                self.edge_visit_count["left"] = 0
        
        elif x > self.W - 4 and self.W < self.max_W:
            self.edge_visit_count["right"] += 1
            if self.edge_visit_count["right"] >= threshold:
                expanded = self._expand_right()
                self.edge_visit_count["right"] = 0
        
        if expanded:
            self.expansion_count += 1
            print(f"🌍 World expanded! New size: {self.W}×{self.H} (expansion #{self.expansion_count})")
        
        return new_x, new_y
    
    def _expand_top(self) -> bool:
        if self.H >= self.max_H:
            return False
        
        old_H = self.H
        new_H = min(self.H + 32, self.max_H)
        expansion = new_H - old_H
        
        for layer_name in self.LAYERS:
            new_layer = np.zeros((new_H, self.W), dtype=np.uint8)
            new_layer[expansion:, :] = self.layers[layer_name]
            self.layers[layer_name] = new_layer
        
        if self.home_positions:
            self.home_positions = [(hx, hy + expansion) for hx, hy in self.home_positions]
        
        self.H = new_H
        self._seed_food_strip(0, self.W, 0, expansion, intensity=1.2)
        return True
    
    def _expand_bottom(self) -> bool:
        if self.H >= self.max_H:
            return False
        
        old_H = self.H
        new_H = min(self.H + 32, self.max_H)
        expansion = new_H - old_H
        
        for layer_name in self.LAYERS:
            new_layer = np.zeros((new_H, self.W), dtype=np.uint8)
            new_layer[:old_H, :] = self.layers[layer_name]
            self.layers[layer_name] = new_layer
        
        self.H = new_H
        self._seed_food_strip(0, self.W, self.H - expansion, self.H, intensity=1.2)
        return True
    
    def _expand_left(self) -> bool:
        if self.W >= self.max_W:
            return False
        
        old_W = self.W
        new_W = min(self.W + 32, self.max_W)
        expansion = new_W - old_W
        
        for layer_name in self.LAYERS:
            new_layer = np.zeros((self.H, new_W), dtype=np.uint8)
            new_layer[:, expansion:] = self.layers[layer_name]
            self.layers[layer_name] = new_layer
        
        if self.home_positions:
            self.home_positions = [(hx + expansion, hy) for hx, hy in self.home_positions]
        
        self.W = new_W
        self._seed_food_strip(0, expansion, 0, self.H, intensity=1.2)
        return True
    
    def _expand_right(self) -> bool:
        if self.W >= self.max_W:
            return False
        
        old_W = self.W
        new_W = min(self.W + 32, self.max_W)
        expansion = new_W - old_W
        
        for layer_name in self.LAYERS:
            new_layer = np.zeros((self.H, new_W), dtype=np.uint8)
            new_layer[:, :old_W] = self.layers[layer_name]
            self.layers[layer_name] = new_layer
        
        self.W = new_W
        self._seed_food_strip(self.W - expansion, self.W, 0, self.H, intensity=1.2)
        return True
    
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
    
    def step(self) -> None:
        self.step_fields()
    
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
        self.home_positions.append((cx, cy))
    
    def place_food_patch(self, cx: int, cy: int, radius: int, amount: int) -> None:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x, y = cx + dx, cy + dy
                    if self.in_bounds(x, y):
                        current = int(self.layers["FOOD"][y, x])
                        self.layers["FOOD"][y, x] = min(255, current + amount)
    
    def place_obstacle_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        x0, x1 = max(0, x0), min(self.W, x1)
        y0, y1 = max(0, y0), min(self.H, y1)
        self.layers["SOLID"][y0:y1, x0:x1] = 1  # SOLID只用0/1
    
    def scatter_food(self, rng: np.random.Generator, num_patches: int, patch_radius: int, amount: int) -> None:
        for _ in range(num_patches):
            cx = rng.integers(patch_radius, self.W - patch_radius)
            cy = rng.integers(patch_radius, self.H - patch_radius)
            self.place_food_patch(cx, cy, patch_radius, amount)

    def nearest_home_vector(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        if not self.home_positions:
            return None
        best_dx, best_dy = None, None
        best_dist = None
        for hx, hy in self.home_positions:
            dx = hx - x
            dy = hy - y
            dist = dx * dx + dy * dy
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_dx, best_dy = dx, dy
        if best_dx is None or best_dy is None:
            return None
        return best_dx, best_dy

    def _seed_food_strip(self, x0: int, x1: int, y0: int, y1: int, intensity: float = 1.0) -> None:
        x0 = max(0, min(self.W, x0))
        x1 = max(0, min(self.W, x1))
        y0 = max(0, min(self.H, y0))
        y1 = max(0, min(self.H, y1))
        if x1 <= x0 or y1 <= y0:
            return
        width = x1 - x0
        height = y1 - y0
        area = width * height
        patch_count = max(1, int(area / 2048))
        radius = max(2, int(3 * intensity))
        base_amount = int(90 * intensity)
        for _ in range(patch_count):
            cx = int(self._rng.integers(x0, x1))
            cy = int(self._rng.integers(y0, y1))
            amount = int(base_amount * self._rng.uniform(0.6, 1.4))
            self.place_food_patch(cx, cy, radius, amount)
