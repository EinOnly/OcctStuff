"""
简化的世界模拟
"""
import numpy as np
from typing import Tuple, List, Optional


class SimpleWorld:
    """简化的2D世界，用于LLM agent实验"""
    
    def __init__(self, size: int = 20):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        
        # 家的位置
        self.home_pos = (size // 2, size // 2)
        self.stored_food = 0
        
        # 食物位置
        self.food_locations = set()
        
        # 墙壁位置
        self.wall_locations = set()
        
        # 初始化
        self._setup_world()
    
    def _setup_world(self):
        """设置世界"""
        # 添加边界墙
        for i in range(self.size):
            self.wall_locations.add((0, i))
            self.wall_locations.add((self.size-1, i))
            self.wall_locations.add((i, 0))
            self.wall_locations.add((i, self.size-1))
        
        # 添加一些随机墙壁
        for _ in range(self.size):
            x, y = np.random.randint(1, self.size-1, 2)
            if (x, y) != self.home_pos:
                self.wall_locations.add((x, y))
        
        # 生成初始食物
        self.spawn_food()
    
    def spawn_food(self):
        """生成食物"""
        for _ in range(5):
            x, y = np.random.randint(1, self.size-1, 2)
            if (x, y) not in self.wall_locations and (x, y) != self.home_pos:
                self.food_locations.add((x, y))
    
    def can_move(self, x: int, y: int) -> bool:
        """检查是否可以移动到该位置"""
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        return (x, y) not in self.wall_locations
    
    def has_food(self, x: int, y: int) -> bool:
        """检查该位置是否有食物"""
        return (x, y) in self.food_locations
    
    def remove_food(self, x: int, y: int):
        """移除食物"""
        self.food_locations.discard((x, y))
    
    def is_home(self, x: int, y: int) -> bool:
        """检查是否在家"""
        return (x, y) == self.home_pos
    
    def add_stored_food(self):
        """增加存储的食物"""
        self.stored_food += 1
    
    def get_nearby_info(self, x: int, y: int, radius: int = 1) -> dict:
        """获取周围信息"""
        info = {}
        
        # 检查四个方向
        directions = [
            ("north", (x, y+1)),
            ("south", (x, y-1)),
            ("east", (x+1, y)),
            ("west", (x-1, y))
        ]
        
        for dir_name, (nx, ny) in directions:
            # 食物
            if (nx, ny) in self.food_locations:
                info[f"food_{dir_name}"] = True
            # 家
            if (nx, ny) == self.home_pos:
                info[f"home_{dir_name}"] = True
            # 墙壁
            if (nx, ny) in self.wall_locations:
                info[f"wall_{dir_name}"] = True
        
        return info
