"""
Mock LLM - 极简模拟模型（无需下载，用于演示）
"""
import random
import re


class MockTinyLLM:
    """
    超轻量级Mock LLM - 基于规则生成action
    
    大小：<1KB
    用途：演示系统架构，无需下载真实模型
    """
    
    def __init__(self):
        self.backend = "mock"
        self.model_path = "mock_tiny_llm"
        print("✓ Mock Tiny LLM loaded (rules-based, <1KB)")
    
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
        """
        根据prompt生成action（基于简单规则 + 记忆/探索策略）
        
        🧠 新增：支持探索和记忆信息
        """
        prompt_lower = prompt.lower()
        
        # 🧠 解析记忆和探索状态
        has_memory = "remember" in prompt_lower and "food" in prompt_lower
        unexplored = "unexplored" in prompt_lower
        familiar = "familiar" in prompt_lower
        has_food = "has food" in prompt_lower or "carrying" in prompt_lower
        
        # ========== 优先级1: 携带食物回家 ==========
        if has_food:
            # 看到home就往home走
            if "home↑" in prompt or "home north" in prompt_lower:
                return "move north to home"
            elif "home↓" in prompt or "home south" in prompt_lower:
                return "move south to home"
            elif "home→" in prompt or "home east" in prompt_lower:
                return "move east to home"
            elif "home←" in prompt or "home west" in prompt_lower:
                return "move west to home"
            
            # 在home位置
            if "at home" in prompt_lower or "home" in prompt_lower:
                return "drop food at home"
            
            # 🧠 使用记忆导航回家（假设home在中心）
            match = re.search(r"At \((\d+),(\d+)\)", prompt)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                home_x, home_y = 10, 10
                
                walls = []
                if "wall↑" in prompt:
                    walls.append("north")
                if "wall↓" in prompt:
                    walls.append("south")
                if "wall→" in prompt:
                    walls.append("east")
                if "wall←" in prompt:
                    walls.append("west")
                
                # 优先X方向
                if abs(x - home_x) > abs(y - home_y):
                    if x > home_x and "west" not in walls:
                        return "move west toward home"
                    elif x < home_x and "east" not in walls:
                        return "move east toward home"
                # 然后Y方向
                if y > home_y and "south" not in walls:
                    return "move south toward home"
                elif y < home_y and "north" not in walls:
                    return "move north toward home"
        
        # ========== 优先级2: 看到食物就去拿 ==========
        if "food↑" in prompt or "food north" in prompt_lower:
            return "move north to get food"
        elif "food↓" in prompt or "food south" in prompt_lower:
            return "move south to get food"
        elif "food→" in prompt or "food east" in prompt_lower:
            return "move east to get food"
        elif "food←" in prompt or "food west" in prompt_lower:
            return "move west to get food"
        
        # ========== 优先级3: 🧠 有食物记忆，探索去找 ==========
        if has_memory and not has_food:
            # 获取墙壁信息
            walls = []
            if "wall↑" in prompt:
                walls.append("north")
            if "wall↓" in prompt:
                walls.append("south")
            if "wall→" in prompt:
                walls.append("east")
            if "wall←" in prompt:
                walls.append("west")
            
            # 优先往记忆中的食物方向探索（这里简化为随机非墙方向）
            directions = ["north", "south", "east", "west"]
            available = [d for d in directions if d not in walls]
            
            if available:
                direction = random.choice(available)
                return f"explore {direction} to find remembered food"
        
        # ========== 优先级4: 🧠 未探索区域，边界探索 ==========
        if unexplored:
            match = re.search(r"At \((\d+),(\d+)\)", prompt)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                
                walls = []
                if "wall↑" in prompt:
                    walls.append("north")
                if "wall↓" in prompt:
                    walls.append("south")
                if "wall→" in prompt:
                    walls.append("east")
                if "wall←" in prompt:
                    walls.append("west")
                
                # 边界探索策略：远离中心
                if x < 5 and "west" not in walls:
                    return "explore west (boundary)"
                elif x > 15 and "east" not in walls:
                    return "explore east (boundary)"
                
                if y < 5 and "south" not in walls:
                    return "explore south (boundary)"
                elif y > 15 and "north" not in walls:
                    return "explore north (boundary)"
        
        # ========== 优先级5: 🧠 熟悉区域，尝试离开 ==========
        if familiar:
            match = re.search(r"At \((\d+),(\d+)\)", prompt)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                
                walls = []
                if "wall↑" in prompt:
                    walls.append("north")
                if "wall↓" in prompt:
                    walls.append("south")
                if "wall→" in prompt:
                    walls.append("east")
                if "wall←" in prompt:
                    walls.append("west")
                
                # 离开熟悉区域：往中心走
                if x < 8 and "east" not in walls:
                    return "move east (leave familiar area)"
                elif x > 12 and "west" not in walls:
                    return "move west (leave familiar area)"
                
                if y < 8 and "north" not in walls:
                    return "move north (leave familiar area)"
                elif y > 12 and "south" not in walls:
                    return "move south (leave familiar area)"
        
        # ========== 默认: 随机探索（避开墙壁） ==========
        walls = []
        if "wall↑" in prompt:
            walls.append("north")
        if "wall↓" in prompt:
            walls.append("south")
        if "wall→" in prompt:
            walls.append("east")
        if "wall←" in prompt:
            walls.append("west")
        
        directions = ["north", "south", "east", "west"]
        available = [d for d in directions if d not in walls]
        
        if available:
            direction = random.choice(available)
            return f"explore {direction}"
        else:
            return "wait"


class MockModelLoader:
    """Mock版本的模型加载器"""
    
    def __init__(self, model_path=None, model_type=None):
        self.model = MockTinyLLM()
        self.backend = "mock"
    
    def load(self):
        """加载（实际上啥都不做，已经在__init__中创建了）"""
        return self
    
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
        """生成"""
        return self.model.generate(prompt, max_tokens, temperature)


class MockSharedModelPool:
    """Mock版本的共享模型池"""
    
    _instance = None
    _base_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_base_model(self):
        """获取共享的base model"""
        if self._base_model is None:
            print("Initializing Mock Shared Model Pool...")
            self._base_model = MockModelLoader().load()
        return self._base_model
    
    def generate_batch(self, prompts):
        """批量生成"""
        model = self.get_base_model()
        return [model.generate(p) for p in prompts]


# 测试
if __name__ == "__main__":
    print("Testing Mock Tiny LLM...")
    
    model = MockTinyLLM()
    
    test_cases = [
        "At (5,3), E80. See: food↑",
        "At (6,3), E79, has food. See: home↓",
        "At (2,2), E85. See: wall← food→",
        "At (1,1), E90, has food, at home",
    ]
    
    for prompt in test_cases:
        response = model.generate(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
