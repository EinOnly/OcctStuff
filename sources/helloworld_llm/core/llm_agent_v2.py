"""
LLM Agent V2 - 使用本地LLM + LoRA在线学习
"""
from typing import Dict, Any, Optional, List, Tuple
import time
import numpy as np

try:
    from core.model_loader import SharedModelPool
except ImportError:
    from core.mock_model import MockSharedModelPool as SharedModelPool
    print("Using Mock Model (no real LLM needed)")

from core.lora_trainer import LoRAAdapter
from core.world import SimpleWorld
from config import Config


class LLMAgentV2:
    """基于本地微型LLM的智能体（带LoRA在线学习）"""
    
    # 类变量：共享的模型池（20个agent共用一个base model）
    _model_pool = SharedModelPool()
    
    def __init__(self, agent_id: int, position: Tuple[int, int], energy: int = 200, lora: Optional[LoRAAdapter] = None):
        self.agent_id = agent_id
        self.position = position  # (x, y)
        self.energy = energy
        self.max_energy = 200
        self.carrying_food = False
        self.alive = True
        
        # LoRA适配器（每个agent的个性化"知识"）
        self.lora = lora or LoRAAdapter(agent_id)
        
        # 历史记录
        self.action_history: List[str] = []
        self.observation_history: List[str] = []
        self.reward_history: List[float] = []
        
        # 🧠 探索和记忆系统
        self.explored_cells: set = set()  # 已探索的位置
        self.food_memory: List[Tuple[int, int]] = []  # 记住的食物位置（最多10个）
        self.path_history: List[Tuple[int, int]] = []  # 最近的路径（最多50步）
        self.successful_paths: List[List[Tuple[int, int]]] = []  # 成功的路径记录
        self.danger_cells: set = set()  # 危险区域（撞墙/困住的地方）
        
        # 统计信息
        self.total_reward = 0.0
        self.food_collected = 0
        self.food_delivered = 0
        self.steps_taken = 0
        self.decision_times = []
        self.exploration_rate = 0.0  # 探索率
        
    def observe(self, world: SimpleWorld) -> Dict[str, Any]:
        """观察周围环境"""
        x, y = self.position
        nearby = world.get_nearby_info(x, y, radius=1)
        
        # 🧠 更新探索记录
        self.explored_cells.add((x, y))
        
        # 🧠 更新食物记忆
        for dir_name, (nx, ny) in [("north", (x, y+1)), ("south", (x, y-1)), 
                                     ("east", (x+1, y)), ("west", (x-1, y))]:
            if nearby.get(f"food_{dir_name}"):
                if (nx, ny) not in self.food_memory:
                    self.food_memory.append((nx, ny))
                    if len(self.food_memory) > 10:  # 最多记住10个
                        self.food_memory.pop(0)
        
        observation = {
            "position": self.position,
            "energy": self.energy,
            "carrying_food": self.carrying_food,
            "nearby": nearby,
            "alive": self.alive,
            "explored_nearby": self._count_explored_nearby(x, y),
            "remembered_food": len(self.food_memory),
        }
        
        return observation
    
    def _count_explored_nearby(self, x: int, y: int, radius: int = 3) -> int:
        """统计周围已探索的格子数"""
        count = 0
        total = 0
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                total += 1
                if (x+dx, y+dy) in self.explored_cells:
                    count += 1
        return count / total if total > 0 else 0.0
    
    def _format_observation(self, obs: Dict[str, Any]) -> str:
        """将观察格式化为简洁的文本prompt（包含记忆和探索信息）"""
        x, y = obs["position"]
        energy = obs["energy"]
        carrying = obs["carrying_food"]
        nearby = obs["nearby"]
        
        # 构建简洁的状态描述（控制token数）
        prompt = f"At ({x},{y}), E{energy}"
        
        if carrying:
            prompt += ", has food"
        
        # 🧠 添加探索信息
        explored = obs.get("explored_nearby", 0)
        if explored < 0.3:
            prompt += ", unexplored area"
        elif explored > 0.8:
            prompt += ", familiar area"
        
        # 🧠 添加记忆信息
        remembered = obs.get("remembered_food", 0)
        if remembered > 0 and not carrying:
            prompt += f", remember {remembered} food"
        
        # 描述周围环境（只说重要的）
        directions = []
        if nearby.get("food_north"):
            directions.append("food↑")
        if nearby.get("food_south"):
            directions.append("food↓")
        if nearby.get("food_east"):
            directions.append("food→")
        if nearby.get("food_west"):
            directions.append("food←")
        
        if nearby.get("home_north"):
            directions.append("home↑")
        if nearby.get("home_south"):
            directions.append("home↓")
        if nearby.get("home_east"):
            directions.append("home→")
        if nearby.get("home_west"):
            directions.append("home←")
        
        if nearby.get("wall_north"):
            directions.append("wall↑")
        if nearby.get("wall_south"):
            directions.append("wall↓")
        if nearby.get("wall_east"):
            directions.append("wall→")
        if nearby.get("wall_west"):
            directions.append("wall←")
        
        if directions:
            prompt += f". See: {' '.join(directions)}"
        
        return prompt
    
    def decide_action(self, observation: Dict[str, Any]) -> str:
        """使用LLM决定行动"""
        # 格式化prompt
        obs_text = self._format_observation(observation)
        prompt = f"{obs_text}. Action:"
        
        # 添加简短的历史上下文
        if len(self.action_history) > 0:
            recent = self.action_history[-2:]
            prompt = f"[{','.join(recent)}] {prompt}"
        
        # 使用共享模型生成
        start_time = time.time()
        base_model = self._model_pool.get_base_model()
        
        # TODO: 应该应用个性化LoRA权重，暂时直接用base model
        raw_action = base_model.generate(prompt, max_tokens=10, temperature=0.7)
        
        decision_time = time.time() - start_time
        self.decision_times.append(decision_time)
        
        # 标准化action
        action = self._normalize_action(raw_action)
        
        return action
    
    def _normalize_action(self, raw_action: str) -> str:
        """从LLM输出提取标准action"""
        action = raw_action.lower().strip()
        
        # 提取关键词
        if "north" in action or "up" in action or "↑" in action:
            return "move_north"
        elif "south" in action or "down" in action or "↓" in action:
            return "move_south"
        elif "east" in action or "right" in action or "→" in action:
            return "move_east"
        elif "west" in action or "left" in action or "←" in action:
            return "move_west"
        elif "pick" in action or "collect" in action or "take" in action:
            return "pick_food"
        elif "drop" in action or "store" in action or "deliver" in action:
            return "drop_food"
        else:
            # 默认随机移动
            return np.random.choice(["move_north", "move_south", "move_east", "move_west"])
    
    def execute_action(self, action: str, world: SimpleWorld) -> float:
        """执行动作并返回奖励"""
        reward = 0.0
        x, y = self.position
        old_pos = (x, y)
        
        if action == "move_north":
            new_pos = (x, y + 1)
        elif action == "move_south":
            new_pos = (x, y - 1)
        elif action == "move_east":
            new_pos = (x + 1, y)
        elif action == "move_west":
            new_pos = (x - 1, y)
        elif action == "pick_food":
            if world.has_food(x, y) and not self.carrying_food:
                world.remove_food(x, y)
                self.carrying_food = True
                self.food_collected += 1
                reward = 10.0
                # 🧠 记录成功路径
                if len(self.path_history) > 0:
                    self.successful_paths.append(self.path_history.copy())
                    if len(self.successful_paths) > 5:
                        self.successful_paths.pop(0)
                # 从记忆中移除这个食物
                if (x, y) in self.food_memory:
                    self.food_memory.remove((x, y))
            else:
                reward = -1.0
            return reward
        elif action == "drop_food":
            if self.carrying_food and world.is_home(x, y):
                world.add_stored_food()
                self.carrying_food = False
                self.food_delivered += 1
                reward = 20.0
                # 🧠 记录超级成功路径
                if len(self.path_history) > 0:
                    self.successful_paths.append(self.path_history.copy())
            else:
                reward = -1.0
            return reward
        else:
            new_pos = (x + np.random.randint(-1, 2), y + np.random.randint(-1, 2))
            reward = -2.0
        
        # 移动
        if world.can_move(*new_pos):
            self.position = new_pos
            
            # 🧠 更新路径记忆
            self.path_history.append(new_pos)
            if len(self.path_history) > 50:
                self.path_history.pop(0)
            
            # 🧠 探索奖励
            if new_pos not in self.explored_cells:
                reward += 0.5  # 探索新区域奖励
            
            reward += -0.2  # 移动成本
            
            if world.has_food(*new_pos):
                reward += 3.0
            elif world.is_home(*new_pos) and self.carrying_food:
                reward += 5.0
        else:
            # 🧠 记录危险区域
            self.danger_cells.add(new_pos)
            reward = -3.0
        
        return reward
    
    def step(self, world: SimpleWorld) -> Dict[str, Any]:
        """执行一步完整循环"""
        if not self.alive:
            return {"action": "dead", "reward": 0.0, "energy": 0, "alive": False}
        
        # 1. 观察
        observation = self.observe(world)
        obs_text = self._format_observation(observation)
        
        # 2. 决策
        action = self.decide_action(observation)
        
        # 3. 执行
        reward = self.execute_action(action, world)
        
        # 4. 下一个观察
        next_observation = self.observe(world)
        next_obs_text = self._format_observation(next_observation)
        
        # 5. 记录经验（用于在线学习）
        self.lora.add_experience(obs_text, action, reward, next_obs_text)
        
        # 6. 在线学习（每10步训练一次）
        if Config.ENABLE_ONLINE_LEARNING and self.steps_taken % 10 == 0:
            if self.lora.should_train():
                self.lora.train_step()
        
        # 7. 更新能量
        self.energy -= 1
        if self.energy <= 0:
            self.alive = False
            self.energy = 0
        
        # 8. 更新统计
        self.observation_history.append(obs_text)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.total_reward += reward
        self.steps_taken += 1
        
        # 🧠 更新探索率
        self.exploration_rate = len(self.explored_cells) / (self.steps_taken + 1)
        
        return {
            "action": action,
            "reward": reward,
            "energy": self.energy,
            "position": self.position,
            "alive": self.alive,
            "total_reward": self.total_reward,
            "explored": len(self.explored_cells),
            "memories": len(self.food_memory),
        }
    
    def get_fitness(self) -> float:
        """计算适应度（用于进化选择）"""
        fitness = 0.0
        fitness += self.total_reward * 1.0
        fitness += self.food_collected * 5.0
        fitness += self.food_delivered * 10.0
        fitness += self.steps_taken * 0.1
        
        if self.steps_taken > 0:
            efficiency = self.food_delivered / (self.steps_taken + 1)
            fitness += efficiency * 50.0
        
        return max(0.0, fitness)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_decision_time = np.mean(self.decision_times) if self.decision_times else 0.0
        avg_reward = np.mean(self.reward_history) if self.reward_history else 0.0
        
        return {
            "agent_id": self.agent_id,
            "alive": self.alive,
            "energy": self.energy,
            "position": self.position,
            "fitness": self.get_fitness(),
            "total_reward": self.total_reward,
            "avg_reward": avg_reward,
            "food_collected": self.food_collected,
            "food_delivered": self.food_delivered,
            "steps": self.steps_taken,
            "avg_decision_ms": avg_decision_time * 1000,
            "lora_stats": self.lora.get_stats()
        }
    
    @classmethod
    def breed(cls, parent1: 'LLMAgentV2', parent2: 'LLMAgentV2', child_id: int, position: Tuple[int, int]) -> 'LLMAgentV2':
        """繁殖：合并两个parent的LoRA权重"""
        # 计算权重比例
        fitness1 = parent1.get_fitness()
        fitness2 = parent2.get_fitness()
        total_fitness = fitness1 + fitness2 + 1e-6
        ratio1 = fitness1 / total_fitness
        
        # 合并LoRA
        child_lora = parent1.lora.merge_with(parent2.lora, ratio=ratio1)
        child_lora.agent_id = child_id
        
        # 创建子代
        child = cls(child_id, position, energy=200, lora=child_lora)
        
        return child
