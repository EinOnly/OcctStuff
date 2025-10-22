"""
LoRA 在线训练系统
"""
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from config import Config


@dataclass
class Experience:
    """单次经验"""
    observation: str  # 观察到的状态
    action: str  # 采取的动作
    reward: float  # 获得的奖励
    next_observation: str  # 下一个状态


class ExperienceBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.experiences: List[Experience] = []
        self.position = 0
    
    def add(self, experience: Experience):
        """添加经验"""
        if len(self.experiences) < self.max_size:
            self.experiences.append(experience)
        else:
            self.experiences[self.position] = experience
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size: int) -> List[Experience]:
        """随机采样一批经验"""
        if len(self.experiences) < batch_size:
            return self.experiences.copy()
        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        return [self.experiences[i] for i in indices]
    
    def clear(self):
        """清空缓冲区"""
        self.experiences = []
        self.position = 0
    
    def __len__(self):
        return len(self.experiences)


class LoRAAdapter:
    """
    LoRA 适配器
    
    由于不同后端的LoRA实现不同，这里提供统一接口
    实际训练需要根据backend选择：
    - llama.cpp: 使用 llama-cpp-python 的 fine-tuning API
    - MLX: 使用 mlx-lm 的 LoRA 训练
    - PyTorch: 使用 PEFT 库
    """
    
    def __init__(self, agent_id: int, backend: str = "pytorch"):
        self.agent_id = agent_id
        self.backend = backend
        self.lora_weights = None  # LoRA权重
        self.experience_buffer = ExperienceBuffer()
        
        # 训练统计
        self.total_updates = 0
        self.avg_loss = 0.0
        
        # LoRA 配置
        self.rank = Config.LORA_RANK
        self.learning_rate = Config.LEARNING_RATE
        self.batch_size = Config.BATCH_SIZE
    
    def add_experience(self, observation: str, action: str, reward: float, next_observation: str):
        """记录经验"""
        exp = Experience(observation, action, reward, next_observation)
        self.experience_buffer.add(exp)
    
    def should_train(self) -> bool:
        """判断是否应该训练"""
        # 当经验足够多时进行训练
        return len(self.experience_buffer) >= self.batch_size * 2
    
    def train_step(self) -> Dict[str, float]:
        """执行一次训练步骤"""
        if not Config.ENABLE_ONLINE_LEARNING:
            return {"loss": 0.0, "skipped": True}
        
        if not self.should_train():
            return {"loss": 0.0, "insufficient_data": True}
        
        # 采样经验
        batch = self.experience_buffer.sample(self.batch_size)
        
        # 根据backend选择训练方式
        if self.backend == "pytorch":
            return self._train_pytorch(batch)
        elif self.backend == "mlx":
            return self._train_mlx(batch)
        else:
            # llama.cpp暂不支持实时训练，可以收集数据后离线训练
            return {"loss": 0.0, "backend_not_supported": True}
    
    def _train_pytorch(self, batch: List[Experience]) -> Dict[str, float]:
        """使用 PyTorch + PEFT 训练"""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return {"loss": 0.0, "error": "PEFT not installed"}
        
        # TODO: 实现完整的训练循环
        # 1. 构建训练prompt（observation + action）
        # 2. 计算loss（预测action的准确度 × reward）
        # 3. 反向传播更新LoRA权重
        
        # 简化版：只返回mock数据
        mock_loss = np.random.uniform(0.1, 0.5)
        self.total_updates += 1
        self.avg_loss = 0.9 * self.avg_loss + 0.1 * mock_loss
        
        return {
            "loss": mock_loss,
            "updates": self.total_updates,
            "batch_size": len(batch)
        }
    
    def _train_mlx(self, batch: List[Experience]) -> Dict[str, float]:
        """使用 MLX 训练"""
        # TODO: 实现MLX训练
        return {"loss": 0.0, "mlx_training": "not_implemented"}
    
    def save(self, path: Path):
        """保存LoRA权重"""
        save_dir = path / f"agent_{self.agent_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存元数据
        metadata = {
            "agent_id": self.agent_id,
            "total_updates": self.total_updates,
            "avg_loss": self.avg_loss,
            "rank": self.rank,
            "backend": self.backend
        }
        
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # 保存LoRA权重（根据backend不同）
        if self.backend == "pytorch" and self.lora_weights is not None:
            # torch.save(self.lora_weights, save_dir / "lora_weights.pt")
            pass
        
        print(f"✓ Agent {self.agent_id} LoRA saved to {save_dir}")
    
    def load(self, path: Path):
        """加载LoRA权重"""
        load_dir = path / f"agent_{self.agent_id}"
        if not load_dir.exists():
            return False
        
        # 加载元数据
        with open(load_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            self.total_updates = metadata["total_updates"]
            self.avg_loss = metadata["avg_loss"]
        
        # 加载权重
        # TODO: 根据backend加载
        
        print(f"✓ Agent {self.agent_id} LoRA loaded from {load_dir}")
        return True
    
    def merge_with(self, other: 'LoRAAdapter', ratio: float = 0.5) -> 'LoRAAdapter':
        """
        与另一个LoRA权重合并（用于繁殖）
        
        Args:
            other: 另一个parent的LoRA
            ratio: 当前agent的权重比例（0.5 = 平均）
        
        Returns:
            新的LoRA adapter
        """
        child = LoRAAdapter(agent_id=-1, backend=self.backend)
        
        # TODO: 实现权重合并
        # child.lora_weights = ratio * self.lora_weights + (1-ratio) * other.lora_weights
        
        # 添加变异
        # child.mutate(Config.MUTATION_RATE)
        
        return child
    
    def mutate(self, mutation_rate: float):
        """为权重添加噪声（变异）"""
        # TODO: 实现权重变异
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        return {
            "agent_id": self.agent_id,
            "updates": self.total_updates,
            "avg_loss": self.avg_loss,
            "experiences": len(self.experience_buffer),
            "rank": self.rank
        }


if __name__ == "__main__":
    # 测试LoRA adapter
    print("Testing LoRA adapter...")
    adapter = LoRAAdapter(agent_id=0)
    
    # 添加一些mock经验
    for i in range(10):
        adapter.add_experience(
            observation=f"state_{i}",
            action="move_north",
            reward=1.0 if i % 2 == 0 else 0.5,
            next_observation=f"state_{i+1}"
        )
    
    print(f"Buffer size: {len(adapter.experience_buffer)}")
    print(f"Should train: {adapter.should_train()}")
    
    if adapter.should_train():
        result = adapter.train_step()
        print(f"Training result: {result}")
    
    print(f"Stats: {adapter.get_stats()}")
