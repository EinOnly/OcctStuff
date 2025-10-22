"""
配置文件
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
load_dotenv()

class Config:
    # 项目路径
    PROJECT_ROOT = Path(__file__).parent
    MODEL_DIR = PROJECT_ROOT / "models"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    
    # 模型配置
    MODEL_PATH = os.getenv("MODEL_PATH", "HuggingFaceTB/SmolLM-135M")
    MODEL_TYPE = os.getenv("MODEL_TYPE", "transformers")  # mock, gguf, mlx, transformers
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "50"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "512"))
    
    # 训练配置
    ENABLE_ONLINE_LEARNING = os.getenv("ENABLE_ONLINE_LEARNING", "true").lower() == "true"
    LORA_RANK = int(os.getenv("LORA_RANK", "8"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0001"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
    
    # 进化配置
    MUTATION_RATE = float(os.getenv("MUTATION_RATE", "0.1"))
    ELITE_RATIO = float(os.getenv("ELITE_RATIO", "0.2"))
    
    # 世界配置
    WORLD_SIZE = 32
    INITIAL_FOOD_PATCHES = 3
    FOOD_PATCH_SIZE = 5
    FOOD_PATCH_AMOUNT = 50
    
    # 代理配置
    INITIAL_ENERGY = 100.0
    ENERGY_PER_STEP = 0.5
    ENERGY_FROM_FOOD = 20.0
    
    # 可视化配置
    RENDER_INTERVAL = 1000  # ms
    SHOW_GRID = False
    
    # 实验模式
    MODE = "single"  # "single", "mixed", "competitive"
    NUM_LLM_AGENTS = 1
    NUM_RULE_AGENTS = 0
    
    @classmethod
    def validate(cls):
        """验证配置"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")
        return True
