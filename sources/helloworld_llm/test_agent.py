"""
测试单个Agent - 快速验证系统是否工作
"""
import numpy as np
from pathlib import Path

from core.world import SimpleWorld
from core.llm_agent_v2 import LLMAgentV2
from config import Config


def test_single_agent():
    """测试单个agent的基本功能"""
    print("=" * 60)
    print("Testing Single LLM Agent")
    print("=" * 60)
    
    # 检查模型
    if Config.MODEL_TYPE == "mock":
        print(f"\n✓ Using Mock Model (no download needed)")
    else:
        model_path = Path(Config.MODEL_PATH)
        if not model_path.exists():
            print(f"\n❌ Model not found: {model_path}")
            print("Please download a model first (see DOWNLOAD_MODEL.md)")
            return
        print(f"\n✓ Model found: {model_path}")
    
    # 创建世界
    print("\nCreating world...")
    world = SimpleWorld(size=10)
    print(f"✓ World created: {world.size}x{world.size}")
    print(f"  Home at: {world.home_pos}")
    print(f"  Food locations: {len(world.food_locations)}")
    print(f"  Wall locations: {len(world.wall_locations)}")
    
    # 创建agent
    print("\nCreating agent...")
    start_pos = (5, 5)
    agent = LLMAgentV2(agent_id=0, position=start_pos, energy=100)
    print(f"✓ Agent created at {start_pos}")
    
    # 运行几步
    print("\n" + "=" * 60)
    print("Running 20 steps...")
    print("=" * 60)
    
    for step in range(20):
        if not agent.alive:
            print(f"\n❌ Agent died at step {step}")
            break
        
        result = agent.step(world)
        
        print(f"\nStep {step+1}:")
        print(f"  Position: {result['position']}")
        print(f"  Action: {result['action']}")
        print(f"  Reward: {result['reward']:+.1f}")
        print(f"  Energy: {result['energy']}")
        print(f"  Total Reward: {result['total_reward']:.1f}")
    
    # 显示最终统计
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    
    stats = agent.get_stats()
    for key, value in stats.items():
        if key != "lora_stats":
            print(f"  {key}: {value}")
    
    # LoRA统计
    print("\nLoRA Statistics:")
    lora_stats = stats["lora_stats"]
    for key, value in lora_stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Test completed!")


def test_model_loading():
    """仅测试模型加载"""
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    from core.model_loader import LocalModelLoader
    
    try:
        print("\nLoading model...")
        loader = LocalModelLoader()
        loader.load()
        
        print("\n✓ Model loaded successfully!")
        print(f"  Backend: {loader.backend}")
        print(f"  Path: {loader.model_path}")
        
        # 测试生成
        print("\nTesting generation...")
        prompt = "At (5,3), E80. See: food↑ Action:"
        response = loader.generate(prompt, max_tokens=10)
        
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response}")
        
        print("\n✓ Generation works!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if model file exists")
        print("2. Verify MODEL_PATH in .env")
        print("3. Install required dependencies")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "model":
        test_model_loading()
    else:
        print("\nOptions:")
        print("  python test_agent.py       - Test full agent")
        print("  python test_agent.py model - Test model loading only")
        print()
        
        choice = input("Press Enter to test full agent, or 'm' for model only: ").strip().lower()
        
        if choice == 'm':
            test_model_loading()
        else:
            test_single_agent()
