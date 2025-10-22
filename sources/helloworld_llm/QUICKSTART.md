# 快速启动指南

## 第一次使用

### 1. 下载模型 (5分钟)

```bash
# 安装huggingface-cli
pip install huggingface-hub

# 创建模型目录
mkdir -p models
cd models

# 下载SmolLM-135M (推荐，约70MB)
huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF \
  smollm-135m-instruct.q4_k_m.gguf \
  --local-dir .

cd ..
```

**国内用户加速**：
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 然后执行上面的下载命令
```

### 2. 安装依赖 (2分钟)

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装核心依赖
pip install llama-cpp-python numpy matplotlib

# 如果要启用在线学习（可选）
pip install torch peft bitsandbytes
```

**macOS Apple Silicon 用户**（推荐使用MLX，更快）：
```bash
pip install mlx mlx-lm numpy matplotlib
```

### 3. 配置 (1分钟)

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env（确认MODEL_PATH正确）
# 如果模型在 models/smollm-135m-instruct.q4_k_m.gguf，无需修改
```

### 4. 测试 (1分钟)

```bash
# 测试模型加载
python test_agent.py model

# 如果成功，测试单个agent
python test_agent.py
```

### 5. 运行完整系统 (10-30分钟)

```bash
# 运行10代进化（每代500步）
python main.py
```

## 一键启动（推荐）

```bash
chmod +x run.sh
./run.sh
```

脚本会自动：
1. 检查Python
2. 创建/激活虚拟环境
3. 安装依赖
4. 检查模型
5. 创建配置文件
6. 运行模拟

## 常见问题

### Q: 下载模型太慢？
A: 使用国内镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 没有GPU，能运行吗？
A: 可以！SmolLM-135M在CPU上也很快（10-30ms）

### Q: 内存不够？
A: 
1. 使用更小的模型：TinyStories-33M (~20MB)
2. 减少agent数量：`EvolutionSimulation(num_agents=10)`

### Q: macOS上推理慢？
A: 使用MLX（Apple Silicon专用）：
```bash
pip install mlx mlx-lm
# 修改 .env: MODEL_TYPE=mlx
```

### Q: 在线学习不工作？
A: 
1. 确认安装了 torch + peft
2. 检查 .env: `ENABLE_ONLINE_LEARNING=true`
3. 注意：目前LoRA训练是mock实现，需要进一步开发

### Q: 想看详细日志？
A: 编辑 `main.py`，取消注释：
```python
# print(f"Agent {self.agent_id} trained: {train_result}")
```

## 配置优化

### 快速测试（5分钟）
```bash
# .env
MAX_TOKENS=20
ENABLE_ONLINE_LEARNING=false
```

```python
# main.py
sim.run(steps_per_generation=100, num_generations=3)
```

### 深度训练（1小时+）
```bash
# .env
MAX_TOKENS=50
ENABLE_ONLINE_LEARNING=true
BATCH_SIZE=8
```

```python
# main.py
sim.run(steps_per_generation=1000, num_generations=50)
```

### 大规模实验（需强力机器）
```python
sim = EvolutionSimulation(num_agents=50, world_size=30)
sim.run(steps_per_generation=2000, num_generations=100)
```

## 性能基准

### 硬件参考

| 配置 | 推理速度 | 20 agents/步 | 500步/代 |
|-----|---------|-------------|---------|
| M1 Mac (MLX) | ~10ms | ~0.2s | ~2min |
| Intel i7 (llama.cpp) | ~30ms | ~0.6s | ~5min |
| 云服务器 (CPU) | ~50ms | ~1s | ~8min |

### 内存使用

| 组件 | 内存 |
|-----|------|
| Base Model | 70MB |
| 20 LoRA adapters | 20MB |
| Python runtime | ~100MB |
| **总计** | **~190MB** |

## 下一步

1. **修改世界规则** → `core/world.py`
2. **调整奖励函数** → `core/llm_agent_v2.py` 的 `execute_action()`
3. **优化prompt** → `core/llm_agent_v2.py` 的 `_format_observation()`
4. **添加新action** → `_normalize_action()` 和 `execute_action()`
5. **实现真正的LoRA训练** → `core/lora_trainer.py` 的 `_train_pytorch()`

## 对比实验

同时运行rule-based和LLM-based：

```bash
# Terminal 1: Rule-based
cd ../helloworld
python run/advanced_evolution_demo.py

# Terminal 2: LLM-based
cd ../helloworld_llm
python main.py
```

观察：
- 决策质量
- 适应速度
- 资源效率
- 可解释性

## 获取帮助

1. 详细架构：[ARCHITECTURE.md](./ARCHITECTURE.md)
2. 模型下载：[DOWNLOAD_MODEL.md](./DOWNLOAD_MODEL.md)
3. 完整文档：[README_NEW.md](./README_NEW.md)
4. 代码注释：各个 `.py` 文件中的docstring

Happy evolving! 🧬🤖
