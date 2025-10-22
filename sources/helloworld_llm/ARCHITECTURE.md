# 本地微型LLM Agent 演化系统

## 🎯 架构设计

### 核心理念
- **本地运行**: 100MB以内的微型LLM（0.1B-1B参数）
- **推理训练一体**: 在线学习，实时适应环境
- **多实例并行**: 20个agent同时运行
- **生命周期**: 可叠代、进化、遗传

## 🏗️ 技术栈

### 模型选择
1. **TinyLlama-1.1B** (~600MB) - 最强但稍大
2. **SmolLM-135M** (~100MB) ✅ **推荐**
3. **Phi-mini** (~200MB) 
4. **Custom Transformer** (~50MB) - 最轻量

### 框架
- **推理**: `llama.cpp` (CPU优化) 或 `MLX` (Apple Silicon)
- **训练**: LoRA微调 (低秩适应)
- **优化**: 4-bit量化

## 📊 系统架构

```
┌─────────────────────────────────────────┐
│         演化环境 (EvolutionEnv)          │
│  ┌────────────┐  ┌────────────┐        │
│  │ Agent 1    │  │ Agent 2    │  ...   │
│  │ - TinyLLM  │  │ - TinyLLM  │        │
│  │ - LoRA_1   │  │ - LoRA_2   │        │
│  │ - Memory   │  │ - Memory   │        │
│  └────────────┘  └────────────┘        │
│         ↓              ↓                │
│  ┌──────────────────────────┐          │
│  │   世界状态 (World)       │          │
│  │   - 食物 - 障碍 - 家园    │          │
│  └──────────────────────────┘          │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│        进化系统 (Evolution)              │
│  - 选择: 适应度排序                      │
│  - 繁殖: 合并LoRA权重                    │
│  - 变异: 添加噪声                        │
│  - 遗传: 传递知识                        │
└─────────────────────────────────────────┘
```

## 🧬 生命周期

### 1. 初始化
```python
agent = TinyLLMAgent(
    base_model="smollm-135m.gguf",
    lora_adapter=None,  # 初始无适配器
    energy=100.0
)
```

### 2. 生存期
```python
while agent.alive:
    # 观察环境
    obs = world.observe(agent.position)
    
    # LLM决策（推理）
    action = agent.decide(obs)
    
    # 执行动作
    reward = agent.act(action)
    
    # 在线学习（训练）
    agent.learn(obs, action, reward)
    
    # 消耗能量
    agent.energy -= cost
```

### 3. 死亡与繁殖
```python
if agent.energy <= 0:
    agent.die()
    # 保存LoRA权重用于遗传
    agent.save_lora("agent_{id}_gen_{gen}.pth")

# 繁殖
if generation_end:
    elite_agents = select_top_k(agents, k=5)
    new_agents = breed(elite_agents)  # 合并LoRA
```

## 💾 模型大小对比

| 模型 | 参数 | 量化后 | 推理速度 | 质量 |
|------|------|--------|----------|------|
| SmolLM-135M | 135M | ~70MB | ~50ms | ⭐⭐⭐ |
| TinyStories-33M | 33M | ~20MB | ~10ms | ⭐⭐ |
| Custom-50M | 50M | ~30MB | ~15ms | ⭐⭐⭐ |

## 🎮 训练策略

### A. 在线LoRA微调
- 每次行动后，根据reward更新LoRA
- 只更新adapter层（~1MB）
- 保持base model冻结

### B. 经验回放
- 维护经验缓冲区 (obs, action, reward)
- 定期批量训练
- 平滑学习曲线

### C. 进化遗传
```python
def breed(parent1, parent2):
    # 合并两个LoRA适配器
    child_lora = (parent1.lora + parent2.lora) / 2
    
    # 添加变异
    child_lora += noise * mutation_rate
    
    return TinyLLMAgent(base_model, child_lora)
```

## 🚀 性能优化

### CPU优化
- **llama.cpp**: 针对CPU的GGUF格式
- **ONNX Runtime**: 优化的推理引擎
- **批处理**: 同时推理多个agent

### Apple Silicon (M系列)
- **MLX**: Apple专用框架
- **Metal加速**: GPU加速
- **统一内存**: 高效数据传输

### 内存优化
```
20个agent × 70MB (共享base model) = 70MB base + 20×1MB LoRA = ~90MB total
```

## 📈 预期性能

### 单次决策
- **推理时间**: 10-50ms
- **训练时间**: 50-200ms（异步）
- **内存占用**: ~100MB

### 20个agent
- **并行推理**: ~500ms/step (批处理)
- **训练**: 异步后台进行
- **总内存**: ~200MB

## 🎯 实现计划

### Phase 1: 基础模型
- [ ] 下载SmolLM-135M
- [ ] 转换为GGUF/MLX格式
- [ ] 实现基础推理

### Phase 2: LoRA训练
- [ ] 实现LoRA适配器
- [ ] 在线学习pipeline
- [ ] 经验回放缓冲区

### Phase 3: 进化系统
- [ ] 适应度评估
- [ ] LoRA合并算法
- [ ] 变异和选择

### Phase 4: 可视化
- [ ] 实时显示20个agent
- [ ] 学习曲线
- [ ] LoRA权重热图

## 💡 创新点

1. **知识遗传**: 通过LoRA权重传递经验
2. **在线学习**: 边玩边学，不需要预训练
3. **轻量级**: 100MB内运行20个智能体
4. **本地化**: 无需API，完全离线

## 🔬 实验想法

1. **对抗进化**: 两个种群竞争资源
2. **知识蒸馏**: 从大模型蒸馏到小模型
3. **元学习**: 学会快速适应新环境
4. **社会学习**: agent互相模仿

想要我实现哪个部分？
