# 高级演化系统 (Advanced Evolution System)

## 🎯 新增功能

### 1. 精英基因遗传 (Elite Genome Inheritance)
- **EliteGenome类**: 7个基因特征控制行为
  - `exploration_rate`: 探索倾向
  - `food_attraction`: 食物吸引力
  - `home_attraction`: 家园吸引力
  - `pheromone_sensitivity`: 信息素敏感度
  - `energy_efficiency`: 能量效率
  - `building_tendency`: 建造倾向
  - `body_size`: 体型大小

- **精英继承策略**: 每代最优个体的基因作为下一代基准
  - 使用`EliteGenome.from_elite()`方法继承精英基因
  - 较小的变异率(0.08)确保优秀特征稳定传承
  - 次优个体的部分特征混合到后代中

### 2. 可视化体型变化 (Visual Body Representation)
- **颜色编码**: 
  - HSV色彩空间表示基因组合
  - 色调(Hue): 探索率 + 食物吸引 + 信息素敏感度
  - 饱和度(Saturation): 能量效率
  - 明度(Value): 体型大小

- **体型大小**: 
  - 基础大小2.0，根据body_size基因在0.5-3.5范围内变化
  - 大体型agent在界面上更显眼
  - 带头部指示方向的身体表示

- **代数标记**: 
  - 每个agent头顶显示"G{代数}"标签
  - 方便追踪世代演化

### 3. 建造系统 (Building System)
三种结构类型：

#### 巢穴 (Nest)
- 能量消耗: 15
- 颜色: 橙棕色 (0.5, 0.3, 0.1)
- 功能: 聚集点，靠近食物和家园
- 策略: 在食物和家园附近的最优位置建造

#### 储藏室 (Storage)
- 能量消耗: 12
- 颜色: 青绿色 (0.3, 0.5, 0.4)
- 功能: 靠近巢穴和路径的存储点
- 策略: 建在巢穴附近

#### 路径 (Trail)
- 能量消耗: 2
- 颜色: 浅棕色 (0.3, 0.25, 0.2)
- 功能: 标记常用路径
- 策略: 移动时高building_tendency个体自动留下

### 4. 可扩展世界 (Expandable World)
- **初始大小**: 64×64 (可配置)
- **最大大小**: 256×256 (可配置)
- **扩展机制**: 
  - 当agents接近边界(距离<3)时记录访问次数
  - 达到阈值(默认5次)后向该方向扩展32像素
  - 四个方向独立扩展: 上/下/左/右

- **实时反馈**: 
  - 扩展时打印: "🌍 World expanded! New size: WxH (expansion #N)"
  - 图表显示世界大小变化趋势

### 5. 优雅的UI布局 (Elegant Layout)
6个面板展示：

1. **世界视图** (2列宽，3行高)
   - 实时渲染agents、食物、建筑
   - 方向指示的身体+头部
   - 图例显示各种元素

2. **状态面板** (右上)
   - 模拟统计: 步数、代数、世界大小、扩展次数
   - 种群信息: 存活数、平均能量、平均fitness、最大fitness、平均代数
   - 建筑统计: 巢穴、储藏室、路径数量

3. **基因面板** (右上)
   - 显示当前精英个体的7个基因值
   - 实时更新最优基因组合

4. **种群图表** (右中)
   - 每代存活agent数量
   - 蓝色曲线

5. **Fitness图表** (右中)
   - 每代平均fitness
   - 绿色曲线

6. **世界大小图表** (右下)
   - 总像素数变化
   - 紫色曲线

7. **建筑数量图表** (右下)
   - 所有建筑物总数
   - 橙色曲线

## 🚀 运行方法

```bash
cd /Users/ein/EinDev/OcctStuff
PYTHONPATH=/Users/ein/EinDev/OcctStuff/sources/helloworld:$PYTHONPATH \
/Users/ein/EinDev/OcctStuff/.venv/bin/python \
sources/helloworld/run/advanced_evolution_demo.py
```

或使用便捷脚本：

```bash
cd sources/helloworld
../../.venv/bin/python run/advanced_evolution_demo.py
```

## 📊 观察要点

1. **世界扩展**: 注意终端输出的扩展通知
2. **精英Fitness**: 每250步繁殖后打印的精英fitness应该逐渐增长
3. **体型/颜色多样性**: 观察agents的大小和颜色如何随代数变化
4. **建筑分布**: 巢穴和储藏室会在食物/家园附近聚集
5. **探索行为**: agents会探索世界边界，触发扩展

## 🎮 参数调整

在`advanced_evolution_demo.py`的main中：

```python
demo = AdvancedEvolutionDemo(
    n_agents=25,      # agent数量
    init_size=64,     # 初始世界大小
    max_size=256,     # 最大世界大小
)
```

在`WorldConf`中调整：

```python
conf = WorldConf(
    H=init_size,
    W=init_size,
    max_H=max_size,
    max_W=max_size,
    expand_threshold=5,   # 扩展阈值（边界访问次数）
    pher_decay=0.95,      # 信息素衰减率
    diffuse_weight=0.2,   # 扩散权重
)
```

## 🧬 基因调优

修改`EliteGenome.from_elite()`中的mutation_rate：
- 默认: 0.08 (较稳定的演化)
- 更高 (0.15): 更多变异，探索更多可能性
- 更低 (0.05): 更保守，保持优秀特征

## 📁 新增文件

- `core/expandable_world.py` - 可扩展世界系统
- `core/advanced_agent.py` - 高级agent with精英基因
- `logic/tools_advanced.py` - 建造感知工具
- `policy/policy_advanced.py` - 建造策略
- `viz/advanced_render.py` - 优雅可视化
- `run/advanced_evolution_demo.py` - 完整演示

## 🐛 已知限制

1. matplotlib字体可能缺少emoji（🌍📊🧬等），不影响功能
2. 信息素可能发生overflow警告，已用np.clip限制在0-255
3. tight_layout警告，布局仍然正常工作

## 🎯 未来增强

- [ ] 多物种竞争（不同颜色的族群）
- [ ] 资源再生机制
- [ ] 复杂建筑组合（巢穴+储藏室形成基地）
- [ ] 社会行为（协作、战斗）
- [ ] 保存/加载精英基因库
- [ ] 3D可视化
