# Evolution Demo - Changes Summary

## 修改日期: 2025-10-20

### 实现的三个主要功能:

## 1. 🎨 可视化信息素 (Pheromone Visualization)

### 修改文件: `viz/advanced_render.py`

**变更内容:**
- 大幅增强食物信息素 (PHER_FOOD) 的可见度:
  - 红色通道: 0.25 → 0.6
  - 绿色通道: 0.2 → 0.5
  - 新增蓝色通道: 0.1
  
- 大幅增强返家信息素 (PHER_HOME) 的可见度:
  - 蓝色通道: 0.2 → 0.5
  - 绿色通道: 0.1 → 0.3
  - 新增红色通道: 0.1

- 在图例中新增了信息素标签:
  - Food Pher (食物信息素) - 橙黄色
  - Home Pher (返家信息素) - 蓝紫色

**效果:** 现在可以清晰地看到代理留下的信息素轨迹，更容易理解它们的行为模式。

---

## 2. 💀 能量耗尽死亡 + 繁殖系统 (Death & Breeding System)

### 修改文件: `run/advanced_evolution_demo.py`

**变更内容:**

### A. 自然死亡机制
在 `step()` 方法中添加了:
- 自动移除死亡的代理
- 打印死亡统计信息
- 当种群数量低于 25% 时触发紧急繁殖

```python
# 自然选择：从列表中移除死亡的代理
previously_alive = len(self.agents)
self.agents = [a for a in self.agents if a.alive]
deaths = previously_alive - len(self.agents)
if deaths > 0:
    print(f"💀 {deaths} agent(s) died from starvation. Remaining: {len(self.agents)}")
```

### B. 繁殖系统改革
将原来的"完全替换"改为"育种繁殖"系统:

**旧系统:** 每个繁殖周期完全替换所有代理
**新系统:** 
1. 保留最优秀的 25% 作为幸存者
2. 最优秀的 40% 成为繁殖父母池
3. 通过交叉繁殖产生后代
4. 后代数量 = 目标数量 - 幸存者数量

```python
# 选择繁殖父母 - 前40%可以繁殖
parent_pool_size = max(2, int(len(alive) * 0.4))
parents = alive[:parent_pool_size]

# 保留最优秀的父母（前25%）
survivors = alive[:max(2, int(len(alive) * 0.25))]
```

**效果:** 
- 更真实的进化模拟
- 优秀基因得以延续
- 种群动态更加自然

---

## 3. 🧱 固体墙壁碰撞检测 (Solid Wall Collision)

### 修改文件: 
- `logic/tools_advanced.py` (主要)
- `policy/policy_advanced.py`

**变更内容:**

### A. 移除模运算包装
所有使用 `% W` 和 `% H` 的地方都改为使用 `world.in_bounds()` 检查:

**之前:**
```python
nx = (x + dx) % W
ny = (y + dy) % H
if world.layers["SOLID"][ny, nx] > 0:
```

**现在:**
```python
nx = x + dx
ny = y + dy
if not world.in_bounds(nx, ny) or world.layers["SOLID"][ny, nx] > 0:
```

### B. 修改的函数
1. `tool_hunt_food_advanced()` - 寻找食物时的移动
2. `tool_go_home_advanced()` - 返回家园时的移动
3. `tool_find_nest_location()` - 查找巢穴位置
4. `tool_find_storage_location()` - 查找储藏位置
5. `tool_should_build_nest()` - 检查是否应该建造巢穴
6. `tool_should_build_storage()` - 检查是否应该建造储藏室
7. `_would_hit_wall()` - 墙壁碰撞检测
8. `policy_advanced.py` 中的邻近食物检测

**效果:**
- 代理不能再穿越固体墙壁
- 代理不能在世界边界"环绕"
- 必须绕路避开障碍物
- 更真实的物理行为

---

## 已验证的现有功能

### 能量耗尽死亡 (在 `core/advanced_agent.py` 中)
```python
# 能量耗尽则死亡
if self.energy <= 0:
    self.alive = False
```

这个功能已经存在于代码中，只需要确保在主循环中正确处理死亡代理即可。

---

## 测试建议

运行演示时应该观察到:

1. **信息素可见性**: 
   - 代理走过的路径应该显示明显的橙黄色(食物信息素)或蓝紫色(返家信息素)轨迹
   
2. **死亡机制**:
   - 代理能量耗尽时会从世界中消失
   - 控制台会打印死亡信息
   - 种群数量会动态变化

3. **繁殖系统**:
   - 每个繁殖周期后，最优秀的代理会存活下来
   - 新生代理是通过交叉繁殖产生的，不是随机生成
   - 控制台会显示幸存者和后代的数量

4. **墙壁碰撞**:
   - 代理遇到 SOLID 单元格时会转向
   - 代理到达世界边缘时会转向
   - 不会再看到代理"瞬移"到地图另一侧

---

## 运行命令

```bash
cd /Users/ein/EinDev/OcctStuff/sources/helloworld
./run_advanced.sh
```

或者:

```bash
cd /Users/ein/EinDev/OcctStuff/sources/helloworld
/Users/ein/EinDev/OcctStuff/.venv/bin/python run/advanced_evolution_demo.py
```
