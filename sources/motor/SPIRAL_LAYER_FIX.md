# 螺旋层数修复 (Spiral Layer Count Fix)

## 问题 (Problem)

之前的实现中，螺旋的圈数(`layer_count`)错误地使用了**pattern数量**（54个），而不是**物理层数**（4层）。

这导致：
- 螺旋转了54圈而不是4圈
- 螺旋半径从6.2mm缩减到接近圆心
- Pattern的相对位置关系错乱
- 视觉效果凌乱、间距不均

## 根本原因 (Root Cause)

在 [step.py:216](step.py#L216)：
```python
# 错误的实现
layer_count = max(len(front_shapes), len(back_shapes))  # 这是54 (pattern数量)
```

应该是：
```python
# 正确的实现
num_physical_layers = 4  # 物理缠绕层数
num_patterns = max(len(front_shapes), len(back_shapes))  # 54个patterns
```

## 修复内容 (Fixes)

### 1. 添加`num_physical_layers`参数

**[step.py:185-192](step.py#L185-L192)**
```python
def create_spiral_wrapped_compound(self, layers: Dict[str, Any],
                                   radius: float = 6.2055,
                                   thick: float = 0.1315,
                                   offset: float = 0.05,
                                   layer_pbh: float = 8.0,
                                   layer_ppw: float = 0.5,
                                   layer_ptc: float = 0.047,
                                   num_physical_layers: int = 4) -> TopoDS_Compound:
```

### 2. 区分物理层和patterns

**[step.py:227-235](step.py#L227-L235)**
```python
# 使用PHYSICAL layer count创建螺旋（4圈）
self.spiral = Spiral(radius, thick, num_physical_layers, offset)

print(f"Spiral geometry created:")
print(f"  Physical layers: {num_physical_layers}")  # 4
print(f"  Patterns per layer: {num_patterns // num_physical_layers}")  # ~13
print(f"  Total patterns: {num_patterns}")  # 54
```

### 3. Pattern均匀分布

**[step.py:240-242](step.py#L240-L242)**
```python
# 计算每个pattern的弧长间隔
arc_per_pattern = self.spiral.total_length / num_patterns

# 每个pattern沿螺旋均匀分布
arc_offset = idx * arc_per_pattern
```

### 4. 从实际加载配置读取物理层数

**[app.py:131-143](app.py#L131-L143)**
```python
# 从当前加载的参数快照中读取实际物理层数
params_snapshot = LPARAMS.snapshot()
num_physical_layers = len(params_snapshot.get("layers", []))  # 动态获取

self.occtWidget.enable_spiral_mode(
    radius=radius,
    thick=thick,
    offset=offset,
    layer_pbh=layer_pbh,
    layer_ppw=layer_ppw,
    layer_ptc=layer_ptc,
    num_physical_layers=num_physical_layers
)
```

**优势：**
- 动态从实际加载的配置读取，而不是硬编码
- 支持不同的配置文件
- 实时反映用户的配置更改

## 预期结果 (Expected Results)

### 螺旋几何

使用参数：
- `radius = 6.2055 mm`
- `thick = 0.1315 mm`
- `num_physical_layers = 4`
- `offset = 0.05 mm`

螺旋半径范围：
```
起始半径: 6.2055 - 0.1315/2 = 6.14 mm
结束半径: 6.14 - (4 * 0.1315) = 5.61 mm
总缩减: 0.526 mm (合理!)
```

### Pattern分布

- 总patterns: 54个
- 物理层: 4层
- 每层patterns: ~13-14个
- Patterns均匀分布在4圈螺旋上
- Pattern相对位置保持正确

## 测试

### 1. 查看螺旋曲线
```bash
cd /Users/ein/EinDev/OcctStuff/sources/motor
python3 test_spiral_curve.py
```

应该看到：
- ✅ 紧密的4圈螺旋
- ✅ 半径从6.14mm递减到5.61mm
- ✅ 均匀、光滑的曲线

### 2. 主应用测试
```bash
python3 main.py
```

步骤：
1. 点击 "Generate Layers"
2. 点击 "Spiral Mode"
3. 点击 "Refresh View"

应该看到：
- ✅ 54个patterns沿4圈螺旋均匀分布
- ✅ Pattern相对位置正确
- ✅ 视觉效果整齐、间距均匀

## 文件改动总结

### 修改文件
1. [step.py](step.py)
   - `create_spiral_wrapped_compound`: 添加`num_physical_layers`参数
   - `enable_spiral_mode`: 添加`num_physical_layers`参数
   - 修正螺旋创建逻辑
   - 修正pattern分布算法

2. [app.py](app.py)
   - 从`settings.py`读取物理层数
   - 传递正确的参数到spiral mode

3. [test_spiral_curve.py](test_spiral_curve.py)
   - 更新`layer_count = 4`（物理层）

## 关键概念区分

| 概念 | 英文 | 值 | 说明 |
|------|------|-----|------|
| 物理层数 | Physical Layers | 4 | 电机的实际缠绕层数 |
| Pattern数量 | Pattern Count | 54 | 每层的pattern图案数量 |
| 螺旋圈数 | Spiral Turns | 4 | 螺旋线转的圈数 = 物理层数 |
| 总Pattern数 | Total Patterns | 54 × 2 = 108 | Front + Back patterns |

## 重要提醒

**永远不要混淆：**
- `num_physical_layers` = 物理层数（来自settings.layers）
- `num_patterns` = pattern数量（来自layers.front/back）

螺旋的圈数应该等于**物理层数**，而不是pattern数量！
