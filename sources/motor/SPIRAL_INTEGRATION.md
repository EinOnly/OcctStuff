# 螺旋缠绕功能 - 集成说明

## 功能概述

已成功将螺旋缠绕功能集成到主应用程序中。现在您可以通过UI中的"Spiral Mode"按钮在平面挤出和螺旋缠绕之间切换。

## 使用方法

### 1. 启动应用
```bash
cd /Users/ein/EinDev/OcctStuff/sources/motor
python main.py
```

### 2. 生成Pattern
1. 在左侧参数面板中设置pattern参数
2. 点击"Generate Layers"生成layers

### 3. 启用螺旋模式
1. 在参数面板中找到"Spiral Mode"按钮（在Twist和Symmetry旁边）
2. 点击"Spiral Mode"按钮使其激活（高亮显示）
3. 点击"Refresh View"按钮重新生成3D视图

### 4. 查看结果
- 3D视图将显示螺旋缠绕的pattern
- Front layer patterns会映射到螺旋外表面
- Back layer patterns会映射到螺旋内表面

### 5. 导出STEP文件
- 点击"Save STEP"按钮导出螺旋缠绕的3D模型

## 当前螺旋参数

目前螺旋参数使用以下默认值（在 [app.py:122-124](app.py#L122-L124) 中定义）：

```python
radius = 6.2055   # 螺旋外半径 (mm)
thick = 0.1315    # 螺旋带宽度/圈距 (mm)
offset = 0.05     # 内外表面径向偏移 (mm)
```

Layer参数会从UI中动态读取：
- `layer_pbh`: Pattern基础高度
- `layer_ppw`: Pattern宽度padding
- `layer_ptc`: Pattern厚度（铜箔厚度）

## 工作原理

### 螺旋曲线生成

系统会自动生成三条同心螺旋曲线：
- **spiral_c (中心线)**: 位于 `radius - thick/2`
- **spiral_i (内螺旋)**: `centerline - offset`
- **spiral_o (外螺旋)**: `centerline + offset`

### Pattern映射

对于每个layer中的pattern：
1. Pattern的X坐标被解释为沿螺旋的弧长
2. Pattern的Y坐标被解释为垂直高度偏移
3. Front patterns映射到`spiral_o`（外表面）
4. Back patterns映射到`spiral_i`（内表面）

### 厚度挤出

每个映射后的pattern会沿径向法线方向挤出`layer_ptc`的厚度：
- Front patterns: 向外挤出
- Back patterns: 向内挤出

## 代码改动

### 新增文件
- [step.py](step.py): 添加了`Spiral`类（L369-612）
- [test_spiral.py](test_spiral.py): 测试脚本
- [debug_spiral.py](debug_spiral.py): 调试脚本
- [SPIRAL_USAGE.md](SPIRAL_USAGE.md): 详细使用文档

### 修改文件
- [app.py](app.py): 添加螺旋模式连接逻辑（L115-147）
- [parameters.py](parameters.py): 添加"Spiral Mode"按钮（L882-906）
- [step.py](step.py):
  - 添加`Spiral`类（L369-612）
  - 添加`create_spiral_wrapped_compound`方法（L174-295）
  - 添加`_create_wrapped_pattern_solid`方法（L297-367）
  - `StepViewer`添加`enable_spiral_mode`方法（L662-684）
  - 更新`refresh_view`支持螺旋模式（L686-731）

## 调试和测试

### 独立测试脚本
```bash
# 简单测试（使用矩形patterns）
python test_spiral.py

# 调试螺旋曲线生成
python debug_spiral.py
```

### 在主应用中调试
1. 启用Spiral Mode
2. 查看控制台输出，会显示：
   ```
   Spiral mode enabled: radius=6.2055, thick=0.1315, offset=0.05
     Layer params: pbh=..., ppw=..., ptc=...
   ```

## 已知问题和注意事项

### 1. Pattern尺寸
- Pattern的X坐标应该相对较小（推荐 < 1.0），因为它代表弧长
- 过大的X坐标会导致pattern跨越多圈螺旋

### 2. 可视化
- 螺旋缠绕后的pattern可能看起来比较小
- 使用视图控制键查看（按键1=俯视图，按键2=适应窗口）

### 3. 性能
- 大量patterns或高layer count可能导致处理时间较长
- 建议从少量patterns开始测试

## 未来改进方向

1. **参数配置化**: 将`radius`, `thick`, `offset`添加到UI参数面板
2. **自适应缩放**: 根据螺旋曲率自动调整pattern尺寸
3. **可变pitch**: 支持变pitch螺旋
4. **螺旋可视化**: 在UI中显示螺旋参考曲线
5. **多起点螺旋**: 支持多个起点的螺旋缠绕

## 参考文档

- [SPIRAL_USAGE.md](SPIRAL_USAGE.md): 详细技术文档
- [bak/alignment.py](bak/alignment.py): 螺旋数学参考实现
- [settings.py](settings.py): Layer配置参数

## 联系和支持

如有问题，请参考：
- 详细文档: [SPIRAL_USAGE.md](SPIRAL_USAGE.md)
- 测试脚本: [test_spiral.py](test_spiral.py)
- 调试脚本: [debug_spiral.py](debug_spiral.py)
