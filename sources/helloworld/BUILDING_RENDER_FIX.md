# 建筑渲染修复

## 问题描述

建筑物（NEST和STORAGE）的统计数值在增加，但在地图上没有显示。

## 原因分析

1. 建造NEST和STORAGE时，会同时设置SOLID=1（标记为固体，不可穿越）
2. 在渲染时，SOLID层显示为白色
3. 由于渲染顺序问题，白色的SOLID覆盖了建筑物的颜色
4. 结果：建筑物被白色完全覆盖，看不见

## 解决方案

修改渲染逻辑，让建筑物的颜色优先显示：

### 修改前：
```python
# SOLID显示为白色
solid = world.layers["SOLID"].astype(np.float32)
img[:, :, 0] += solid * 1.0  # 白色覆盖一切
img[:, :, 1] += solid * 1.0
img[:, :, 2] += solid * 1.0

# 建筑物颜色（但已被SOLID覆盖）
nest = world.layers["NEST"].astype(np.float32) / 255.0
img[:, :, 0] += nest * 0.9  # 被白色覆盖，看不见
```

### 修改后：
```python
# 检测哪些位置有建筑物
nest = world.layers["NEST"].astype(np.float32)
storage = world.layers["STORAGE"].astype(np.float32)
has_building = (nest > 0) | (storage > 0)

# 只在没有建筑物的地方显示白色SOLID
solid_display = solid * (~has_building)
img[:, :, 0] += solid_display * 1.0  # 白色

# 建筑物颜色正常显示
nest_norm = nest / 255.0
img[:, :, 0] += nest_norm * 0.9  # 现在可见
```

## 效果

- ✅ NEST（巢穴）显示为红橙色
- ✅ STORAGE（储藏室）显示为青绿色
- ✅ 普通SOLID墙壁显示为白色
- ✅ 建筑物的SOLID属性保持（不可穿越）

## 修改的文件

- `viz/advanced_render.py` - 修改渲染顺序和逻辑

## 测试

运行模拟后应该能看到：
1. 建造者（橙色代理）在食物丰富的地方建造
2. 红橙色的巢穴（NEST）出现
3. 青绿色的储藏室（STORAGE）出现
4. 普通白色墙壁仍然可见
5. 代理无法穿越建筑物（SOLID属性有效）

现在建筑物既有碰撞效果，又能正确显示颜色了！🏗️✨
