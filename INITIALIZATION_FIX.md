# 初始化问题修复说明

## 问题分析

当前组件使用 `settings.py` 中的参数初始化，但 `ct`、`cb` 以及 pattern 画面没有正确在初始化时刷新。

### 根本原因

1. **模式未正确设置**: 在应用 `ct`/`cb` 参数之前，需要先将模式设置为 Mode A
2. **初始化视图更新不完整**: `_initialize_ui_from_settings()` 调用了 `_update_views_without_chart()`，导致图表未计算

## 修复方案

### 1. 在 `__init__` 中正确设置模式

```python
# 修改前：直接应用参数，不考虑模式
pattern_defaults = {
    'vb': pattern_cfg.get('vbh'),
    'ct': pattern_cfg.get('ct'),
    'cb': pattern_cfg.get('cb'),
    'epn': pattern_cfg.get('epn'),
    'epm': pattern_cfg.get('epm'),
}
for label, value in pattern_defaults.items():
    if value is not None:
        self.assembly_builder.set_pattern_variable(label, value)

# 修改后：先判断并设置正确的模式
has_ct_cb = pattern_cfg.get('ct') is not None or pattern_cfg.get('cb') is not None
has_epn_epm = pattern_cfg.get('epn') is not None or pattern_cfg.get('epm') is not None

# 根据配置文件中的参数类型设置模式
if has_ct_cb and not has_epn_epm:
    self.assembly_builder.set_pattern_mode('A')  # ct/cb -> Mode A
elif has_epn_epm and not has_ct_cb:
    self.assembly_builder.set_pattern_mode('B')  # epn/epm -> Mode B

# 然后再应用参数
pattern_defaults = {
    'vb': pattern_cfg.get('vbh'),
    'ct': pattern_cfg.get('ct'),
    'cb': pattern_cfg.get('cb'),
    'epn': pattern_cfg.get('epn'),
    'epm': pattern_cfg.get('epm'),
}
for label, value in pattern_defaults.items():
    if value is not None:
        self.assembly_builder.set_pattern_variable(label, value)
```

### 2. 完整更新初始化视图

```python
# 修改前：不更新图表
def _initialize_ui_from_settings(self):
    self._sync_input_fields()
    self._build_slider_panel()
    self._update_slider_ranges()
    self._update_sliders_from_pattern()
    self._update_views_without_chart()  # 不更新图表！

# 修改后：完整更新包括图表
def _initialize_ui_from_settings(self):
    """Initialize UI components from settings and trigger full update."""
    self._sync_input_fields()
    self._build_slider_panel()
    self._update_slider_ranges()
    self._update_sliders_from_pattern()
    
    # Trigger full view update including chart on initialization
    self._update_views()  # 包含图表更新
```

## settings.py 配置说明

### Mode A 配置 (使用 ct/cb)

```python
pattern_p = {
    "bbox": {
        "width": 5.890,
        "height": 7.500,
    },
    "pattern": {
        "vbh": 10.0,        # vertical bottom height
        "ct": 2.94500,      # corner top (只在 Mode A 使用)
        "cb": 2.94500,      # corner bottom (只在 Mode A 使用)
        # 不要同时设置 epn/epm
        "thickness": 0.047,
        "width": 0.544,
    },
    # ...
}
```

### Mode B 配置 (使用 epn/epm)

```python
pattern_p = {
    "bbox": {
        "width": 5.890,
        "height": 7.500,
    },
    "pattern": {
        "vbh": 10.0,
        # 不要设置 ct/cb
        "epn": 1.60,        # exponent (只在 Mode B 使用)
        "epm": 0.450,       # exponent_m (只在 Mode B 使用)
        "thickness": 0.047,
        "width": 0.544,
    },
    # ...
}
```

## 初始化流程

1. **读取 settings.py**
   - bbox 配置 (width, height)
   - pattern 配置 (vbh, ct, cb, epn, epm)
   - assembly 配置 (spacing, count)

2. **创建基础对象**
   ```python
   pattern = Pattern(width=bbox_width, height=bbox_height)
   assembly_builder = AssemblyBuilder(pattern=pattern, ...)
   ```

3. **判断并设置模式**
   ```python
   if has_ct_cb:
       set_pattern_mode('A')
   elif has_epn_epm:
       set_pattern_mode('B')
   ```

4. **应用参数值**
   ```python
   for label, value in pattern_defaults.items():
       if value is not None:
           set_pattern_variable(label, value)
   ```

5. **初始化 UI**
   ```python
   QTimer.singleShot(0, self._initialize_ui_from_settings)
   ```

6. **完整更新视图**
   - 同步输入框
   - 重建滑块面板
   - 更新滑块范围
   - 从 pattern 同步滑块值
   - **触发完整视图更新（包括图表）**

## 验证

初始化完成后应该看到：

1. ✓ Mode 按钮显示正确的模式 (Mode A 或 Mode B)
2. ✓ 滑块显示正确的参数值 (ct, cb 或 epn, epm)
3. ✓ Pattern 窗口显示正确的图形
4. ✓ Chart 窗口显示完整的曲线图
5. ✓ Assembly 窗口显示阵列

## 注意事项

1. **不要混合使用参数**: 
   - Mode A 只使用 ct/cb
   - Mode B 只使用 epn/epm
   - 在 settings.py 中只设置一组参数

2. **初始化时机**: 
   - 使用 `QTimer.singleShot(0, ...)` 确保 UI 组件完全创建后再初始化
   
3. **视图更新**: 
   - 初始化时必须调用 `_update_views()` 而不是 `_update_views_without_chart()`
   - 这样才能确保图表被正确计算和显示
