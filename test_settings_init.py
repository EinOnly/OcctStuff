#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test settings initialization for visualizer.
"""

import sys
sys.path.insert(0, '/Users/ein/EinDev/OcctStuff/sources/motor')

from pattern import Pattern
from assamble import AssemblyBuilder
from step import StepExporter
from settings import pattern_p

def test_initialization():
    """Test that settings are correctly applied during initialization."""
    print("=" * 60)
    print("Testing Settings Initialization")
    print("=" * 60)
    
    # Get settings
    cfg = pattern_p or {}
    bbox_cfg = cfg.get("bbox", {})
    pattern_cfg = cfg.get("pattern", {})
    assembly_cfg = cfg.get("assembly", {})
    
    print("\nSettings from pattern_p:")
    print(f"  bbox: {bbox_cfg}")
    print(f"  pattern: {pattern_cfg}")
    print(f"  assembly: {assembly_cfg}")
    
    # Initialize components
    extrude_thickness = pattern_cfg.get("thickness", 0.047)
    bbox_width = bbox_cfg.get("width", 5.89)
    bbox_height = bbox_cfg.get("height", 7.5)
    
    print(f"\nInitializing Pattern:")
    print(f"  width: {bbox_width}")
    print(f"  height: {bbox_height}")
    
    step_exporter = StepExporter(thickness=extrude_thickness)
    pattern = Pattern(width=bbox_width, height=bbox_height)
    assembly_builder = AssemblyBuilder(
        pattern=pattern,
        step_exporter=step_exporter,
    )
    
    # Determine mode from settings
    has_ct_cb = pattern_cfg.get('ct') is not None or pattern_cfg.get('cb') is not None
    has_epn_epm = pattern_cfg.get('epn') is not None or pattern_cfg.get('epm') is not None
    
    print(f"\nDetermining mode:")
    print(f"  has ct/cb: {has_ct_cb} (ct={pattern_cfg.get('ct')}, cb={pattern_cfg.get('cb')})")
    print(f"  has epn/epm: {has_epn_epm} (epn={pattern_cfg.get('epn')}, epm={pattern_cfg.get('epm')})")
    
    # Set mode before applying parameters
    if has_ct_cb and not has_epn_epm:
        print("  -> Setting Mode A")
        assembly_builder.set_pattern_mode('A')
    elif has_epn_epm and not has_ct_cb:
        print("  -> Setting Mode B")
        assembly_builder.set_pattern_mode('B')
    else:
        print(f"  -> Keeping default mode: {assembly_builder.get_pattern_mode()}")
    
    # Apply pattern parameters
    pattern_defaults = {
        'vb': pattern_cfg.get('vbh'),
        'ct': pattern_cfg.get('ct'),
        'cb': pattern_cfg.get('cb'),
        'epn': pattern_cfg.get('epn'),
        'epm': pattern_cfg.get('epm'),
    }
    
    print(f"\nApplying pattern parameters:")
    for label, value in pattern_defaults.items():
        if value is not None:
            print(f"  Setting {label} = {value}")
            assembly_builder.set_pattern_variable(label, value)
    
    # Check final state
    print(f"\nFinal pattern state:")
    print(f"  Mode: {assembly_builder.get_pattern_mode()}")
    print(f"  Pattern values: {assembly_builder.get_pattern_values()}")
    
    # Get variables to display
    print(f"\nPattern variables (for UI):")
    for var in assembly_builder.get_pattern_variables():
        print(f"  {var['label']}: {var['value']} (range: {var['min']}-{var['max']})")
    
    # Get shape and check it
    offset = assembly_builder.assembly.offset
    spacing = assembly_builder.assembly.spacing
    
    print(f"\nGenerating shape:")
    print(f"  offset: {offset}")
    print(f"  spacing: {spacing}")
    
    try:
        shape = assembly_builder.get_shape(offset, spacing)
        print(f"  ✓ Shape generated: {len(shape)} points")
        
        area = assembly_builder.get_shape_area(offset, spacing)
        print(f"  ✓ Area: {area:.4f} mm²")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_initialization()
    if success:
        print("\n✓ Initialization test passed!")
        sys.exit(0)
    else:
        print("\n✗ Initialization test failed!")
        sys.exit(1)
