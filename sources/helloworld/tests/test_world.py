import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.world import World, WorldConf


def test_world_creation():
    conf = WorldConf(H=32, W=32)
    world = World(conf)
    
    assert world.H == 32
    assert world.W == 32
    assert len(world.layers) == len(World.LAYERS)
    
    for layer_name in World.LAYERS:
        assert layer_name in world.layers
        assert world.layers[layer_name].shape == (32, 32)
        assert world.layers[layer_name].dtype == np.uint8
    
    print("✓ test_world_creation passed")


def test_read_write_cell():
    conf = WorldConf(H=32, W=32)
    world = World(conf)
    
    world.write_cell("FOOD", 10, 10, 100)
    val = world.read_cell("FOOD", 10, 10)
    assert val == 100
    
    world.write_cell("FOOD", 10, 10, 300)
    val = world.read_cell("FOOD", 10, 10)
    assert val == 255
    
    out_of_bounds = world.read_cell("FOOD", 100, 100)
    assert out_of_bounds == 0
    
    world.write_cell("FOOD", 100, 100, 50)
    
    print("✓ test_read_write_cell passed")


def test_in_bounds():
    conf = WorldConf(H=32, W=32)
    world = World(conf)
    
    assert world.in_bounds(0, 0) == True
    assert world.in_bounds(31, 31) == True
    assert world.in_bounds(15, 15) == True
    
    assert world.in_bounds(-1, 0) == False
    assert world.in_bounds(0, -1) == False
    assert world.in_bounds(32, 0) == False
    assert world.in_bounds(0, 32) == False
    
    print("✓ test_in_bounds passed")


def test_pheromone_decay():
    conf = WorldConf(H=32, W=32, pher_decay=0.9, diffuse_weight=0.0)
    world = World(conf)
    
    world.write_cell("PHER_FOOD", 16, 16, 100)
    
    world.step_fields()
    
    val_after = world.read_cell("PHER_FOOD", 16, 16)
    
    assert val_after < 100
    assert val_after >= 85
    
    print("✓ test_pheromone_decay passed")


def test_pheromone_diffusion():
    conf = WorldConf(H=32, W=32, pher_decay=1.0, diffuse_weight=0.5)
    world = World(conf)
    
    world.write_cell("PHER_HOME", 16, 16, 200)
    
    world.step_fields()
    
    center = world.read_cell("PHER_HOME", 16, 16)
    north = world.read_cell("PHER_HOME", 16, 15)
    east = world.read_cell("PHER_HOME", 17, 16)
    
    assert north > 0
    assert east > 0
    assert center < 200
    
    print("✓ test_pheromone_diffusion passed")


def test_place_home():
    conf = WorldConf(H=32, W=32)
    world = World(conf)
    
    world.place_home(16, 16, radius=5)
    
    center_home = world.read_cell("HOME", 16, 16)
    assert center_home == 255
    
    edge_home = world.read_cell("HOME", 20, 16)
    assert edge_home == 255
    
    outside_home = world.read_cell("HOME", 25, 16)
    assert outside_home == 0
    
    print("✓ test_place_home passed")


def test_place_food_patch():
    conf = WorldConf(H=32, W=32)
    world = World(conf)
    
    world.place_food_patch(16, 16, radius=3, amount=50)
    
    center_food = world.read_cell("FOOD", 16, 16)
    assert center_food >= 50
    
    world.place_food_patch(16, 16, radius=3, amount=50)
    center_food_2 = world.read_cell("FOOD", 16, 16)
    assert center_food_2 >= center_food
    
    print("✓ test_place_food_patch passed")


def test_place_obstacle():
    conf = WorldConf(H=32, W=32)
    world = World(conf)
    
    world.place_obstacle_rect(5, 5, 10, 10)
    
    assert world.read_cell("SOLID", 5, 5) == 255
    assert world.read_cell("SOLID", 9, 9) == 255
    assert world.read_cell("SOLID", 10, 10) == 0
    assert world.read_cell("SOLID", 4, 4) == 0
    
    print("✓ test_place_obstacle passed")


def test_scatter_food():
    conf = WorldConf(H=64, W=64)
    world = World(conf)
    
    rng = np.random.default_rng(42)
    world.scatter_food(rng, num_patches=5, patch_radius=3, amount=50)
    
    total_food = np.sum(world.layers["FOOD"])
    assert total_food > 0
    
    print("✓ test_scatter_food passed")


if __name__ == "__main__":
    test_world_creation()
    test_read_write_cell()
    test_in_bounds()
    test_pheromone_decay()
    test_pheromone_diffusion()
    test_place_home()
    test_place_food_patch()
    test_place_obstacle()
    test_scatter_food()
    
    print("\nAll world tests passed! ✓")
