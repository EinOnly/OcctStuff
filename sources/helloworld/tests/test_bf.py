import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.world import World, WorldConf
from core.agent import Agent, Dir
from logic.bf_interpreter import BFInterpreter


def test_bf_increment():
    conf = WorldConf(H=32, W=64)
    world = World(conf)
    
    program = "++++"
    
    interpreter = BFInterpreter()
    interpreter.load_program(world, program, start_x=0, start_y=0)
    
    agent = Agent(id=0, x=0, y=0, d=Dir.N)
    
    for _ in range(len(program)):
        interpreter.step(agent, world)
    
    tape_val = world.read_cell("TAPE", interpreter.ptr_x, interpreter.ptr_y)
    assert tape_val == 4
    
    print("✓ test_bf_increment passed")


def test_bf_decrement():
    conf = WorldConf(H=32, W=64)
    world = World(conf)
    
    program = "++++--"
    
    interpreter = BFInterpreter()
    interpreter.load_program(world, program, start_x=0, start_y=0)
    
    agent = Agent(id=0, x=0, y=0, d=Dir.N)
    
    for _ in range(len(program)):
        interpreter.step(agent, world)
    
    tape_val = world.read_cell("TAPE", interpreter.ptr_x, interpreter.ptr_y)
    assert tape_val == 2
    
    print("✓ test_bf_decrement passed")


def test_bf_pointer_movement():
    conf = WorldConf(H=32, W=64)
    world = World(conf)
    
    program = "+++>++>+"
    
    interpreter = BFInterpreter()
    interpreter.load_program(world, program, start_x=0, start_y=0)
    start_x, start_y = interpreter.ptr_x, interpreter.ptr_y
    
    agent = Agent(id=0, x=0, y=0, d=Dir.N)
    
    for _ in range(len(program)):
        interpreter.step(agent, world)
    
    tape_val_0 = world.read_cell("TAPE", start_x, start_y)
    tape_val_1 = world.read_cell("TAPE", start_x + 1, start_y)
    tape_val_2 = world.read_cell("TAPE", start_x + 2, start_y)
    
    assert tape_val_0 == 3
    assert tape_val_1 == 2
    assert tape_val_2 == 1
    
    print("✓ test_bf_pointer_movement passed")


def test_bf_output():
    conf = WorldConf(H=32, W=64)
    world = World(conf)
    
    program = "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.++."
    
    interpreter = BFInterpreter()
    interpreter.load_program(world, program, start_x=0, start_y=0)
    
    agent = Agent(id=0, x=0, y=0, d=Dir.N)
    
    for _ in range(len(program)):
        interpreter.step(agent, world)
    
    output = interpreter.get_output()
    assert len(output) == 2
    assert ord(output[0]) == 65
    assert ord(output[1]) == 67
    
    print("✓ test_bf_output passed")


def test_bf_loop():
    conf = WorldConf(H=32, W=64)
    world = World(conf)
    
    program = "+++[>++<-]"
    
    interpreter = BFInterpreter()
    interpreter.load_program(world, program, start_x=0, start_y=0)
    start_x, start_y = interpreter.ptr_x, interpreter.ptr_y
    
    agent = Agent(id=0, x=0, y=0, d=Dir.N)
    
    max_steps = 100
    for _ in range(max_steps):
        if interpreter.halted:
            break
        interpreter.step(agent, world)
    
    tape_val_0 = world.read_cell("TAPE", start_x, start_y)
    tape_val_1 = world.read_cell("TAPE", start_x + 1, start_y)
    
    assert tape_val_0 == 0
    assert tape_val_1 == 6
    
    print("✓ test_bf_loop passed")


def test_bf_hello_world():
    conf = WorldConf(H=32, W=128)
    world = World(conf)
    
    program = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++."
    
    interpreter = BFInterpreter()
    interpreter.load_program(world, program, start_x=0, start_y=0)
    
    agent = Agent(id=0, x=0, y=0, d=Dir.N)
    
    max_steps = 1000
    for _ in range(max_steps):
        if interpreter.halted:
            break
        interpreter.step(agent, world)
    
    output = interpreter.get_output()
    
    assert "Hello" in output or "HELLO" in output.upper()
    
    print(f"✓ test_bf_hello_world passed (output: {repr(output)})")


def test_bf_input():
    conf = WorldConf(H=32, W=64)
    world = World(conf)
    
    program = ",+."
    
    interpreter = BFInterpreter()
    interpreter.load_program(world, program, start_x=0, start_y=0)
    interpreter.set_input("A")
    
    agent = Agent(id=0, x=0, y=0, d=Dir.N)
    
    for _ in range(len(program)):
        interpreter.step(agent, world)
    
    output = interpreter.get_output()
    assert len(output) == 1
    assert ord(output[0]) == ord('A') + 1
    
    print("✓ test_bf_input passed")


def test_bf_halted():
    conf = WorldConf(H=32, W=64)
    world = World(conf)
    
    program = "+++"
    
    interpreter = BFInterpreter()
    interpreter.load_program(world, program, start_x=0, start_y=0)
    
    agent = Agent(id=0, x=0, y=0, d=Dir.N)
    
    assert not interpreter.halted
    
    for _ in range(len(program)):
        interpreter.step(agent, world)
    
    assert interpreter.halted
    
    old_pc = interpreter.pc
    interpreter.step(agent, world)
    assert interpreter.pc == old_pc
    
    print("✓ test_bf_halted passed")


if __name__ == "__main__":
    test_bf_increment()
    test_bf_decrement()
    test_bf_pointer_movement()
    test_bf_output()
    test_bf_loop()
    test_bf_hello_world()
    test_bf_input()
    test_bf_halted()
    
    print("\nAll BF tests passed! ✓")
