from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from core.agent import Agent
    from core.world import World


class BFInterpreter:
    
    def __init__(self, program_marks_layer: str = "MARK", tape_layer: str = "TAPE"):
        self.program_layer = program_marks_layer
        self.tape_layer = tape_layer
        
        self.pc = 0
        self.ptr_x = 0
        self.ptr_y = 0
        
        self.loop_stack = []
        
        self.program = []
        self.program_coords = []
        
        self.output_buffer = []
        self.input_buffer = []
        self.input_pos = 0
        
        self.halted = False
    
    def load_program(self, world: 'World', prog_string: str, start_x: int = 0, start_y: int = 0) -> None:
        self.program = list(prog_string)
        self.program_coords = []
        
        x, y = start_x, start_y
        for char in prog_string:
            self.program_coords.append((x, y))
            world.write_cell(self.program_layer, x, y, ord(char))
            x += 1
            if x >= world.W:
                x = 0
                y += 1
        
        self.pc = 0
        self.ptr_x = start_x
        self.ptr_y = max(start_y + 2, 10)
        self.loop_stack = []
        self.output_buffer = []
        self.input_pos = 0
        self.halted = False
    
    def set_input(self, input_string: str) -> None:
        self.input_buffer = [ord(c) for c in input_string]
        self.input_pos = 0
    
    def step(self, agent: 'Agent', world: 'World') -> None:
        if self.halted or self.pc >= len(self.program):
            self.halted = True
            return
        
        cmd = self.program[self.pc]
        
        if cmd == '>':
            self.ptr_x += 1
            if self.ptr_x >= world.W:
                self.ptr_x = 0
                self.ptr_y += 1
                if self.ptr_y >= world.H:
                    self.ptr_y = 0
        
        elif cmd == '<':
            self.ptr_x -= 1
            if self.ptr_x < 0:
                self.ptr_x = world.W - 1
                self.ptr_y -= 1
                if self.ptr_y < 0:
                    self.ptr_y = world.H - 1
        
        elif cmd == '+':
            val = world.read_cell(self.tape_layer, self.ptr_x, self.ptr_y)
            world.write_cell(self.tape_layer, self.ptr_x, self.ptr_y, (val + 1) % 256)
        
        elif cmd == '-':
            val = world.read_cell(self.tape_layer, self.ptr_x, self.ptr_y)
            world.write_cell(self.tape_layer, self.ptr_x, self.ptr_y, (val - 1) % 256)
        
        elif cmd == '.':
            val = world.read_cell(self.tape_layer, self.ptr_x, self.ptr_y)
            self.output_buffer.append(chr(val))
        
        elif cmd == ',':
            if self.input_pos < len(self.input_buffer):
                val = self.input_buffer[self.input_pos]
                world.write_cell(self.tape_layer, self.ptr_x, self.ptr_y, val)
                self.input_pos += 1
            else:
                world.write_cell(self.tape_layer, self.ptr_x, self.ptr_y, 0)
        
        elif cmd == '[':
            val = world.read_cell(self.tape_layer, self.ptr_x, self.ptr_y)
            if val == 0:
                depth = 1
                self.pc += 1
                while self.pc < len(self.program) and depth > 0:
                    if self.program[self.pc] == '[':
                        depth += 1
                    elif self.program[self.pc] == ']':
                        depth -= 1
                    self.pc += 1
                self.pc -= 1
            else:
                self.loop_stack.append(self.pc)
        
        elif cmd == ']':
            val = world.read_cell(self.tape_layer, self.ptr_x, self.ptr_y)
            if val != 0 and self.loop_stack:
                self.pc = self.loop_stack[-1]
            else:
                if self.loop_stack:
                    self.loop_stack.pop()
        
        self.pc += 1
        
        if self.pc >= len(self.program):
            self.halted = True
    
    def get_output(self) -> str:
        return ''.join(self.output_buffer)
