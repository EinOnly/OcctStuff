# 2D Finite Pixel World with Multi-Agent Simulation

A discrete 2D grid world simulation featuring:
- Multiple simple agents (ant/mice-like) with local perception
- Instinct tools for foraging behavior (hunt food, go home, deposit pheromones)
- Brainf*ck interpreter demonstrating Turing completeness
- Minimal visualization using matplotlib
- Deterministic simulation with fixed random seed

## Features

### World Model
- Layered 2D grid with fields: obstacles (SOLID), FOOD, HOME, pheromones (PHER_FOOD, PHER_HOME, PHER_ALERT), TAPE, MARK
- Pheromone diffusion and decay using 4-neighbor averaging
- Configurable world size (default 128×128)

### Agents
- Local perception with 3×3 neighborhood observation
- Primitive actions: movement, read/write, pickup/drop food, deposit pheromones
- Eight general-purpose registers and stack for computation
- Directional orientation (N, E, S, W)

### Instinct Tools
- `tool_hunt_food`: Follow food/pheromone gradients
- `tool_go_home`: Navigate back to home base
- `tool_drop_breadcrumb`: Deposit pheromone trails

### Policy Module
- Placeholder for tiny LLM-based decision making
- Simple rule-based baseline: carry food → go home, else hunt food
- Clean interface: observation string → action string

### Brainf*ck Interpreter
- Demonstrates Turing completeness
- Executes BF programs stored in MARK layer
- Uses TAPE layer for memory operations
- Supports all 8 BF instructions: `><+-.,[]`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Evolution Demo (Natural Selection) 🧬 BEST!
Watch agents **evolve** through natural selection with energy, death, reproduction, and genetics:
```bash
./start_evolution_demo.sh
# or
python run/evolution_demo.py --agents 10 --interval 10
```
- **Energy system**: Agents must eat to survive
- **Natural selection**: Weak agents die, fit ones reproduce
- **Genetic evolution**: 5 traits mutate over generations
- **Real-time charts**: Population, fitness, and energy trends
- **Emergent strategies**: Watch population optimize itself

**📖 See [EVOLUTION_GUIDE.md](EVOLUTION_GUIDE.md) for full details!**

### Live Demo (Infinite Loop) ⭐
Real-time visualization with continuous updates (no evolution):
```bash
python run/live_demo.py --agents 5 --world-size 128 --seed 42 --interval 50
```
- Runs indefinitely until stopped (Ctrl+C)
- Shows live agent positions, carry status, and statistics
- Auto-replenishes food every 100 steps
- Updates every 50ms (configurable with --interval)

**📖 See [LIVE_DEMO_GUIDE.md](LIVE_DEMO_GUIDE.md) for detailed usage, examples, and tips!**

### Foraging Demo (Multi-Agent)
Fixed number of steps with snapshot collection:
```bash
python run/sim.py --demo foraging --agents 5 --steps 1000 --seed 42
```

### Brainf*ck Demo
Demonstrates Turing completeness:
```bash
python run/sim.py --demo bf --steps 500 --seed 42
```

## Project Structure

```
helloworld/
  README.md              - This file
  pyproject.toml         - Project metadata
  requirements.txt       - Python dependencies
  start_live_demo.sh     - Quick launch script for live demo
  LIVE_DEMO_GUIDE.md     - Comprehensive live demo documentation
  LIVE_DEMO_COMPLETE.md  - Live demo feature summary
  run/
    sim.py              - Main simulation runner with CLI
    live_demo.py        - ⭐ NEW: Infinite loop live visualization
  core/
    world.py            - World model with layered fields
    agent.py            - Agent implementation with actions
  logic/
    tools.py            - Instinct tool functions
    bf_interpreter.py   - Brainf*ck interpreter
  policy/
    tiny_policy.py      - Policy decision module (LLM placeholder)
  viz/
    render.py           - Matplotlib visualization
  tests/
    test_world.py       - World model unit tests
    test_bf.py          - BF interpreter unit tests
    test_live.py        - Live simulation test (headless)
```

## Extending the Policy

The `tiny_policy.py` module provides a clean interface for integrating a distilled small LLM:

1. **Observation String**: Compact, stable format (<200 chars) with agent state and local perception
2. **Action Space**: Primitive actions + tool calls
3. **Integration Point**: Replace `decide_action()` with your LLM inference call

To integrate a real LLM:
- Keep the observation format stable
- Map LLM output tokens to action strings
- Add temperature/sampling controls if needed
- Consider caching for repeated observations

## Technical Notes

- Python 3.10+ required
- Pure numpy + matplotlib (no seaborn or heavyweight frameworks)
- Deterministic with `--seed` flag
- Headless rendering support (degrades gracefully without display)
- PEP 8 compliant code with English single-line comments

## Testing

```bash
python -m pytest tests/
```

## Performance

- Default world: 128×128 cells
- Typical runtime: <10 seconds for 1000 steps with 5 agents
- Pheromone diffusion optimized with numpy operations
