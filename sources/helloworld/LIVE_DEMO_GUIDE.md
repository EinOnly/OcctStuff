# Live Demo Guide

## Overview

The **Live Demo** provides an infinite loop visualization showing real-time agent behavior in the 2D pixel world. Unlike the fixed-step demos, this runs continuously until you stop it, with live updates showing agent movements, food collection, pheromone trails, and statistics.

## Quick Start

```bash
cd /Users/ein/EinDev/OcctStuff/sources/helloworld
source /Users/ein/EinDev/OcctStuff/.venv/bin/activate
python run/live_demo.py --agents 5 --interval 50
```

Press `Ctrl+C` to stop the simulation.

## Command Line Options

```bash
python run/live_demo.py [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--agents` | int | 5 | Number of foraging agents |
| `--world-size` | int | 128 | Size of the world grid (NxN) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--interval` | int | 50 | Animation update interval in milliseconds |

### Examples

**Standard demo (5 agents, medium speed):**
```bash
python run/live_demo.py
```

**High-speed with many agents:**
```bash
python run/live_demo.py --agents 10 --interval 1
```

**Small world, slow motion:**
```bash
python run/live_demo.py --agents 3 --world-size 64 --interval 200
```

**Large world, many agents:**
```bash
python run/live_demo.py --agents 20 --world-size 256 --interval 25
```

## Display Layout

The visualization window has two panels:

### Left Panel: World View
- **Green areas**: Food patches
- **Blue area**: Home base (center)
- **Gray borders**: Obstacles/walls
- **White triangles**: Agents (empty handed)
- **Yellow triangles**: Agents carrying food
- **Red tint**: Pheromone trails (food and home markers)
- **Purple tint**: Tape/mark data (for BF operations)

Triangle orientation shows agent direction (N/E/S/W).

### Right Panel: Statistics
Real-time information updated every frame:

**Simulation Status:**
- Current step count
- Number of active agents
- Food currently at home base
- Total food gathered so far

**Agent Details (for each agent):**
- Agent ID
- Current position (x, y)
- Facing direction (N/E/S/W)
- Carry status (🟡 carrying food / ⚪ empty)

**World State:**
- Total food remaining in world
- Food pheromone intensity
- Home pheromone intensity

## Features

### 1. Infinite Loop
The simulation runs continuously until you stop it with `Ctrl+C`. Perfect for:
- Observing long-term emergent behavior
- Watching pheromone trail formation
- Studying agent foraging strategies
- Demonstrations and presentations

### 2. Auto-Replenishment
Every 100 steps, the simulation spawns 2 new food patches to keep the environment dynamic. This prevents the world from becoming depleted and maintains interesting agent behavior indefinitely.

### 3. Real-Time Statistics
All metrics update live in the info panel:
- See exactly which agent is carrying food
- Track total food gathered over time
- Monitor pheromone trail strength
- Watch agent positions update

### 4. Smooth Animation
Using matplotlib's FuncAnimation for smooth, efficient rendering:
- Adjustable frame rate with `--interval`
- Responsive to window resizing
- Clean matplotlib rendering

## How Agents Behave

Agents follow a simple but effective policy:

**When NOT carrying food:**
1. Use `tool_hunt_food`: Follow food pheromone gradients
2. Pick up food when found
3. Occasionally drop breadcrumb pheromones

**When carrying food:**
1. Use `tool_go_home`: Follow home pheromone gradients
2. Navigate back to home base
3. Drop food when home
4. Leave stronger pheromone trails

The pheromones naturally diffuse and decay, creating emergent trail networks that help all agents find food and home more efficiently over time.

## Performance Tips

### For Smooth Visualization
- **Small world** (64x64): Very smooth, good for detailed observation
- **Medium world** (128x128): Default, balanced performance
- **Large world** (256x256): May slow down with many agents

### For High Speed
- Use `--interval 1` for fastest updates
- Fewer agents (3-5) run faster than many (10+)
- Smaller world size improves performance

### For Detailed Observation
- Use `--interval 200` for slow motion
- Small world size (64x64) shows more detail
- Fewer agents (2-3) easier to track individually

## Stopping the Simulation

Press `Ctrl+C` in the terminal. The demo will:
1. Stop the animation gracefully
2. Close the visualization window
3. Print final statistics:
   - Total steps executed
   - Total food gathered
   
Example output:
```
Demo stopped at step 1547
Total food gathered: 3821
```

## Troubleshooting

### Issue: Window doesn't appear
**Solution:** Make sure you're not running headless. The demo requires a display.

If running over SSH:
```bash
export DISPLAY=:0
python run/live_demo.py
```

Or use the non-interactive demos instead:
```bash
python run/sim.py --demo foraging --steps 1000
```

### Issue: Animation is too slow
**Solution:** Reduce interval or world size:
```bash
python run/live_demo.py --interval 1 --world-size 64
```

### Issue: Animation is too fast to see
**Solution:** Increase interval:
```bash
python run/live_demo.py --interval 200
```

### Issue: Too many agents clutter the view
**Solution:** Reduce agent count:
```bash
python run/live_demo.py --agents 2
```

## Comparison with Other Demos

| Feature | Live Demo | sim.py Foraging | sim.py BF |
|---------|-----------|-----------------|-----------|
| Duration | Infinite | Fixed steps | Fixed steps |
| Visualization | Real-time | Snapshots only | Snapshots only |
| Statistics | Live updates | Final summary | Output text |
| Food replenishment | Auto (every 100 steps) | Initial only | N/A |
| Use case | Observation, presentation | Testing, benchmarks | Turing completeness |
| Interaction | Watch live | Run and review | Run and review |

## Advanced Usage

### Custom World Setup

Edit `run/live_demo.py` to customize:

**More food patches:**
```python
self.world.scatter_food(self.rng, num_patches=30, patch_radius=5, amount=100)
```

**Different home location:**
```python
home_x, home_y = 30, 30  # Top-left instead of center
```

**Add interior obstacles:**
```python
self.world.place_obstacle_rect(40, 40, 80, 80)  # Center block
```

### Recording the Demo

Use matplotlib's save functionality or screen recording:

**Save final frame:**
```python
plt.savefig('final_state.png', dpi=150)
```

**Screen recording (macOS):**
```bash
# Start recording with Cmd+Shift+5, then:
python run/live_demo.py
# Stop recording when done
```

## Integration with Policy Module

The live demo uses the same `policy/tiny_policy.py` module as other demos. To integrate your own LLM:

1. Replace `decide_action()` in `policy/tiny_policy.py`
2. The live demo will automatically use your new policy
3. Observe agent behavior in real-time
4. Iterate and improve based on visual feedback

This makes the live demo perfect for **policy development and debugging**.

## Educational Use

Great for teaching concepts:
- **Emergent behavior**: Watch pheromone trails self-organize
- **Multi-agent systems**: See cooperation emerge without coordination
- **Reinforcement learning**: Observe exploration vs exploitation
- **Swarm intelligence**: Ant-like foraging patterns

Students can modify parameters and immediately see effects!

## Summary

The live demo is your window into the pixel world. Use it to:
- ✅ Watch agents forage in real-time
- ✅ See pheromone trails form and evolve
- ✅ Monitor statistics and agent status live
- ✅ Debug and develop policies visually
- ✅ Create engaging demonstrations
- ✅ Study emergent behavior over time

Start watching now:
```bash
python run/live_demo.py
```

Enjoy the show! 🎬🐜
