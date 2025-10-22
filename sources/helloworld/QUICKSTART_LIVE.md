# 🎬 Quick Reference: Live Demo

## One-Line Launch
```bash
./start_live_demo.sh
```

## Common Commands

### Default Demo
```bash
python run/live_demo.py
```

### Fast Mode
```bash
python run/live_demo.py --interval 1
```

### Slow Motion
```bash
python run/live_demo.py --interval 200
```

### Many Agents
```bash
python run/live_demo.py --agents 10
```

### Small World
```bash
python run/live_demo.py --world-size 64
```

### Large World
```bash
python run/live_demo.py --agents 20 --world-size 256
```

## Controls
- **Stop**: Press `Ctrl+C`
- **Resize**: Drag window edges (matplotlib)

## What to Watch For

🟢 **Green patches** = Food  
🔵 **Blue center** = Home  
🟡 **Yellow triangle** = Agent carrying food  
⚪ **White triangle** = Agent empty  
🔴 **Red tint** = Pheromone trails  

## Display Panels

**Left**: Live world visualization  
**Right**: Real-time statistics

## Key Behaviors

1. **Hunt** → Follow food pheromones
2. **Collect** → Pick up food (turn yellow)
3. **Return** → Follow home pheromones
4. **Deliver** → Drop at home (turn white)
5. **Trail** → Leave pheromones
6. **Repeat** → Forever!

## Docs
- 📖 Full guide: `LIVE_DEMO_GUIDE.md`
- 📋 Feature list: `LIVE_DEMO_COMPLETE.md`
- 📚 Main docs: `README.md`

## Tests
```bash
# Headless test (no GUI)
python tests/test_live.py

# All tests
python tests/test_world.py
python tests/test_bf.py
```

## Parameters Quick Reference

| Flag | Default | Range | Effect |
|------|---------|-------|--------|
| `--agents` | 5 | 1-50 | More agents = more activity |
| `--world-size` | 128 | 32-512 | Bigger = more space |
| `--seed` | 42 | any int | Different patterns |
| `--interval` | 50 | 1-1000 | Lower = faster |

## Performance Guide

⚡ **Fastest**: `--world-size 64 --agents 3 --interval 1`  
✓ **Balanced**: `--world-size 128 --agents 5 --interval 50`  
🎥 **Smooth**: `--world-size 64 --agents 3 --interval 100`  
🌍 **Epic**: `--world-size 256 --agents 20 --interval 50`  

## Troubleshooting

**No window?**  
→ Check display, try `export DISPLAY=:0`

**Too fast?**  
→ Use `--interval 200`

**Too slow?**  
→ Use `--interval 1` or smaller world

**Can't see agents?**  
→ Fewer agents: `--agents 2`

## Stop & Stats

Press `Ctrl+C` to see:
- Total steps run
- Total food gathered

Example:
```
Demo stopped at step 2543
Total food gathered: 5821
```

## That's It! 

Launch and watch:
```bash
./start_live_demo.sh
```

Enjoy! 🎬🐜✨
