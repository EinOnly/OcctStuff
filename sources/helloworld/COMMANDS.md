# 🎬 Live Demo - All Commands

## Launch Commands

### 1. Easiest (Recommended) ⭐
```bash
cd /Users/ein/EinDev/OcctStuff/sources/helloworld
./start_live_demo.sh
```

### 2. Standard Launch
```bash
cd /Users/ein/EinDev/OcctStuff/sources/helloworld
source /Users/ein/EinDev/OcctStuff/.venv/bin/activate
python run/live_demo.py
```

### 3. With All Default Options (Explicit)
```bash
python run/live_demo.py --agents 5 --world-size 128 --seed 42 --interval 50
```

## Pre-configured Setups

### Fastest Animation (1ms updates)
```bash
python run/live_demo.py --interval 1
```

### Slow Motion (500ms updates)
```bash
python run/live_demo.py --interval 500
```

### Small World (64x64, easier to see details)
```bash
python run/live_demo.py --world-size 64 --agents 3
```

### Large World (256x256, epic scale)
```bash
python run/live_demo.py --world-size 256 --agents 20 --interval 25
```

### Many Agents (busy simulation)
```bash
python run/live_demo.py --agents 15 --interval 25
```

### Few Agents (easy to track)
```bash
python run/live_demo.py --agents 2 --interval 100
```

### Different Random Seed
```bash
python run/live_demo.py --seed 12345
```

## Testing Commands

### Run Live Simulation Test (Headless, No GUI)
```bash
cd /Users/ein/EinDev/OcctStuff/sources/helloworld
source /Users/ein/EinDev/OcctStuff/.venv/bin/activate
python tests/test_live.py
```

### Run All Tests
```bash
python tests/test_world.py
python tests/test_bf.py
python tests/test_live.py
```

### Run Original Demos (Fixed Steps)
```bash
# Foraging demo
python run/sim.py --demo foraging --agents 5 --steps 1000 --seed 42

# Brainf*ck demo
python run/sim.py --demo bf --steps 2000 --seed 42
```

## Stop Command

### Stop Live Demo
```
Press: Ctrl+C
```

Output will show:
```
Demo stopped at step XXXX
Total food gathered: YYYY
```

## Help Command

### Show All Options
```bash
python run/live_demo.py --help
```

Output:
```
usage: live_demo.py [-h] [--agents AGENTS] [--world-size WORLD_SIZE] 
                    [--seed SEED] [--interval INTERVAL]

Live 2D World Simulation - Infinite Loop Demo

optional arguments:
  -h, --help            show this help message and exit
  --agents AGENTS       Number of agents (default: 5)
  --world-size WORLD_SIZE
                        World size (default: 128)
  --seed SEED           Random seed (default: 42)
  --interval INTERVAL   Animation interval in ms (default: 50)
```

## Quick Scenarios

### Scenario 1: Demo for Presentation
```bash
python run/live_demo.py --agents 5 --interval 50
```

### Scenario 2: Debug Agent Behavior
```bash
python run/live_demo.py --agents 1 --world-size 64 --interval 200
```

### Scenario 3: Performance Test
```bash
python run/live_demo.py --agents 30 --world-size 256 --interval 1
```

### Scenario 4: Smooth Visualization
```bash
python run/live_demo.py --agents 3 --world-size 64 --interval 100
```

### Scenario 5: Reproduce Specific Behavior
```bash
python run/live_demo.py --seed 777
```

## Documentation Commands

### View Quick Reference
```bash
cat QUICKSTART_LIVE.md
```

### View Full Guide
```bash
less LIVE_DEMO_GUIDE.md
# or
open LIVE_DEMO_GUIDE.md
```

### View Implementation Summary
```bash
cat IMPLEMENTATION_SUMMARY.md
```

### View Main README
```bash
cat README.md
```

## Setup Commands (First Time Only)

### Install Dependencies
```bash
cd /Users/ein/EinDev/OcctStuff/sources/helloworld
source /Users/ein/EinDev/OcctStuff/.venv/bin/activate
pip install -r requirements.txt
```

### Make Launch Script Executable
```bash
chmod +x start_live_demo.sh
```

## Environment Commands

### Activate Virtual Environment
```bash
source /Users/ein/EinDev/OcctStuff/.venv/bin/activate
```

### Check Python Version
```bash
python --version
# Should be 3.10 or higher
```

### Check Installed Packages
```bash
pip list | grep -E "(numpy|matplotlib|pytest)"
```

## File Listing Commands

### See All Project Files
```bash
ls -la
```

### See Demo Scripts
```bash
ls -la run/
```

### See Tests
```bash
ls -la tests/
```

### See Documentation
```bash
ls -la *.md
```

## Useful One-Liners

### Launch and Time It
```bash
time ./start_live_demo.sh
```

### Launch in Background (Not Recommended)
```bash
python run/live_demo.py > /dev/null 2>&1 &
```

### Launch with Logging
```bash
python run/live_demo.py 2>&1 | tee live_demo.log
```

### Check if Demo is Running
```bash
ps aux | grep live_demo.py
```

### Kill Demo if Stuck
```bash
pkill -f live_demo.py
```

## Copy-Paste Ready Commands

**Standard demo:**
```bash
cd /Users/ein/EinDev/OcctStuff/sources/helloworld && ./start_live_demo.sh
```

**Fast demo:**
```bash
cd /Users/ein/EinDev/OcctStuff/sources/helloworld && source /Users/ein/EinDev/OcctStuff/.venv/bin/activate && python run/live_demo.py --interval 1
```

**Small world demo:**
```bash
cd /Users/ein/EinDev/OcctStuff/sources/helloworld && source /Users/ein/EinDev/OcctStuff/.venv/bin/activate && python run/live_demo.py --world-size 64 --agents 3
```

**Run all tests:**
```bash
cd /Users/ein/EinDev/OcctStuff/sources/helloworld && source /Users/ein/EinDev/OcctStuff/.venv/bin/activate && python tests/test_world.py && python tests/test_bf.py && python tests/test_live.py
```

## That's Everything!

Pick a command and start watching your agents! 🎬🐜

Most common:
```bash
./start_live_demo.sh
```
