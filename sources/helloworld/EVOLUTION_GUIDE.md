# 🧬 Evolution Demo - Agents with Natural Selection

## 🎯 What's New?

This is a **completely redesigned demo** with real evolution! Unlike the simple live demo, this version features:

### ✨ Key Features

1. **Energy System** ⚡
   - Agents have energy that depletes over time
   - Must collect and eat food to survive
   - Different actions cost different amounts of energy
   - Low energy = death

2. **Natural Selection** 💀
   - Agents die when energy reaches zero
   - Only the fittest survive
   - Population dynamically changes

3. **Reproduction System** 👶
   - Top-performing agents can reproduce every 200 steps
   - Offspring inherit parent's "genome" (behavioral traits)
   - Requires energy to reproduce
   - Must have delivered food to qualify

4. **Genetic Evolution** 🧬
   - Each agent has 5 genetic traits:
     - **Exploration Rate**: How much they wander vs follow trails
     - **Food Attraction**: How strongly drawn to food
     - **Home Attraction**: How strongly drawn to home
     - **Pheromone Sensitivity**: How well they follow trails
     - **Energy Efficiency**: How efficiently they use energy
   - Traits mutate in offspring (15% variance)
   - Better strategies emerge over generations

5. **Fitness Tracking** 📊
   - Fitness = Food Delivered × 100 + Food Collected × 50 + Age × 0.1 - Distance × 0.01
   - Only fit agents reproduce
   - Population self-optimizes

## 🚀 Quick Start

```bash
./start_evolution_demo.sh
```

Or with custom settings:
```bash
python run/evolution_demo.py --agents 10 --interval 10
```

## 📊 What You'll See

### Main Window Layout (6 Panels)

**Top Left - World View (Large)**
- Same visualization as before
- But now agents DIE when energy runs out
- Population changes dynamically

**Top Middle - Evolution Status**
- Current step and generation
- Population count (changes over time!)
- Total births and deaths
- Average energy, fitness, age
- Top 3 agents (best performers)

**Top Right - Genome Statistics**
- Average genetic traits of population
- Shows how population is evolving
- Generation diversity (which generations are alive)

**Bottom Left - Population Chart**
- Population over time
- See extinctions, booms, crashes

**Bottom Middle - Average Fitness Chart**
- Fitness improving over time
- Shows natural selection working

**Bottom Right - Average Energy Chart**
- Energy levels of population
- See if they're surviving well

## 🧬 Genetic Traits Explained

Each agent has 5 genes that control behavior:

### 1. Exploration Rate (0.05 - 0.95)
- **Low**: Follows pheromone trails religiously
- **High**: Explores randomly, ignores trails
- **Optimal**: ~0.2-0.4 (some exploration, mostly following)

### 2. Food Attraction (0.1 - 3.0)
- **Low**: Not very motivated by food
- **High**: Strongly drawn to food sources
- **Optimal**: ~1.5-2.5 (strong motivation)

### 3. Home Attraction (0.1 - 3.0)
- **Low**: Doesn't prioritize returning home
- **High**: Rushes back to home base
- **Optimal**: ~1.5-2.5 (balanced return)

### 4. Pheromone Sensitivity (0.1 - 2.0)
- **Low**: Mostly ignores pheromone trails
- **High**: Follows trails very closely
- **Optimal**: ~0.8-1.5 (good trail following)

### 5. Energy Efficiency (0.5 - 2.0)
- **Low**: Burns energy quickly (dies fast)
- **High**: Very efficient, long-lasting
- **Optimal**: ~1.2-1.8 (efficient survival)

## 🎮 How Evolution Works

### Every Step:
1. Each agent acts (costs energy)
2. Agents with energy <= 0 die
3. Dead agents removed from population
4. World updates (pheromones diffuse)

### Every 100 Steps:
- New food patches spawn (keep environment rich)

### Every 200 Steps (Reproduction Phase):
1. Agents sorted by fitness
2. Top 33% qualify as parents
3. Up to 5 offspring created from top performers
4. Parents lose 20 energy to reproduce
5. Offspring get mutated genes (±15% variation)
6. Generation counter increases

### Population Control:
- **Extinction**: If population = 0, restart with 5 new random agents
- **Overpopulation**: If population > 50, only keep top 50 (by fitness)

## 📈 What to Expect

### Early Stages (Steps 0-500)
- Random initial population
- High death rate as weak agents die
- Population fluctuates wildly
- Low average fitness

### Middle Stages (Steps 500-2000)
- Survivors have decent genes
- Population stabilizes somewhat
- Fitness gradually improves
- Clear trait patterns emerge

### Late Stages (Steps 2000+)
- Highly optimized population
- Most agents are descendants of early winners
- High fitness, good energy management
- Emergent efficient foraging strategies
- Clear behavioral patterns

## 🎯 Success Indicators

You'll know evolution is working when you see:

✅ **Population Stabilizes**: Not crashing to zero repeatedly  
✅ **Fitness Increases**: Average fitness trending upward  
✅ **Energy Improves**: Agents maintaining good energy levels  
✅ **Generation Diversity**: Mix of generations, not just Gen 0  
✅ **Trait Convergence**: Genome stats settling on optimal values  
✅ **Efficient Behavior**: Agents forming effective trail networks

## ⚙️ Command Line Options

```bash
python run/evolution_demo.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--agents` | 10 | Initial population size |
| `--world-size` | 128 | World grid size (N×N) |
| `--seed` | 42 | Random seed for reproducibility |
| `--interval` | 10 | Animation speed (ms per frame, lower = faster) |

### Examples

**Fast evolution (very quick animation):**
```bash
python run/evolution_demo.py --interval 1
```

**Large initial population:**
```bash
python run/evolution_demo.py --agents 20
```

**Small world (easier to see detail):**
```bash
python run/evolution_demo.py --world-size 64 --agents 8
```

**Different random outcomes:**
```bash
python run/evolution_demo.py --seed 12345
```

## 🔬 Experimental Ideas

Try these to see interesting evolution:

### Harsh Environment
```bash
# Start with fewer agents, they must adapt quickly
python run/evolution_demo.py --agents 5
```

### Large Scale
```bash
# Big world, big population, complex evolution
python run/evolution_demo.py --agents 30 --world-size 256
```

### Speed Run
```bash
# Watch 10,000 steps of evolution in minutes
python run/evolution_demo.py --interval 1
```

## 📊 Understanding the Charts

### Population Chart (Bottom Left)
- **Spikes Up**: Reproduction event successful
- **Drops**: Multiple agents died
- **Stable**: Births ≈ Deaths (sustainable)
- **Crashes to Zero**: Extinction event

### Fitness Chart (Bottom Middle)
- **Trending Up**: Evolution working! ✅
- **Flat**: Population stagnant
- **Drops**: Mass die-off or inferior generation

### Energy Chart (Bottom Right)
- **High (>70)**: Population healthy
- **Medium (40-70)**: Population struggling
- **Low (<40)**: Danger! Many about to die

## 🐛 Known Behaviors

### Population Cycles
You may see boom-bust cycles:
1. Population grows (reproduction)
2. Food becomes scarce
3. Mass die-off
4. Survivors thrive in empty world
5. Repeat

This is **normal and realistic**! It's predator-prey dynamics.

### Extinction Events
Sometimes entire population dies. The system will:
1. Detect extinction (pop = 0)
2. Auto-restart with 5 random agents
3. Evolution continues from scratch

### Convergent Evolution
After many generations, most agents will have similar "optimal" genes. This shows evolution working correctly!

## 🎓 Educational Value

This demo teaches:
- **Natural Selection**: Fitness-based survival
- **Genetic Algorithms**: Mutation and selection
- **Emergent Behavior**: Complex strategies from simple rules
- **Population Dynamics**: Boom-bust cycles
- **Optimization**: Systems self-improving
- **Survival Strategies**: Energy management

## 🆚 Comparison with Simple Live Demo

| Feature | Simple Live Demo | Evolution Demo |
|---------|-----------------|----------------|
| Agents | Fixed number | Dynamic (births/deaths) |
| Behavior | Random/fixed | Evolving strategies |
| Survival | Infinite | Energy-dependent |
| Reproduction | No | Yes (fitness-based) |
| Genetics | No | Yes (5 traits) |
| Selection | No | Yes (natural) |
| Charts | No | 3 real-time charts |
| Learning | No | Yes (population improves) |

## 🎬 What Makes It Interesting

Unlike the simple demo where agents just wander randomly:

1. **Stakes**: Agents can DIE. There's real survival pressure
2. **Progress**: Population gets BETTER over time
3. **Drama**: Watch extinctions, booms, competition
4. **Emergence**: Complex strategies emerge naturally
5. **Unpredictable**: Each run is different
6. **Scientific**: See actual evolutionary algorithms at work

## 🚀 Try It Now!

```bash
./start_evolution_demo.sh
```

Watch your population evolve from random agents to efficient foragers!

Press `Ctrl+C` when done to see final statistics.

## 📝 Final Notes

- **Be Patient**: Evolution takes time (500-1000 steps to see clear improvement)
- **Watch the Charts**: They tell the story of your population
- **Try Different Seeds**: Each seed produces different evolutionary paths
- **Look for Patterns**: Optimal genomes tend to converge
- **Enjoy the Drama**: Extinctions, booms, and success stories!

## 🎉 Enjoy Natural Selection!

This is **real artificial life** - agents competing, dying, reproducing, and evolving. Much more interesting than simple random movement!

Start watching evolution now:
```bash
./start_evolution_demo.sh
```

🧬 Let natural selection do its magic! 🐜✨
