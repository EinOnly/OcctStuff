#!/usr/bin/env bash

# Advanced Evolution Demo Launcher
# This script runs the advanced evolution simulation with elite inheritance,
# visual body representation, building system, and expandable world.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
DEMO_SCRIPT="$SCRIPT_DIR/run/advanced_evolution_demo.py"

echo "🧬 Starting Advanced Evolution Demo..."
echo "📁 Project root: $PROJECT_ROOT"
echo "🐍 Python: $VENV_PYTHON"
echo ""

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PYTHON"
    echo "Please activate your virtual environment first."
    exit 1
fi

if [ ! -f "$DEMO_SCRIPT" ]; then
    echo "❌ Error: Demo script not found at $DEMO_SCRIPT"
    exit 1
fi

echo "🚀 Launching simulation..."
echo "   - Elite genome inheritance"
echo "   - Visual body representation"
echo "   - Building system (nests, storage, trails)"
echo "   - Expandable world (64x64 → 256x256)"
echo ""

"$VENV_PYTHON" "$DEMO_SCRIPT"

echo ""
echo "✅ Demo finished."
