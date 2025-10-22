#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="/Users/ein/EinDev/OcctStuff/.venv"

cd "$SCRIPT_DIR"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "⚠ Virtual environment not found at: $VENV_PATH"
    echo "  Using system Python..."
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║    2D Pixel World - EVOLUTION DEMO with Natural Selection  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Features:"
echo "  ✓ Energy system - agents must eat to survive"
echo "  ✓ Natural selection - weak agents die"
echo "  ✓ Reproduction - fit agents create offspring"
echo "  ✓ Genetic evolution - traits mutate over generations"
echo "  ✓ Real-time statistics and charts"
echo ""
echo "Starting simulation..."
echo "Press Ctrl+C to stop"
echo ""

python run/evolution_demo.py "$@"

echo ""
echo "Evolution stopped. Thanks for watching!"
