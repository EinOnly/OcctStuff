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
echo "║       2D Pixel World - Live Demo (Infinite Loop)           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting simulation..."
echo "Press Ctrl+C to stop"
echo ""

python run/live_demo.py "$@"

echo ""
echo "Demo stopped. Thanks for watching!"
