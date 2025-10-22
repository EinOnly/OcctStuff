#!/bin/bash

# 快速启动脚本

set -e

echo "========================================="
echo "  LLM Agent Evolution System"
echo "========================================="
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# 安装依赖
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 检查模型
echo ""
echo "Checking model..."

if [ ! -f "models/smollm-135m-instruct.q4_k_m.gguf" ]; then
    echo "⚠️  Model not found"
    echo ""
    echo "Please download a model first:"
    echo "  1. mkdir -p models"
    echo "  2. cd models"
    echo "  3. pip install huggingface-hub"
    echo "  4. huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF smollm-135m-instruct.q4_k_m.gguf --local-dir ."
    echo ""
    echo "Or see DOWNLOAD_MODEL.md for more options"
    echo ""
    exit 1
else
    echo "✓ Model found"
fi

# 检查配置
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env from template..."
    cp .env.example .env
    echo "✓ .env created (please review settings)"
fi

# 运行
echo ""
echo "========================================="
echo "  Starting simulation..."
echo "========================================="
echo ""

python main.py

echo ""
echo "Done!"
