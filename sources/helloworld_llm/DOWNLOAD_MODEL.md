# 模型下载指南

## 推荐模型

为了实现20个agent同时运行且总大小<100MB，推荐以下微型模型：

### 1. SmolLM-135M (推荐)

**大小**: ~70MB (量化后)  
**参数**: 135M  
**性能**: 针对小模型优化，推理快

```bash
# 下载 GGUF 量化版本
mkdir -p models
cd models

# 使用 huggingface-cli 下载
pip install huggingface-hub
huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF smollm-135m-instruct.q4_k_m.gguf --local-dir .

# 或者手动下载
# https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct-GGUF/tree/main
```

### 2. TinyStories-33M (最小)

**大小**: ~20MB  
**参数**: 33M  
**性能**: 极小极快，但能力有限

```bash
cd models

# 下载并转换为GGUF
git clone https://huggingface.co/roneneldan/TinyStories-33M
cd TinyStories-33M

# 使用 llama.cpp 转换
# (需要先安装 llama.cpp)
python convert.py TinyStories-33M --outtype q4_0
```

### 3. Phi-2 (350M) - 如果内存充足

**大小**: ~200MB (量化后)  
**参数**: 350M  
**性能**: 能力更强，但稍大

```bash
cd models
huggingface-cli download microsoft/phi-2 --local-dir phi-2
```

## 配置

下载完模型后，修改 `.env` 文件：

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env
MODEL_PATH=models/smollm-135m-instruct.q4_k_m.gguf
MODEL_TYPE=gguf
```

## 验证安装

```bash
# 测试模型加载
python -c "from core.model_loader import LocalModelLoader; loader = LocalModelLoader(); loader.load(); print('✓ Model loaded successfully')"
```

## 内存占用估算

| 模型 | Base Size | LoRA (20个) | Total |
|-----|----------|------------|-------|
| SmolLM-135M | 70MB | 20MB | ~90MB |
| TinyStories-33M | 20MB | 20MB | ~40MB |
| Phi-2 | 200MB | 20MB | ~220MB |

## 使用 MLX (Apple Silicon)

如果在 M1/M2/M3 Mac 上，可以使用 MLX 获得更快速度：

```bash
pip install mlx mlx-lm

# 下载 MLX 格式模型
mlx_lm.convert --hf-path HuggingFaceTB/SmolLM-135M-Instruct --mlx-path models/smollm-135m-mlx

# 修改 .env
MODEL_PATH=models/smollm-135m-mlx
MODEL_TYPE=mlx
```

## 故障排除

### 1. 下载速度慢

使用镜像站：
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download ...
```

### 2. 内存不足

使用更小的模型或增加量化：
```bash
# Q4 = 4-bit 量化 (~70MB)
# Q2 = 2-bit 量化 (~35MB, 但精度下降)
```

### 3. CPU 推理太慢

考虑：
1. 使用 MLX (Apple Silicon)
2. 减少 MAX_TOKENS (在 .env 中设置为 20-30)
3. 批量推理（代码已支持）

## 快速开始

```bash
# 1. 下载推荐模型
mkdir -p models
cd models
huggingface-cli download HuggingFaceTB/SmolLM-135M-Instruct-GGUF smollm-135m-instruct.q4_k_m.gguf --local-dir .
cd ..

# 2. 配置
cp .env.example .env
# 编辑 .env，确保 MODEL_PATH 正确

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行
python main.py
```
