"""
本地模型加载器 - 支持多种后端
"""
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys

from config import Config


class LocalModelLoader:
    """统一的本地模型加载接口"""
    
    def __init__(self, model_path: Optional[str] = None, model_type: Optional[str] = None):
        self.model_path = Path(model_path or Config.MODEL_PATH)
        self.model_type = model_type or Config.MODEL_TYPE
        self.model = None
        self.backend = None
        
    def load(self):
        """根据模型类型加载对应的后端"""
        if self.model_type == "mock":
            return self._load_mock()
        elif self.model_type == "gguf":
            return self._load_llama_cpp()
        elif self.model_type == "mlx":
            return self._load_mlx()
        elif self.model_type == "pytorch" or self.model_type == "transformers":
            return self._load_pytorch()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_mock(self):
        """加载Mock模型"""
        from core.mock_model import MockTinyLLM
        print("Loading Mock Model...")
        self.model = MockTinyLLM()
        self.backend = "mock"
        print(f"✓ Mock Model loaded")
        return self
    
    def _load_llama_cpp(self):
        """使用 llama.cpp 加载 GGUF 模型"""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with:\n"
                "pip install llama-cpp-python"
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Download a model (e.g., SmolLM-135M) and place it in models/"
            )
        
        print(f"Loading GGUF model: {self.model_path}")
        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=Config.CONTEXT_SIZE,
            n_threads=4,  # 根据CPU核心数调整
            n_gpu_layers=0,  # CPU only, set >0 for GPU
            verbose=False
        )
        self.backend = "llama.cpp"
        print(f"✓ Model loaded with {self.backend}")
        return self
    
    def _load_mlx(self):
        """使用 MLX 加载模型 (Apple Silicon only)"""
        try:
            import mlx.core as mx
            from mlx_lm import load, generate
        except ImportError:
            raise ImportError(
                "MLX not installed. Install with:\n"
                "pip install mlx mlx-lm"
            )
        
        print(f"Loading MLX model: {self.model_path}")
        self.model = load(str(self.model_path))
        self.backend = "mlx"
        print(f"✓ Model loaded with {self.backend}")
        return self
    
    def _load_pytorch(self):
        """使用 PyTorch + Transformers 加载模型（支持int8量化）"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
        except ImportError:
            raise ImportError(
                "Transformers not installed. Install with:\n"
                "pip install transformers torch"
            )
        
        # 检测可用设备：MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            device = "mps"
            print(f"🚀 Using Apple MPS (GPU acceleration)")
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"🚀 Using CUDA GPU")
        else:
            device = "cpu"
            print(f"⚠️  Using CPU (slow)")
        
        print(f"Loading PyTorch model: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        # 根据设备选择加载策略
        if device == "cuda":
            # CUDA支持int8量化（需要bitsandbytes）
            try:
                print("🔧 Loading with int8 quantization...")
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                print("✓ Int8 quantization enabled")
            except Exception as e:
                print(f"⚠️  Int8 quantization failed: {e}, falling back to float16")
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
        elif device == "mps":
            # MPS使用float32（int8在MPS上支持有限）
            print("🔧 Loading with float32 for MPS...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(device)
        else:
            # CPU使用float32
            print("🔧 Loading with float32 for CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(device)
        
        self.device = device
        self.backend = "pytorch"
        print(f"✓ Model loaded with {self.backend} on {device}")
        return self
    
    def generate(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """统一的生成接口"""
        max_tokens = max_tokens or Config.MAX_TOKENS
        temperature = temperature or Config.TEMPERATURE
        
        if self.backend == "mock":
            return self._generate_mock(prompt, max_tokens, temperature)
        elif self.backend == "llama.cpp":
            return self._generate_llama_cpp(prompt, max_tokens, temperature)
        elif self.backend == "mlx":
            return self._generate_mlx(prompt, max_tokens, temperature)
        elif self.backend == "pytorch":
            return self._generate_pytorch(prompt, max_tokens, temperature)
        else:
            raise RuntimeError("Model not loaded")
    
    def _generate_mock(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Mock模型生成"""
        return self.model.generate(prompt, max_tokens, temperature)
    
    def _generate_llama_cpp(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """llama.cpp 生成"""
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "\n\n"],
            echo=False
        )
        return output["choices"][0]["text"].strip()
    
    def _generate_mlx(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """MLX 生成"""
        from mlx_lm import generate
        response = generate(
            self.model,
            prompt,
            max_tokens=max_tokens,
            temp=temperature
        )
        return response.strip()
    
    def _generate_pytorch(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """PyTorch 生成"""
        import torch
        
        # 将输入移动到模型所在设备
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 去掉 prompt 部分
        response = response[len(prompt):].strip()
        return response


class SharedModelPool:
    """共享模型池 - 20个agent共享同一个base model"""
    
    _instance = None
    _base_model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_base_model(self) -> LocalModelLoader:
        """获取共享的base model"""
        if self._base_model is None:
            print("Initializing shared base model...")
            self._base_model = LocalModelLoader().load()
        return self._base_model
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """批量生成（减少overhead）"""
        model = self.get_base_model()
        results = []
        for prompt in prompts:
            result = model.generate(prompt)
            results.append(result)
        return results


if __name__ == "__main__":
    # 测试加载
    print("Testing model loader...")
    loader = LocalModelLoader()
    try:
        loader.load()
        
        # 测试生成
        test_prompt = "You are a tiny agent. Current state: at position (5,3), energy 80, see food north. Action:"
        response = loader.generate(test_prompt, max_tokens=20)
        print(f"\nTest prompt: {test_prompt}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\n提示：请先下载模型文件")
        print("推荐模型：")
        print("1. SmolLM-135M-GGUF: https://huggingface.co/HuggingFaceTB/SmolLM-135M-GGUF")
        print("2. TinyStories-33M: https://huggingface.co/roneneldan/TinyStories-33M")
