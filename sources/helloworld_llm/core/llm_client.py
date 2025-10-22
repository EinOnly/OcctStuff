"""
LLM API 客户端
"""
import os
from openai import OpenAI
from config import Config


class LLMClient:
    """OpenAI API 客户端封装"""
    
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL
        self.max_tokens = Config.OPENAI_MAX_TOKENS
        self.temperature = Config.OPENAI_TEMPERATURE
        
        # 统计
        self.total_calls = 0
        self.total_tokens = 0
    
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """发送聊天请求"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            # 更新统计
            self.total_calls += 1
            if hasattr(response, 'usage'):
                self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"❌ LLM API Error: {e}")
            return "FORWARD"  # 默认动作
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimate_cost(),
        }
    
    def estimate_cost(self) -> float:
        """估算成本（USD）"""
        # GPT-4o-mini 价格：$0.15/1M input tokens, $0.60/1M output tokens
        # 简化计算，假设 input:output = 3:1
        input_tokens = self.total_tokens * 0.75
        output_tokens = self.total_tokens * 0.25
        
        cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens * 0.60 / 1_000_000)
        return cost
