"""
LLM Configuration for Financial Analysis
Best open-source models for trading and multi-agent systems
"""
from dataclasses import dataclass
from typing import Dict, Any
import os

# Recommended Models for Financial Trading Analysis
# Ordered by capability (best first)

RECOMMENDED_MODELS = {
    # TIER 1: Best for complex financial reasoning (requires good GPU)
    "deepseek-r1-distill-llama-70b": {
        "description": "DeepSeek R1 Distill (Groq) - SOTA reasoning model, great for complex analysis",
        "context": 128000,
        "speed": "fast",
        "quality": "excellent",
        "ram_required": "N/A (Cloud)",
    },
    "llama-3.3-70b-versatile": {
        "description": "Llama 3.3 70B (Groq) - Latest Llama model, improved reasoning over 3.1",
        "context": 128000,
        "speed": "fast",
        "quality": "excellent",
        "ram_required": "N/A (Cloud)",
    },
    "qwen2.5:32b": {
        "description": "Qwen 2.5 32B - Excellent numerical reasoning, best for financial analysis",
        "context": 32768,
        "speed": "slow",
        "quality": "excellent",
        "ram_required": "24GB+",
    },
    "llama3.1:70b": {
        "description": "Llama 3.1 70B - Best overall reasoning and instruction following",
        "context": 131072,
        "speed": "slow",
        "quality": "excellent", 
        "ram_required": "48GB+",
    },
    "mixtral:8x7b": {
        "description": "Mixtral 8x7B MoE - Great balance of speed and quality",
        "context": 32768,
        "speed": "medium",
        "quality": "very_good",
        "ram_required": "32GB+",
    },
    
    # TIER 2: Good balance for most users
    "qwen2.5:14b": {
        "description": "Qwen 2.5 14B - Strong numerical/financial reasoning",
        "context": 32768,
        "speed": "medium",
        "quality": "very_good",
        "ram_required": "12GB+",
    },
    "llama3.2:latest": {
        "description": "Llama 3.2 - Good general purpose, fast",
        "context": 131072,
        "speed": "fast",
        "quality": "good",
        "ram_required": "8GB+",
    },
    "deepseek-coder-v2:16b": {
        "description": "DeepSeek V2 - Excellent for analytical/structured tasks",
        "context": 65536,
        "speed": "medium",
        "quality": "very_good",
        "ram_required": "12GB+",
    },
    
    # TIER 3: Fast models for quick decisions
    "mistral:7b": {
        "description": "Mistral 7B - Fast and capable",
        "context": 32768,
        "speed": "fast",
        "quality": "good",
        "ram_required": "6GB+",
    },
    "phi3:14b": {
        "description": "Phi-3 14B - Microsoft's efficient model",
        "context": 4096,
        "speed": "fast",
        "quality": "good",
        "ram_required": "10GB+",
    },
    "gemma2:9b": {
        "description": "Gemma 2 9B - Google's efficient model",
        "context": 8192,
        "speed": "fast",
        "quality": "good",
        "ram_required": "8GB+",
    },
    
    # CLOUD MODELS (API Keys required)
    "gemini-1.5-flash": {
        "description": "Google Gemini 1.5 Flash - Fast, cheap, and very capable (2M context)",
        "context": 2000000,
        "speed": "very_fast",
        "quality": "excellent",
        "ram_required": "N/A (Cloud)",
    },
    "gemini-2.0-flash-exp": {
        "description": "Google Gemini 2.0 Flash (Preview) - Next gen speed and intelligence",
        "context": 1000000,
        "speed": "very_fast",
        "quality": "excellent",
        "ram_required": "N/A (Cloud)",
    },
    "llama-3.1-70b-versatile": {
        "description": "Groq Llama 3.1 70B - Extremely fast inference, high quality",
        "context": 32768,
        "speed": "extreme",
        "quality": "excellent",
        "ram_required": "N/A (Cloud/Free)",
    },
    "llama-3.1-8b-instant": {
        "description": "Groq Llama 3.1 8B - Instant inference speed",
        "context": 32768,
        "speed": "extreme",
        "quality": "good",
        "ram_required": "N/A (Cloud/Free)",
    },
    "claude-sonnet-4-6": {
        "description": "Anthropic Claude Sonnet 4.6 - Best combination of speed and intelligence for financial analysis",
        "context": 200000,
        "speed": "fast",
        "quality": "excellent",
        "ram_required": "N/A (Cloud)",
    },
    "claude-3-5-sonnet-20241022": {
        "description": "Anthropic Claude 3.5 Sonnet - Strong reasoning, great for analysis",
        "context": 200000,
        "speed": "fast",
        "quality": "excellent",
        "ram_required": "N/A (Cloud)",
    },
}

# Model recommendations by use case
USE_CASE_RECOMMENDATIONS = {
    "day_trading": {
        "primary": "deepseek-r1-distill-llama-70b",  # Best for deep reasoning
        "fallback": "llama-3.3-70b-versatile",
        "reason": "DeepSeek R1 Distill provides superior reasoning for complex market analysis"
    },
    "swing_trading": {
        "primary": "llama-3.3-70b-versatile",
        "fallback": "llama-3.1-70b-versatile",
        "reason": "Excellent balance of reasoning and reliability"
    },
    "multi_agent": {
        "primary": "llama-3.3-70b-versatile",
        "fallback": "llama3.2:latest",
        "reason": "Strong instruction following for agent coordination"
    },
    "sentiment_analysis": {
        "primary": "llama-3.3-70b-versatile",
        "fallback": "mistral:7b",
        "reason": "Best for understanding nuanced news and social sentiment"
    },
    "technical_analysis": {
        "primary": "deepseek-r1-distill-llama-70b",
        "fallback": "llama-3.3-70b-versatile",
        "reason": "Provides deep technical analysis with fewer hallucinations"
    },
}


@dataclass 
class ModelConfig:
    """Configuration for the selected model"""
    name: str
    temperature: float = 0.1  # Low temp for consistent financial analysis
    num_ctx: int = 8192
    top_p: float = 0.9
    repeat_penalty: float = 1.1


def get_recommended_model(use_case: str = "day_trading") -> str:
    """Get the recommended model for a specific use case"""
    if use_case in USE_CASE_RECOMMENDATIONS:
        return USE_CASE_RECOMMENDATIONS[use_case]["primary"]
    return "llama-3.3-70b-versatile"  # Default recommendation


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a specific model"""
    return RECOMMENDED_MODELS.get(model_name, {
        "description": "Unknown model",
        "context": 4096,
        "speed": "unknown",
        "quality": "unknown",
    })


def print_model_recommendations():
    """Print model recommendations for user"""
    print("\n" + "="*60)
    print("🤖 RECOMMENDED MODELS FOR TRADING ANALYSIS")
    print("="*60)
    
    print("\n☁️  CLOUD MODELS (Recommended for best performance):")
    print("   claude-sonnet-4-6              # Anthropic: Best speed/intelligence balance")
    print("   deepseek-r1-distill-llama-70b    # Groq: SOTA Reasoning (Preview)")
    print("   llama-3.3-70b-versatile          # Groq: Latest Llama 3.3")
    print("   gemini-2.0-flash-exp             # Google: Fast & Smart (Free Preview)")
    
    print("\n🏠 LOCAL MODELS (Best for privacy/offline):")
    print("   ollama pull llama3.2:latest    # RECOMMENDED - No refusals")
    
    print("\n💡 SETUP INSTRUCTIONS:")
    print("   For Anthropic: Get API key from console.anthropic.com")
    print("                   Add to .env: ANTHROPIC_API_KEY=sk-ant-...")
    print("                   Add to .env: LLM_PROVIDER=anthropic")
    print("   For Groq: Get API key from console.groq.com")
    print("             Add to .env: GROQ_API_KEY=your_key_here")
    print("             Add to .env: LLM_PROVIDER=groq")
    
    print("\n📊 CURRENT MODEL:")
    from config import ollama_config, api_config, gemini_config, groq_config, anthropic_config
    if api_config.llm_provider == "gemini":
        print(f"   Provider: Google Gemini")
        print(f"   Model: {gemini_config.model}")
    elif api_config.llm_provider == "groq":
        print(f"   Provider: Groq (Ultra-Fast)")
        print(f"   Model: {groq_config.model}")
    elif api_config.llm_provider == "anthropic":
        print(f"   Provider: Anthropic Claude")
        print(f"   Model: {anthropic_config.model}")
    else:
        print(f"   Provider: Ollama (Local)")
        print(f"   Model: {ollama_config.model}")
    print("="*60 + "\n")
