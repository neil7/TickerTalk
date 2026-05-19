"""
Base Agent Class
Provides common functionality for all specialized agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import os
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import ollama_config, api_config, gemini_config, groq_config, anthropic_config


class BaseAgent(ABC):
    """Base class for all analysis agents"""
    
    def __init__(self, name: str):
        self.name = name
        
        if api_config.llm_provider == "gemini":
            if not api_config.google_api_key:
                print("⚠️  WARNING: Google API Key not found. Please set GOOGLE_API_KEY in .env")
                print("   Falling back to Ollama...")
                self._init_ollama()
            else:
                self.llm = ChatGoogleGenerativeAI(
                    model=gemini_config.model,
                    google_api_key=api_config.google_api_key,
                    temperature=gemini_config.temperature,
                    convert_system_message_to_human=True
                )
        elif api_config.llm_provider == "groq":
            if not api_config.groq_api_key:
                print("⚠️  WARNING: Groq API Key not found. Please set GROQ_API_KEY in .env")
                print("   Falling back to Ollama...")
                self._init_ollama()
            else:
                try:
                    from langchain_groq import ChatGroq
                    self.llm = ChatGroq(
                        model=groq_config.model,
                        api_key=api_config.groq_api_key,
                        temperature=groq_config.temperature
                    )
                except ImportError:
                    print("⚠️  WARNING: langchain-groq not installed.")
                    print("   Please run: pip install langchain-groq")
                    print("   Falling back to Ollama...")
                    self._init_ollama()
        elif api_config.llm_provider == "anthropic":
            if not api_config.anthropic_api_key:
                print("⚠️  WARNING: Anthropic API Key not found. Please set ANTHROPIC_API_KEY in .env")
                print("   Falling back to Ollama...")
                self._init_ollama()
            else:
                try:
                    from langchain_anthropic import ChatAnthropic
                    self.llm = ChatAnthropic(
                        model=anthropic_config.model,
                        api_key=api_config.anthropic_api_key,
                        temperature=anthropic_config.temperature
                    )
                except ImportError:
                    print("⚠️  WARNING: langchain-anthropic not installed.")
                    print("   Please run: pip install langchain-anthropic")
                    print("   Falling back to Ollama...")
                    self._init_ollama()
        else:
            self._init_ollama()

    def _init_ollama(self):
        self.llm = OllamaLLM(
            model=ollama_config.model,
            base_url=ollama_config.base_url,
            temperature=ollama_config.temperature,
            num_ctx=ollama_config.num_ctx,
        )
    
    @abstractmethod
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis and return updated state"""
        pass
    
    def _create_prompt(self, template: str) -> ChatPromptTemplate:
        """Create a prompt template"""
        return ChatPromptTemplate.from_template(template)
    
    def _invoke_llm(self, prompt: ChatPromptTemplate, **kwargs) -> str:
        """Invoke the LLM with given prompt and parameters"""
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(kwargs)
    
    def _format_dict_for_prompt(self, data: Dict, max_items: int = 20) -> str:
        """Format dictionary data for LLM prompt"""
        lines = []
        count = 0
        
        for key, value in data.items():
            if count >= max_items:
                lines.append("... (additional data truncated)")
                break
            
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in list(value.items())[:5]:
                    lines.append(f"  - {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"{key}: {len(value)} items")
            else:
                lines.append(f"{key}: {value}")
            
            count += 1
        
        return "\n".join(lines)
