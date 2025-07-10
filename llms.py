"""
CareerMind AI - Multi-Agent GenAI Career Assistant
Enhanced LLM Management with Multiple Providers and Advanced Configuration
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# LLM imports
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

# Configuration and monitoring
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Enum for supported model providers"""
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


class ModelTier(Enum):
    """Enum for model performance tiers"""
    PREMIUM = "premium"
    STANDARD = "standard"
    FAST = "fast"
    BUDGET = "budget"


@dataclass
class ModelConfig:
    """Configuration class for LLM models"""
    name: str
    provider: ModelProvider
    tier: ModelTier
    max_tokens: int
    supports_streaming: bool
    supports_functions: bool
    cost_per_1k_tokens: float
    context_window: int
    description: str


class EnhancedLLMManager:
    """
    Enhanced LLM manager with support for multiple providers and intelligent model selection.
    """
    
    def __init__(self):
        """Initialize the enhanced LLM manager"""
        self.models_config = self._initialize_model_configs()
        self.usage_stats = {}
        self.model_cache = {}
        
        logger.info("Enhanced LLM Manager initialized")
    
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """
        Initialize configuration for all supported models.
        
        Returns:
            Dict mapping model names to their configurations
        """
        return {
            # OpenAI Models
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                provider=ModelProvider.OPENAI,
                tier=ModelTier.PREMIUM,
                max_tokens=4096,
                supports_streaming=True,
                supports_functions=True,
                cost_per_1k_tokens=0.03,
                context_window=128000,
                description="Most capable OpenAI model with advanced reasoning"
            ),
            "gpt-4o-mini": ModelConfig(
                name="gpt-4o-mini",
                provider=ModelProvider.OPENAI,
                tier=ModelTier.FAST,
                max_tokens=4096,
                supports_streaming=True,
                supports_functions=True,
                cost_per_1k_tokens=0.0015,
                context_window=128000,
                description="Fast and cost-effective OpenAI model"
            ),
            "gpt-4-turbo": ModelConfig(
                name="gpt-4-turbo",
                provider=ModelProvider.OPENAI,
                tier=ModelTier.STANDARD,
                max_tokens=4096,
                supports_streaming=True,
                supports_functions=True,
                cost_per_1k_tokens=0.01,
                context_window=128000,
                description="Balanced performance and cost OpenAI model"
            ),
            
            # Groq Models
            "llama-3.3-70b-versatile": ModelConfig(
                name="llama-3.3-70b-versatile",
                provider=ModelProvider.GROQ,
                tier=ModelTier.PREMIUM,
                max_tokens=8192,
                supports_streaming=True,
                supports_functions=True,
                cost_per_1k_tokens=0.0005,
                context_window=32768,
                description="High-performance Llama model on Groq infrastructure"
            ),
            "mixtral-8x7b-32768": ModelConfig(
                name="mixtral-8x7b-32768",
                provider=ModelProvider.GROQ,
                tier=ModelTier.STANDARD,
                max_tokens=8192,
                supports_streaming=True,
                supports_functions=True,
                cost_per_1k_tokens=0.0003,
                context_window=32768,
                description="Mixtral model optimized for speed and efficiency"
            ),
            
            # Anthropic Models
            "claude-3-opus-20240229": ModelConfig(
                name="claude-3-opus-20240229",
                provider=ModelProvider.ANTHROPIC,
                tier=ModelTier.PREMIUM,
                max_tokens=4096,
                supports_streaming=True,
                supports_functions=True,
                cost_per_1k_tokens=0.015,
                context_window=200000,
                description="Most capable Anthropic model with excellent reasoning"
            ),
            "claude-3-sonnet-20240229": ModelConfig(
                name="claude-3-sonnet-20240229",
                provider=ModelProvider.ANTHROPIC,
                tier=ModelTier.STANDARD,
                max_tokens=4096,
                supports_streaming=True,
                supports_functions=True,
                cost_per_1k_tokens=0.003,
                context_window=200000,
                description="Balanced Anthropic model for most use cases"
            ),

        }
    
    def load_llm(
        self,
        model_name: str = None,
        provider: str = None,
        temperature: float = 0.2,
        max_tokens: int = None,
        streaming: bool = True,
        **kwargs
    ) -> BaseChatModel:
        """
        Enhanced LLM loading with intelligent model selection and configuration.
        
        Args:
            model_name: Specific model name to load
            provider: Provider preference (openai, groq, anthropic, ollama)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            streaming: Enable streaming responses
            **kwargs: Additional model-specific parameters
            
        Returns:
            BaseChatModel: Configured language model
        """
        try:
            # Intelligent model selection if not specified
            if not model_name:
                model_name = self._select_optimal_model(provider)
            
            # Get model configuration
            config = self.models_config.get(model_name)
            if not config:
                logger.warning(f"Unknown model {model_name}, falling back to default")
                model_name = "gpt-4o-mini" if provider == "openai" else "llama-3.3-70b-versatile"
                config = self.models_config[model_name]
            
            # Check cache first
            cache_key = self._generate_cache_key(model_name, temperature, max_tokens, streaming)
            if cache_key in self.model_cache:
                logger.info(f"Using cached model: {model_name}")
                return self.model_cache[cache_key]
            
            # Load model based on provider
            llm = self._create_model_instance(config, temperature, max_tokens, streaming, **kwargs)
            
            # Cache the model
            self.model_cache[cache_key] = llm
            
            # Update usage statistics
            self._update_usage_stats(model_name)
            
            logger.info(f"Successfully loaded model: {model_name} ({config.provider.value})")
            return llm
            
        except Exception as e:
            logger.error(f"Error loading LLM {model_name}: {str(e)}")
            # Fallback to basic model
            return self._create_fallback_model(temperature)
    
    def _create_model_instance(
        self,
        config: ModelConfig,
        temperature: float,
        max_tokens: int,
        streaming: bool,
        **kwargs
    ) -> BaseChatModel:
        """
        Create a model instance based on provider configuration.
        
        Args:
            config: Model configuration
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            streaming: Enable streaming
            **kwargs: Additional parameters
            
        Returns:
            BaseChatModel: Configured model instance
        """
        try:
            # Prepare common parameters
            common_params = {
                "temperature": temperature,
                "streaming": streaming,
                **kwargs
            }
            
            if max_tokens:
                common_params["max_tokens"] = min(max_tokens, config.max_tokens)
            
            # Create model based on provider
            if config.provider == ModelProvider.OPENAI:
                return ChatOpenAI(
                    model_name=config.name,
                    openai_api_key=os.environ.get("OPENAI_API_KEY"),
                    **common_params
                )
            
            elif config.provider == ModelProvider.GROQ:
                return ChatGroq(
                    groq_api_key=os.environ.get("GROQ_API_KEY"),
                    model_name=config.name,
                    **common_params
                )
            
            elif config.provider == ModelProvider.ANTHROPIC:
                return ChatAnthropic(
                    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
                    model=config.name,
                    **common_params
                )

            
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
                
        except Exception as e:
            logger.error(f"Error creating model instance: {str(e)}")
            raise
    
    def _select_optimal_model(self, provider_preference: str = None) -> str:
        """
        Intelligently select the optimal model based on availability and preferences.
        
        Args:
            provider_preference: Preferred provider
            
        Returns:
            str: Selected model name
        """
        try:
            # Check API key availability
            available_providers = []
            
            if os.environ.get("OPENAI_API_KEY"):
                available_providers.append(ModelProvider.OPENAI)
            if os.environ.get("GROQ_API_KEY"):
                available_providers.append(ModelProvider.GROQ)
            if os.environ.get("ANTHROPIC_API_KEY"):
                available_providers.append(ModelProvider.ANTHROPIC)

            
            # Select based on preference and availability
            if provider_preference:
                provider_enum = ModelProvider(provider_preference)
                if provider_enum in available_providers:
                    # Return best model for preferred provider
                    return self._get_best_model_for_provider(provider_enum)
            
            # Default selection logic
            if ModelProvider.GROQ in available_providers:
                return "llama-3.3-70b-versatile"  # Fast and efficient
            elif ModelProvider.OPENAI in available_providers:
                return "gpt-4o-mini"  # Cost-effective
            elif ModelProvider.ANTHROPIC in available_providers:
                return "claude-3-sonnet-20240229"  # Balanced
            else:
                return "llama3"  # Local fallback
                
        except Exception as e:
            logger.error(f"Error selecting optimal model: {str(e)}")
            return "llama3"  # Safe fallback
    
    def _get_best_model_for_provider(self, provider: ModelProvider) -> str:
        """
        Get the best available model for a specific provider.
        
        Args:
            provider: Model provider
            
        Returns:
            str: Best model name for the provider
        """
        provider_models = {
            model_name: config for model_name, config in self.models_config.items()
            if config.provider == provider
        }
        
        if not provider_models:
            raise ValueError(f"No models available for provider: {provider}")
        
        # Return premium tier model if available, otherwise standard
        premium_models = [name for name, config in provider_models.items() if config.tier == ModelTier.PREMIUM]
        if premium_models:
            return premium_models[0]
        
        standard_models = [name for name, config in provider_models.items() if config.tier == ModelTier.STANDARD]
        if standard_models:
            return standard_models[0]
        
        # Return any available model
        return list(provider_models.keys())[0]
    
    def _generate_cache_key(self, model_name: str, temperature: float, max_tokens: int, streaming: bool) -> str:
        """
        Generate a cache key for model instances.
        
        Args:
            model_name: Model name
            temperature: Temperature setting
            max_tokens: Max tokens setting
            streaming: Streaming setting
            
        Returns:
            str: Cache key
        """
        return f"{model_name}_{temperature}_{max_tokens}_{streaming}"
    
    def _create_fallback_model(self, temperature: float = 0.2) -> BaseChatModel:
        """
        Create a fallback model when primary loading fails.
        
        Args:
            temperature: Sampling temperature
            
        Returns:
            BaseChatModel: Fallback model
        """
        try:
            # Try Groq first (usually most reliable)
            if os.environ.get("GROQ_API_KEY"):
                return ChatGroq(
                    temperature=temperature,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                    model_name="llama-3.3-70b-versatile"
                )
            
            # Then OpenAI
            if os.environ.get("OPENAI_API_KEY"):
                return ChatOpenAI(
                    model_name="gpt-4o-mini",
                    openai_api_key=os.environ["OPENAI_API_KEY"],
                    temperature=temperature
                )
            
        except Exception as e:
            logger.error(f"Error creating fallback model: {str(e)}")
            raise RuntimeError("Unable to create any LLM instance")
    
    def _update_usage_stats(self, model_name: str) -> None:
        """
        Update usage statistics for the model.
        
        Args:
            model_name: Name of the model being used
        """
        try:
            if model_name not in self.usage_stats:
                self.usage_stats[model_name] = {
                    "usage_count": 0,
                    "first_used": datetime.now().isoformat(),
                    "last_used": None
                }
            
            self.usage_stats[model_name]["usage_count"] += 1
            self.usage_stats[model_name]["last_used"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error updating usage stats: {str(e)}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models with their configurations.
        
        Returns:
            List of model information dictionaries
        """
        try:
            available_models = []
            
            for model_name, config in self.models_config.items():
                # Check if API key is available for the provider
                is_available = True
                if config.provider == ModelProvider.OPENAI and not os.environ.get("OPENAI_API_KEY"):
                    is_available = False
                elif config.provider == ModelProvider.GROQ and not os.environ.get("GROQ_API_KEY"):
                    is_available = False
                elif config.provider == ModelProvider.ANTHROPIC and not os.environ.get("ANTHROPIC_API_KEY"):
                    is_available = False
                
                model_info = {
                    "name": model_name,
                    "provider": config.provider.value,
                    "tier": config.tier.value,
                    "description": config.description,
                    "context_window": config.context_window,
                    "cost_per_1k_tokens": config.cost_per_1k_tokens,
                    "supports_streaming": config.supports_streaming,
                    "supports_functions": config.supports_functions,
                    "is_available": is_available,
                    "usage_count": self.usage_stats.get(model_name, {}).get("usage_count", 0)
                }
                
                available_models.append(model_info)
            
            return sorted(available_models, key=lambda x: (not x["is_available"], x["cost_per_1k_tokens"]))
            
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []
    
    def get_model_recommendations(self, use_case: str = "general") -> List[str]:
        """
        Get model recommendations based on use case.
        
        Args:
            use_case: Specific use case (general, speed, cost, quality)
            
        Returns:
            List of recommended model names
        """
        try:
            recommendations = {
                "general": ["llama-3.3-70b-versatile", "gpt-4o-mini", "claude-3-sonnet-20240229"],
                "speed": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gpt-4o-mini"],
                "cost": ["llama3", "mixtral-8x7b-32768", "gpt-4o-mini"],
                "quality": ["gpt-4o", "claude-3-opus-20240229", "gpt-4-turbo"],
                "privacy": ["llama3"]  # Local models only
            }
            
            return recommendations.get(use_case, recommendations["general"])
            
        except Exception as e:
            logger.error(f"Error getting model recommendations: {str(e)}")
            return ["llama-3.3-70b-versatile"]
    
    def clear_cache(self) -> None:
        """Clear the model cache to free memory."""
        try:
            self.model_cache.clear()
            logger.info("Model cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics.
        
        Returns:
            Dict containing usage statistics
        """
        try:
            total_usage = sum(stats["usage_count"] for stats in self.usage_stats.values())
            
            return {
                "total_model_calls": total_usage,
                "unique_models_used": len(self.usage_stats),
                "model_usage": self.usage_stats,
                "cache_size": len(self.model_cache),
                "available_models": len(self.models_config)
            }
            
        except Exception as e:
            logger.error(f"Error getting usage statistics: {str(e)}")
            return {}


# Global LLM manager instance
llm_manager = EnhancedLLMManager()


# Backward compatibility function
def load_llm(llm_name: str = None, **kwargs) -> BaseChatModel:
    """
    Backward compatible LLM loading function.
    
    Args:
        llm_name: LLM provider name (openai, groq, llama3)
        **kwargs: Additional parameters
        
    Returns:
        BaseChatModel: Configured language model
    """
    try:
        # Map old naming to new system
        provider_mapping = {
            "openai": "openai",
            "groq": "groq", 

            "anthropic": "anthropic"
        }
        
        provider = provider_mapping.get(llm_name, llm_name)
        
        return llm_manager.load_llm(
            provider=provider,
            **kwargs
        )
        
    except Exception as e:
        logger.error(f"Error in backward compatible load_llm: {str(e)}")
        # Ultimate fallback
        return ChatGroq(
            temperature=0.2,
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            model_name="llama-3.3-70b-versatile"
        )


# Utility functions
def get_optimal_model_for_task(task_type: str = "general") -> str:
    """
    Get the optimal model for a specific task type.
    
    Args:
        task_type: Type of task (analysis, generation, search, etc.)
        
    Returns:
        str: Recommended model name
    """
    task_models = {
        "analysis": "gpt-4o",
        "generation": "llama-3.3-70b-versatile", 
        "search": "gpt-4o-mini",
        "research": "claude-3-sonnet-20240229",
        "coding": "gpt-4o",
        "general": "llama-3.3-70b-versatile"
    }
    
    return task_models.get(task_type, "llama-3.3-70b-versatile")


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate which API keys are available.
    
    Returns:
        Dict mapping provider names to availability status
    """
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),

    }