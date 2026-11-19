"""Build the configured LLM provider. xAI is the default."""

from __future__ import annotations

from agentic_rag_platform.providers.anthropic import AnthropicProvider
from agentic_rag_platform.providers.base import LLMProvider, ProviderError
from agentic_rag_platform.providers.openai_provider import OpenAIProvider
from agentic_rag_platform.providers.xai import XAIProvider
from agentic_rag_platform.settings import Settings, get_settings


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Return the adapter selected by ``LLM_PROVIDER`` (default: ``xai``)."""
    cfg = settings or get_settings()
    timeout = cfg.llm_timeout_seconds
    temperature = cfg.llm_temperature

    if cfg.llm_provider == "openai":
        key = cfg.openai_api_key.get_secret_value() if cfg.openai_api_key else ""
        return OpenAIProvider(key, cfg.openai_model, timeout=timeout, temperature=temperature)
    if cfg.llm_provider == "anthropic":
        key = cfg.anthropic_api_key.get_secret_value() if cfg.anthropic_api_key else ""
        return AnthropicProvider(key, cfg.anthropic_model, timeout=timeout, temperature=temperature)
    if cfg.llm_provider == "xai":
        key = cfg.xai_api_key.get_secret_value() if cfg.xai_api_key else ""
        return XAIProvider(
            key,
            cfg.xai_model,
            api_host=cfg.xai_api_host,
            timeout=timeout,
            temperature=temperature,
        )
    raise ProviderError(f"Unknown LLM provider: {cfg.llm_provider}")
