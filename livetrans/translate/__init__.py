"""翻译后端工厂。"""

from __future__ import annotations

from ..config import TranslateConfig
from .base import Translator, TranslationError, clean_output

__all__ = ["Translator", "TranslationError", "create_translator", "clean_output"]

_OPENAI_ALIASES = {
    "openai",
    "openai-compatible",
    "openai_compatible",
    "compatible",
    # 下面这些厂商都提供 OpenAI 兼容入口，走的都是同一个后端，
    # 换个 base_url + model + key 就行（别名只是让你写起来顺一点）
    "deepseek",
    "moonshot",  # 月之暗面 Kimi
    "kimi",
    "zhipu",  # 智谱 GLM
    "glm",
    "dashscope",  # 阿里通义千问
    "qwen",
    "xai",  # Grok
    "grok",
    "mistral",
    "groq",
    "together",
    "openrouter",
    "siliconflow",
    "volcengine",  # 火山方舟 / 豆包
    "ark",
    "ollama",
    "vllm",
    "lmstudio",
    "azure",  # Azure OpenAI
}
_ANTHROPIC_ALIASES = {"anthropic", "claude"}
_DISABLED = {"", "none", "off", "disabled", "no"}


def create_translator(cfg: TranslateConfig) -> Translator | None:
    """按配置创建翻译后端；provider 为 none 时返回 None（只显示识别原文）。"""
    provider = cfg.provider.strip().lower()

    if provider in _DISABLED:
        return None

    if provider in _OPENAI_ALIASES:
        from .openai_compat import OpenAICompatTranslator

        return OpenAICompatTranslator(cfg)

    if provider == "gemini":
        from .gemini import GeminiTranslator

        return GeminiTranslator(cfg)

    if provider in _ANTHROPIC_ALIASES:
        from .anthropic import AnthropicTranslator

        return AnthropicTranslator(cfg)

    raise ValueError(
        f"未知的翻译 provider：{cfg.provider!r}。支持：openai / anthropic / gemini / none"
    )
