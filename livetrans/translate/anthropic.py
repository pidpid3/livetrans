"""Anthropic Claude 翻译后端。

Claude 的接口和 OpenAI **不兼容**，所以得单独写一个：

* 鉴权用 ``x-api-key`` 头（不是 ``Authorization: Bearer``），另外必须带 ``anthropic-version``
* 系统提示词是**顶层参数** ``system``，不能塞进 messages 里的 system role
* 回复在 ``content`` 数组里，是一段段 block，要拼起来

其余主流厂商（DeepSeek / OpenAI / Kimi / 智谱 / 通义 / Grok / Mistral / Groq / OpenRouter /
硅基流动 / 火山方舟 / 本地 ollama…）都提供 OpenAI 兼容入口，用 ``provider = "openai"`` 就行。
"""

from __future__ import annotations

import httpx

from .base import Translator

__all__ = ["AnthropicTranslator"]

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-3-5-haiku-latest"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicTranslator(Translator):
    name = "anthropic"

    @property
    def _base(self) -> str:
        return (self.cfg.base_url or DEFAULT_BASE_URL).rstrip("/")

    @property
    def _model(self) -> str:
        return self.cfg.model or DEFAULT_MODEL

    @property
    def _endpoint(self) -> str:
        base = self._base
        # 允许用户把 base_url 写成 .../v1 或直接写到主机名，两种都兼容
        if base.endswith("/v1"):
            return base + "/messages"
        return base + "/v1/messages"

    def _headers(self) -> dict[str, str]:
        if not self.cfg.api_key:
            raise RuntimeError(
                f"Anthropic 需要 API key：请在 config.toml 的 [translate] api_key 填写，"
                f"或设置环境变量 {self.cfg.api_key_env}"
            )
        return {
            "Content-Type": "application/json",
            "x-api-key": self.cfg.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    async def _request(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": 512,
            "temperature": 0.2,
        }

        resp = await self._client.post(self._endpoint, json=payload, headers=self._headers())
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} {resp.text[:300]}",
                request=resp.request,
                response=resp,
            )

        data = resp.json()
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ValueError(f"响应里没有 content 数组：{str(data)[:300]}")

        parts = [
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") in (None, "text")
        ]
        return "".join(parts)
