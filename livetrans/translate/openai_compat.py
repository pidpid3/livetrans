"""OpenAI 兼容的 chat completions 翻译后端。

国内外的绝大多数厂商都提供 OpenAI 兼容入口，所以一个后端全通：
DeepSeek、OpenAI、月之暗面 Kimi、智谱 GLM、通义千问、Grok、Mistral、Groq、Together、
OpenRouter、硅基流动、火山方舟、Azure OpenAI，以及本地 ollama / vLLM / LM Studio。
换个 base_url + model + key 就行。
"""

from __future__ import annotations

import logging

import httpx

from .base import Translator

logger = logging.getLogger(__name__)

__all__ = ["OpenAICompatTranslator"]

# 默认走 DeepSeek：中文输出自然、便宜、国内直连。
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

# 本地推理服务（ollama / vLLM / LM Studio）通常不校验 key，缺 key 不该直接报错
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")

# 400 里出现这些词，说明是参数不被接受，值得换个参数组合再试
_PARAM_HINTS = (
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "unsupported parameter",
    "unsupported value",
    "unrecognized",
    "unknown parameter",
)


class OpenAICompatTranslator(Translator):
    name = "openai-compatible"

    @property
    def _model(self) -> str:
        return self.cfg.model or DEFAULT_MODEL

    @property
    def _base(self) -> str:
        return (self.cfg.base_url or DEFAULT_BASE_URL).rstrip("/")

    @property
    def _endpoint(self) -> str:
        return self._base + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.cfg.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        elif not any(host in self._base for host in _LOCAL_HOSTS):
            raise RuntimeError(
                f"缺少翻译 API key：请在 config.toml 的 [translate] api_key 填写，"
                f"或设置环境变量 {self.cfg.api_key_env}"
            )
        return headers

    def _payload(self, system: str, user: str) -> dict:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
            "stream": False,
        }

    async def _post(self, payload: dict) -> httpx.Response:
        return await self._client.post(self._endpoint, json=payload, headers=self._headers())

    async def _request(self, system: str, user: str) -> str:
        payload = self._payload(system, user)
        resp = await self._post(payload)

        # 有的新模型（OpenAI 的 o 系列 / gpt-5 之类，以及跟进的厂商）不接受 temperature，
        # 或者把 max_tokens 换成了 max_completion_tokens。碰到这种 400 就退回最小参数再试一次。
        if resp.status_code == 400 and any(hint in resp.text.lower() for hint in _PARAM_HINTS):
            logger.debug("参数被拒绝，改用最小参数重试：%s", resp.text[:200])
            retry = dict(payload)
            retry.pop("temperature", None)
            if "max_completion_tokens" in resp.text:
                retry.pop("max_tokens", None)
                retry["max_completion_tokens"] = 512
            resp = await self._post(retry)

        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} {resp.text[:300]}",
                request=resp.request,
                response=resp,
            )

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"响应里没有 choices：{str(data)[:300]}")
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")
