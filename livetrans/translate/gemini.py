"""Google Gemini 翻译后端。"""

from __future__ import annotations

import httpx

from .base import Translator

__all__ = ["GeminiTranslator"]

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiTranslator(Translator):
    name = "gemini"

    @property
    def _base(self) -> str:
        return (self.cfg.base_url or DEFAULT_BASE_URL).rstrip("/")

    @property
    def _model(self) -> str:
        return self.cfg.model or DEFAULT_MODEL

    @property
    def _endpoint(self) -> str:
        return f"{self._base}/models/{self._model}:generateContent"

    async def _request(self, system: str, user: str) -> str:
        key = self.cfg.api_key
        if not key:
            raise RuntimeError(
                f"Gemini 需要 API key，请在 config.toml 的 [translate] api_key 填写，"
                f"或设置环境变量 {self.cfg.api_key_env}"
            )

        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        base_payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
        }

        # 2.5 系列默认会“思考”，实时字幕等不起，先试关掉思考
        payload = {
            **base_payload,
            "generationConfig": {**base_payload["generationConfig"], "thinkingConfig": {"thinkingBudget": 0}},
        }

        resp = await self._client.post(self._endpoint, json=payload, headers=headers)
        if resp.status_code == 400 and "thinking" in resp.text.lower():
            resp = await self._client.post(self._endpoint, json=base_payload, headers=headers)

        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} {resp.text[:300]}",
                request=resp.request,
                response=resp,
            )

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise ValueError(f"响应里没有 candidates：{str(data)[:300]}")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(str(part.get("text") or "") for part in parts)
