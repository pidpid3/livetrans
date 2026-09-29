"""翻译后端的协议测试。

用 httpx.MockTransport 拦下真实请求，检查发出去的 URL / 头 / 请求体长什么样，
以及响应怎么解析 —— 不用联网、不用真 key。

这些细节很容易写错且不容易发现：比如 Claude 的 system 必须是**顶层参数**，
塞进 messages 里会被拒；又比如新模型不接受 temperature。
"""

from __future__ import annotations

import json
import unittest

import httpx

from livetrans.config import TranslateConfig
from livetrans.translate import create_translator


def _client_returning(handler) -> httpx.AsyncClient:  # noqa: ANN001
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestBackendSelection(unittest.TestCase):
    def test_claude_aliases(self):
        for alias in ("anthropic", "claude", "CLAUDE"):
            with self.subTest(provider=alias):
                translator = create_translator(TranslateConfig(provider=alias, api_key="k"))
                self.assertEqual(translator.name, "anthropic")

    def test_other_major_providers_use_openai_compat(self):
        """国内外的厂商基本都提供 OpenAI 兼容入口，应该都落到同一个后端。"""
        for alias in ("deepseek", "moonshot", "kimi", "zhipu", "glm", "qwen", "grok", "groq", "openrouter", "azure"):
            with self.subTest(provider=alias):
                translator = create_translator(TranslateConfig(provider=alias, api_key="k"))
                self.assertEqual(translator.name, "openai-compatible")


class TestAnthropicBackend(unittest.IsolatedAsyncioTestCase):
    async def test_request_shape_and_response_parsing(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "你好"}, {"type": "text", "text": "！"}]},
            )

        translator = create_translator(
            TranslateConfig(provider="claude", api_key="sk-ant-test", context_lines=0)
        )
        await translator._client.aclose()  # 换掉自带 client，改走 mock
        translator._client = _client_returning(handler)
        try:
            out = await translator.translate("こんにちは")
        finally:
            await translator.aclose()

        self.assertEqual(out, "你好！", "多个 content block 要拼起来")
        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(captured["headers"]["x-api-key"], "sk-ant-test")
        self.assertIn("anthropic-version", captured["headers"])

        body = captured["body"]
        # 关键：Claude 的 system 是顶层参数，不能放进 messages
        self.assertIn("system", body)
        self.assertEqual([m["role"] for m in body["messages"]], ["user"])
        self.assertIn("max_tokens", body)

    async def test_base_url_ending_in_v1_is_not_doubled(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"content": [{"type": "text", "text": "好"}]})

        translator = create_translator(
            TranslateConfig(
                provider="claude",
                api_key="k",
                base_url="https://api.anthropic.com/v1",
            )
        )
        await translator._client.aclose()
        translator._client = _client_returning(handler)
        try:
            await translator.translate("テスト")
        finally:
            await translator.aclose()

        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")


class TestOpenAICompatParameterFallback(unittest.IsolatedAsyncioTestCase):
    async def test_retries_without_rejected_temperature(self):
        """有些新模型不接受 temperature，要能自动去掉重试。"""
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append(body)
            if "temperature" in body:
                return httpx.Response(
                    400, json={"error": {"message": "Unsupported parameter: 'temperature'"}}
                )
            return httpx.Response(200, json={"choices": [{"message": {"content": "你好"}}]})

        translator = create_translator(TranslateConfig(provider="openai", api_key="k", context_lines=0))
        await translator._client.aclose()
        translator._client = _client_returning(handler)
        try:
            out = await translator.translate("こんにちは")
        finally:
            await translator.aclose()

        self.assertEqual(out, "你好")
        self.assertEqual(len(seen), 2, "应该重试一次")
        self.assertNotIn("temperature", seen[1])

    async def test_switches_to_max_completion_tokens(self):
        """有的模型把 max_tokens 换成了 max_completion_tokens。"""
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append(body)
            if "max_tokens" in body:
                return httpx.Response(
                    400,
                    json={"error": {"message": "Use 'max_completion_tokens' instead of 'max_tokens'"}},
                )
            return httpx.Response(200, json={"choices": [{"message": {"content": "好"}}]})

        translator = create_translator(TranslateConfig(provider="openai", api_key="k", context_lines=0))
        await translator._client.aclose()
        translator._client = _client_returning(handler)
        try:
            out = await translator.translate("テスト")
        finally:
            await translator.aclose()

        self.assertEqual(out, "好")
        self.assertEqual(len(seen), 2)
        self.assertNotIn("max_tokens", seen[1])
        self.assertEqual(seen[1]["max_completion_tokens"], 512)

    async def test_real_errors_are_not_retried(self):
        """不是参数问题的 400（比如 key 无效）不该白试第二次。"""
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

        translator = create_translator(
            TranslateConfig(provider="openai", api_key="bad", context_lines=0, retries=0)
        )
        await translator._client.aclose()
        translator._client = _client_returning(handler)
        try:
            with self.assertRaises(Exception):
                await translator.translate("テスト")
        finally:
            await translator.aclose()

        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
