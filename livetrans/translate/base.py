"""翻译 provider 的公共部分：提示词构造、重试、错误类型。"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

import httpx

from ..config import TranslateConfig

logger = logging.getLogger(__name__)

__all__ = ["Translator", "TranslationError", "build_messages"]

ContextPair = tuple[str, str]

_SYSTEM = """你是 VTuber 直播的实时字幕翻译引擎，负责把日语口语翻译成{target}。

严格遵守：
1. 只输出译文本身。不要解释、不要加引号、不要输出罗马音、不要重复原文、不要用「翻译：」这类前缀。
2. 译文要是自然的{target}口语，符合直播口播的习惯，不要书面语、不要生硬直译。
3. 保留语气和情绪：撒娇、吐槽、惊讶、兴奋这些要用对应的{target}语气词体现出来。
4. 人名、游戏名、作品名、专有名词若没有公认译名，保留日文原文，绝不臆造。
5. 听不清或无意义的语气词、拟声词（あー、えっと、ん？），用最接近的{target}语气词带过，不要编造内容。
6. 长度尽量和原文相当，一行说完，适合做字幕。
7. 输入是从直播里切出来的口语，可能没有标点、语序颠倒、上下句粘连。先在心里还原成通顺的日语再译，不要逐词硬译；句子明显没说完时，用最自然的方式收尾，不要自行编造具体内容。"""

_CLEAN_PREFIX = re.compile(
    r"^\s*(翻译|譯文|译文|中文|简体中文|繁體中文|translation|output)\s*[:：]\s*", re.IGNORECASE
)


class TranslationError(RuntimeError):
    """翻译多次重试后仍然失败。"""


def build_messages(
    cfg: TranslateConfig,
    text: str,
    context: Sequence[ContextPair] = (),
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)。"""
    system = _SYSTEM.format(target=cfg.target_language)
    if cfg.extra_prompt.strip():
        system += "\n\n补充要求：\n" + cfg.extra_prompt.strip()

    lines: list[str] = []
    if context and cfg.context_lines > 0:
        recent = list(context)[-cfg.context_lines :]
        lines.append("以下是前文，仅用于理解语境，不要翻译这些内容：")
        for ja, zh in recent:
            lines.append(f"  原文：{ja}")
            lines.append(f"  译文：{zh}")
        lines.append("")

    lines.append("现在翻译下面这一句，只输出译文：")
    lines.append(text)
    return system, "\n".join(lines)


def clean_output(raw: str) -> str:
    """去掉模型喜欢加的前缀、引号和换行。"""
    out = (raw or "").strip()
    out = _CLEAN_PREFIX.sub("", out)
    if len(out) >= 2 and out[0] in "「『\"'" and out[-1] in "」』\"'":
        out = out[1:-1].strip()
    return out


class Translator(ABC):
    """所有翻译后端的基类。"""

    name = "base"

    def __init__(self, cfg: TranslateConfig) -> None:
        self.cfg = cfg
        self._client = httpx.AsyncClient(timeout=cfg.timeout)

    @abstractmethod
    async def _request(self, system: str, user: str) -> str:
        """发一次请求，返回模型原始输出。"""

    async def translate(self, text: str, context: Sequence[ContextPair] = ()) -> str:
        system, user = build_messages(self.cfg, text, context)
        last_error: Exception | None = None

        for attempt in range(self.cfg.retries + 1):
            try:
                raw = await self._request(system, user)
                out = clean_output(raw)
                if out:
                    return out
                last_error = TranslationError("模型返回了空文本")
            except Exception as exc:  # noqa: BLE001 - 网络/限流都要能重试
                last_error = exc
                logger.warning("[%s] 翻译失败（第 %d 次）：%s", self.name, attempt + 1, exc)
            if attempt < self.cfg.retries:
                await asyncio.sleep(0.4 * (attempt + 1))

        raise TranslationError(f"{self.name} 翻译失败：{last_error}") from last_error

    async def aclose(self) -> None:
        await self._client.aclose()
