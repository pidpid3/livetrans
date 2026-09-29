"""音频 → 识别 → 翻译 的多线程流水线。

线程分工：

* **capture 线程**：从 WASAPI 环回抓声音，交给 VAD 切句，推入 ASR 队列
* **asr 线程**：跑 faster-whisper 出日文原文，推入翻译队列
* **translate 线程**：跑 asyncio 事件循环调用云翻译，产出字幕行

三者用有界队列连接，任一级跟不上就丢最旧的片段——实时字幕宁可漏一句，也不要越积越延迟。
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .asr import AsrEngine
from .audio import AudioError, LoopbackCapture
from .config import Config
from .translate import Translator, create_translator
from .vad import SpeechSegmenter, load_vad_model

logger = logging.getLogger(__name__)

__all__ = ["SubtitleLine", "PipelineCallbacks", "Pipeline"]

# 本地推理地址不需要 API key（与 translate/openai_compat.py 的判断保持一致）
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


@dataclass
class SubtitleLine:
    """一条成品字幕。"""

    ja: str
    zh: str = ""
    error: str = ""
    asr_latency: float = 0.0
    translate_latency: float = 0.0
    audio_seconds: float = 0.0

    @property
    def total_latency(self) -> float:
        return self.asr_latency + self.translate_latency


@dataclass
class PipelineCallbacks:
    on_line: Callable[[SubtitleLine], None] = lambda line: None
    on_status: Callable[[str], None] = lambda msg: None
    on_error: Callable[[str], None] = lambda msg: None
    on_level: Callable[[float], None] = lambda level: None
    stats: dict = field(default_factory=dict)


def _offer(q: "queue.Queue", item: object) -> None:
    """尽力入队；满了就丢掉最旧的一条。"""
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
        q.put_nowait(item)
    except (queue.Empty, queue.Full):
        pass


def _take(q: "queue.Queue", timeout: float = 0.25):
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


class Pipeline:
    """把各个子系统串起来。UI 只跟 callbacks 打交道。"""

    def __init__(self, cfg: Config, callbacks: PipelineCallbacks | None = None) -> None:
        self.cfg = cfg
        self.cb = callbacks or PipelineCallbacks()

        self.asr = AsrEngine(cfg.asr)
        self.translator: Translator | None = None
        self.capture: LoopbackCapture | None = None
        self.segmenter: SpeechSegmenter | None = None

        self._stop = threading.Event()
        self._stopping = False
        self._asr_queue: queue.Queue = queue.Queue(maxsize=max(1, cfg.asr.max_queue))
        self._tr_queue: queue.Queue = queue.Queue(maxsize=max(2, cfg.asr.max_queue * 2))
        self._context: list[tuple[str, str]] = []
        self._threads: list[threading.Thread] = []

    # ---- 生命周期 --------------------------------------------------------

    def start(self) -> None:
        cfg = self.cfg

        # 环境变量里的 key 落到配置上，各 translator 只认 cfg.api_key
        resolved = cfg.resolve_api_key()
        if resolved and not cfg.translate.api_key:
            cfg.translate.api_key = resolved

        self.translator = create_translator(cfg.translate)
        if self.translator is None:
            self.cb.on_status("翻译已关闭（provider = none），只显示日文原文")
        else:
            problem = self._translation_problem()
            if problem:
                # 没 key 就别一句一句白试，直接关掉翻译，只显示日文
                self.cb.on_status(f"⚠ {problem}（本次只显示日文原文）")
                self.translator = None

        self.cb.on_status("正在加载 VAD 模型…")
        self.segmenter = SpeechSegmenter(cfg.vad, load_vad_model())

        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._asr_loop, name="asr", daemon=True),
            threading.Thread(target=self._translate_loop, name="translate", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _translation_problem(self) -> str:
        """翻译配置能不能用；返回空串表示没问题。"""
        translate = self.cfg.translate
        base_url = (translate.base_url or "").lower()
        if translate.api_key or any(host in base_url for host in _LOCAL_HOSTS):
            return ""
        return (
            f"没有翻译 API key：请在 config.toml 填 [translate] api_key，"
            f"或设置环境变量 {translate.api_key_env}"
        )

    def apply_vad_config(self) -> None:
        """VAD 参数被运行时改过了，让分段器重算门槛。不用重启，也不用重载模型。"""
        if self.segmenter is not None:
            self.segmenter.apply_config()

    def stop(self) -> None:
        """停掉所有线程。可以重复调用（Qt 的 aboutToQuit 和 finally 会各来一次）。"""
        if self._stopping:
            return
        self._stopping = True
        self._stop.set()

        if self.capture is not None:
            self.capture.stop()

        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads = []

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # ---- capture 线程 ----------------------------------------------------

    def _capture_loop(self) -> None:
        # 先把对象挂到 self 上，stop() 才拿得到它
        capture = LoopbackCapture(self.cfg.audio)
        self.capture = capture

        try:
            self.cb.on_status("正在打开音频环回设备…")
            capture.start()
        except AudioError as exc:
            self.cb.on_error(f"音频捕获失败：{exc}")
            self._stop.set()
            return

        self.cb.on_status(
            f"✓ 音频已接入：{capture.device_name} "
            f"（{capture.native_rate} Hz → {capture.target_rate} Hz）"
        )
        for msg in capture.status_messages:
            self.cb.on_status(f"⚠ {msg}")

        segmenter = self.segmenter
        if segmenter is None:
            return

        level_counter = 0
        while not self._stop.is_set():
            block = capture.read(timeout=0.4)
            if block is None or block.size == 0:
                continue

            for segment in segmenter.process(block):
                _offer(self._asr_queue, segment)

            level_counter += 1
            if level_counter >= 3:
                level_counter = 0
                self.cb.on_level(segmenter.level(block))

        if capture.dropped_blocks:
            self.cb.on_status(f"⚠ 音频丢块 {capture.dropped_blocks} 次，可调大 [audio] block_seconds")

    # ---- asr 线程 --------------------------------------------------------

    def _asr_loop(self) -> None:
        self.cb.on_status("正在加载语音识别模型…（首次运行需要下载模型，可能要几分钟）")
        try:
            self.asr.load()
        except Exception as exc:  # noqa: BLE001
            self.cb.on_error(str(exc))
            self._stop.set()
            return

        # CUDA 上第一次推理要编译 kernel，不预热的话首句字幕会卡十几秒
        self.cb.on_status("正在预热识别引擎…（首次约十几秒，之后每句只要一两百毫秒）")
        self.asr.warmup()

        self.cb.on_status(
            f"✓ 识别就绪：{self.cfg.asr.model} @ {self.asr.device}/{self.asr.compute_type}"
        )

        while not self._stop.is_set():
            segment = _take(self._asr_queue)
            if segment is None:
                continue
            try:
                transcript = self.asr.transcribe(segment, started_at=time.monotonic())
            except Exception as exc:  # noqa: BLE001
                logger.exception("识别失败")
                self.cb.on_error(f"识别失败：{exc}")
                continue
            if transcript is None:
                continue

            _offer(self._tr_queue, transcript)

    # ---- translate 线程 --------------------------------------------------

    def _translate_loop(self) -> None:
        try:
            asyncio.run(self._translate_async())
        except Exception:  # noqa: BLE001
            logger.exception("翻译线程异常退出")
            self.cb.on_error("翻译线程异常退出，详情见控制台日志")

    async def _translate_async(self) -> None:
        try:
            while not self._stop.is_set():
                transcript = await asyncio.to_thread(_take, self._tr_queue)
                if transcript is None:
                    continue
                await self._handle(transcript)
        finally:
            # AsyncClient 必须在创建它的那个事件循环里关闭
            if self.translator is not None:
                try:
                    await self.translator.aclose()
                except Exception:  # noqa: BLE001
                    logger.debug("关闭翻译客户端时出错", exc_info=True)

    async def _handle(self, transcript) -> None:  # noqa: ANN001 - asr.Transcript
        started = time.monotonic()
        zh, error = "", ""

        if self.translator is not None:
            try:
                zh = await self.translator.translate(transcript.text, list(self._context))
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                logger.warning("翻译失败：%s", exc)

        line = SubtitleLine(
            ja=transcript.text,
            zh=zh,
            error=error,
            asr_latency=transcript.latency,
            translate_latency=time.monotonic() - started,
            audio_seconds=transcript.audio_seconds,
        )
        self.cb.on_line(line)

        if zh:
            self._context.append((transcript.text, zh))
            limit = max(0, self.cfg.translate.context_lines) + 4
            if len(self._context) > limit:
                self._context = self._context[-limit:]
