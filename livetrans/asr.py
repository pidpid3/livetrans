"""语音识别：faster-whisper 本地跑日语 ASR。

默认在独显上用 float16 跑 large-v3-turbo，显存不够或没有 CUDA 时自动退到 CPU int8。
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .config import AsrConfig

logger = logging.getLogger(__name__)

__all__ = ["AsrEngine", "Transcript"]


# Whisper 在没人说话 / 只有音乐时最爱幻想出来的句子，命中就丢掉
_HALLUCINATIONS = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ご視聴いただきありがとうございました",
    "ありがとうございました",
    "ありがとうございます",
    "おやすみなさい",
    "おはようございます",
    "チャンネル登録お願いします",
    "チャンネル登録よろしくお願いします",
    "ごちそうさまでした",
    "おめでとうございます",
    "字幕",
    "字幕by",
    "ん",
    "はい",
    "www",
    "えー",
    "あー",
}

_PUNCT = re.compile(r"[\s。、，．！？!?…・~〜「」『』（）()\[\]\-—ー]+")


def _normalize(text: str) -> str:
    return _PUNCT.sub("", text).strip()


def _looks_like_hallucination(text: str) -> bool:
    norm = _normalize(text)
    if not norm:
        return True
    if norm in _HALLUCINATIONS:
        return True
    # 同一个字重复到底，或者整句复读两遍，都是典型幻觉
    if len(norm) >= 2 and len(set(norm)) == 1:
        return True
    if len(norm) >= 8 and norm == norm[: len(norm) // 2] * 2:
        return True
    return False


@dataclass
class Transcript:
    text: str
    started_at: float
    audio_seconds: float
    latency: float
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    meta: dict = field(default_factory=dict)


class AsrEngine:
    """faster-whisper 封装。第一次 `load()` 会下载/加载模型，比较慢。"""

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self.model = None
        self.device = ""
        self.compute_type = ""

    # ---- 初始化 ----------------------------------------------------------

    def load(self) -> None:
        if self.cfg.hf_endpoint:
            # 必须在 huggingface_hub 真正发请求前设置好，国内镜像用
            os.environ.setdefault("HF_ENDPOINT", self.cfg.hf_endpoint)
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        # 没开 Windows 开发者模式时，huggingface_hub 会刷一堆 symlink 警告，压掉
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        from faster_whisper import WhisperModel

        last_error: Exception | None = None
        for device, compute_type in self._candidates():
            try:
                logger.info(
                    "加载 whisper 模型 %s (device=%s, compute_type=%s)",
                    self.cfg.model,
                    device,
                    compute_type,
                )
                started = time.monotonic()
                self.model = WhisperModel(
                    self.cfg.model,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=os.cpu_count() or 4,
                )
                self.device = device
                self.compute_type = compute_type
                logger.info("模型加载完成，用时 %.1fs", time.monotonic() - started)
                break
            except Exception as exc:  # noqa: BLE001 - 逐个候选降级
                last_error = exc
                logger.warning("以 device=%s compute_type=%s 加载失败：%s", device, compute_type, exc)

        if self.model is None:
            raise RuntimeError(
                f"whisper 模型加载失败（{self.cfg.model}）：{last_error}\n"
                "常见原因：显存不足、CUDA/cuDNN 运行库缺失、网络下载不通。"
                '可以先把 config.toml 里 [asr] device 改成 "cpu"、model 改成 "medium" 试试。'
            ) from last_error

    def _candidates(self) -> list[tuple[str, str]]:
        want_device = self.cfg.device
        want_compute = self.cfg.compute_type

        if want_device != "auto":
            if want_compute != "auto":
                return [(want_device, want_compute)]
            return [(want_device, "float16" if want_device == "cuda" else "int8")]

        auto_compute = "float16" if want_compute == "auto" else want_compute
        fallback_compute = "int8" if want_compute == "auto" else want_compute
        # Windows 上 ctranslate2 找 CUDA 运行库偶尔会翻车，所以永远留一条 CPU 后路
        return [("cuda", auto_compute), ("cpu", fallback_compute)]

    def warmup(self, seconds: float = 2.0) -> float:
        """用一段静音空跑一次推理。

        CUDA 上第一次推理要编译 kernel，实测首句能卡到 16 秒；提前跑掉这份开销，
        之后每句就只要一两百毫秒。返回预热耗时（秒）。
        """
        if self.model is None:
            return 0.0

        silence = np.zeros(int(16000 * max(0.5, seconds)), dtype=np.float32)
        started = time.monotonic()
        try:
            segments, _ = self.model.transcribe(
                silence,
                language=self.cfg.language,
                task="transcribe",
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
                without_timestamps=True,
            )
            # transcribe 返回的是生成器，必须消费掉才真的会跑推理
            for _ in segments:
                pass
        except Exception:  # noqa: BLE001 - 预热失败不该拦住整条链路
            logger.warning("预热失败，首句字幕可能会慢一次", exc_info=True)
            return 0.0

        elapsed = time.monotonic() - started
        logger.info("识别引擎预热完成，用时 %.1fs", elapsed)
        return elapsed

    # ---- 识别 ------------------------------------------------------------

    def transcribe(self, audio: np.ndarray, started_at: float) -> Transcript | None:
        if self.model is None:
            raise RuntimeError("AsrEngine.load() 还没调用")

        audio_seconds = audio.size / 16000.0
        t0 = time.monotonic()

        segments, info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            task="transcribe",
            beam_size=self.cfg.beam_size,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
            # initial_prompt 既接受字符串也接受 token id 序列；用来固化人名和术语
            initial_prompt=self.cfg.initial_prompt or None,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.2,
            compression_ratio_threshold=2.4,
        )

        parts: list[str] = []
        logprobs: list[float] = []
        no_speech: list[float] = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                parts.append(text)
                logprobs.append(float(seg.avg_logprob))
                no_speech.append(float(getattr(seg, "no_speech_prob", 0.0)))

        latency = time.monotonic() - t0
        text = re.sub(r"\s+", " ", "".join(parts)).strip()

        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
        no_speech_prob = sum(no_speech) / len(no_speech) if no_speech else 0.0

        if not text or _looks_like_hallucination(text):
            return None

        return Transcript(
            text=text,
            started_at=started_at,
            audio_seconds=audio_seconds,
            latency=latency,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            meta={"language_probability": float(getattr(info, "language_probability", 0.0))},
        )
