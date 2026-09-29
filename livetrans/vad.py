"""流式语音活动检测（VAD）+ 语音分段。

用 faster-whisper 自带的 Silero VAD（ONNX，CPU 跑）做逐帧语音概率判断，
再把连续语音攒成一段一段送给识别器。这样既能把「没人说话」的时间省掉，
也能让每一段都是相对完整的一句话，翻译质量比固定切窗好很多。

最关键的是**短句缓冲**：静音达标断句后，如果这一段太短，不立刻送出去，
而是攒着等下一段接上。主播换气、思考造成的短停顿不该被当成句末——
否则「ゲームで遊ばせていただこうと思い」会被拆成四段，每段单独翻译必然翻错。
"""

from __future__ import annotations

import logging

import numpy as np

from .config import VadConfig

logger = logging.getLogger(__name__)

__all__ = ["SpeechSegmenter", "load_vad_model"]

# Silero VAD v5 在 16kHz 下的固定窗口长度
FRAME = 512


def load_vad_model():  # noqa: ANN201 - 返回 faster_whisper 内部类型
    """加载 Silero VAD ONNX 模型（复用 faster-whisper 打包的权重，不额外下模型）。"""
    from faster_whisper.vad import get_vad_model

    return get_vad_model()


def _append_trim(buf: np.ndarray, chunk: np.ndarray, keep: int) -> np.ndarray:
    """把 chunk 追加到 buf 尾部，并只保留最后 keep 个样本。"""
    merged = np.concatenate([buf, chunk]) if buf.size else chunk
    if keep > 0 and merged.size > keep:
        merged = merged[-keep:]
    return merged


class SpeechSegmenter:
    """把连续音频流切成一句一句的语音片段。

    ``process()`` 每次喂入任意长度的一小块音频（16kHz 单声道 float32），
    返回本次内部新完成的语音段列表（每段也是 float32 numpy 数组）。
    没人说话时返回空列表。
    """

    def apply_config(self) -> None:
        """按当前 cfg 重算各种帧数门槛。

        运行时改了 VAD 参数（比如从托盘菜单调）之后调一次就生效，不用重启、
        也不用重新加载模型。这里只是几个整数赋值，capture 线程即使正好在执行，
        最坏也就是某一帧用了旧门槛。
        """
        cfg = self.cfg
        # VAD 每次只看最近这么长的一段做判断，给 LSTM 提供上下文
        self.ctx_frames = max(1, int(round(400.0 / self.frame_ms)))
        self.pad_frames = max(0, int(round(cfg.padding_ms / self.frame_ms)))
        self.min_speech_frames = max(1, int(round(cfg.min_speech_ms / self.frame_ms)))
        self.min_silence_frames = max(1, int(round(cfg.min_silence_ms / self.frame_ms)))
        self.max_frames = max(1, int(round(cfg.max_segment_seconds * 1000.0 / self.frame_ms)))
        self.min_seg_frames = max(1, int(round(cfg.min_segment_seconds * 1000.0 / self.frame_ms)))
        # 短于 merge_frames 的段先攒着；hold_frames 是「攒着还能等多久」
        self.merge_frames = max(1, int(round(cfg.merge_seconds * 1000.0 / self.frame_ms)))
        self.hold_frames = max(1, int(round(cfg.hold_ms / self.frame_ms)))
        self.preroll_keep = (self.pad_frames + 4) * FRAME

    def __init__(self, cfg: VadConfig, model, sample_rate: int = 16000) -> None:
        self.cfg = cfg
        self.model = model
        self.sr = sample_rate
        self.frame_ms = 1000.0 * FRAME / sample_rate
        self.apply_config()

        self._incoming = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._preroll = np.zeros(0, dtype=np.float32)
        self._seg = np.zeros(0, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

        self._speaking = False
        self._frames_in_seg = 0
        self._speech_run = 0
        self._silence_run = 0
        self._idle_frames = 0

    # ---- 对外接口 --------------------------------------------------------

    def process(self, audio: np.ndarray) -> list[np.ndarray]:
        if audio.size == 0:
            return []

        buf = np.concatenate([self._incoming, audio]) if self._incoming.size else audio
        n_frames = buf.size // FRAME
        if n_frames == 0:
            self._incoming = buf
            return []

        body = buf[: n_frames * FRAME]
        self._incoming = buf[n_frames * FRAME :]
        frames = body.reshape(n_frames, FRAME)

        probs = self._speech_probs(frames)

        finished: list[np.ndarray] = []
        for i in range(n_frames):
            finished.extend(self._advance(float(probs[i]), frames[i]))
        return finished

    def flush(self) -> list[np.ndarray]:
        """停止捕获时，把手里没送出去的东西全吐出来。"""
        out: list[np.ndarray] = []
        if self._speaking and self._seg.size and self._frames_in_seg >= self.min_seg_frames:
            out.append(self._seg)
        if self._pending.size:
            out.append(self._pending)
        self._reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._idle_frames = 0
        return out

    def level(self, audio: np.ndarray) -> float:
        """块音量（RMS），用于 UI 显示播放电平。"""
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    # ---- 内部实现 --------------------------------------------------------

    def _speech_probs(self, frames: np.ndarray) -> np.ndarray:
        flat = frames.reshape(-1)
        window = np.concatenate([self._context, flat]) if self._context.size else flat

        try:
            probs = np.asarray(self.model(window.astype(np.float32, copy=False))).reshape(-1)
        except Exception:  # noqa: BLE001 - VAD 出问题不该让整条链路挂掉
            logger.exception("VAD 推理失败，本批按静音处理")
            probs = np.zeros(window.size // FRAME, dtype=np.float32)

        keep = self.ctx_frames * FRAME
        self._context = window[-keep:]

        n = frames.shape[0]
        if probs.size < n:
            probs = np.pad(probs, (n - probs.size, 0), constant_values=0.0)
        return probs[-n:]

    def _advance(self, prob: float, frame: np.ndarray) -> list[np.ndarray]:
        threshold = self.cfg.threshold
        out: list[np.ndarray] = []

        if not self._speaking:
            self._preroll = _append_trim(self._preroll, frame, self.preroll_keep)

            if prob >= threshold:
                self._speech_run += 1
                self._idle_frames = 0
                if self._speech_run >= self.min_speech_frames:
                    self._speaking = True
                    self._seg = self._preroll.copy()
                    self._frames_in_seg = self._speech_run
                    self._silence_run = 0
                return out

            # 静音：如果手里攒着短句，数着还要等多久才认定「确实说完了」
            self._speech_run = 0
            if self._pending.size:
                self._idle_frames += 1
                if self._idle_frames >= self.hold_frames:
                    out.append(self._take_pending())
            return out

        self._seg = np.concatenate([self._seg, frame])
        self._frames_in_seg += 1
        if prob >= threshold:
            self._silence_run = 0
        else:
            self._silence_run += 1

        if self._silence_run >= self.min_silence_frames or self._frames_in_seg >= self.max_frames:
            closed = self._close_segment()
            if closed is not None:
                out.extend(self._absorb(closed))
        return out

    def _absorb(self, seg: np.ndarray) -> list[np.ndarray]:
        """收下一段刚结束的语音：够长就直接放行，太短就先攒着等后文接上。"""
        if self._pending.size:
            seg = np.concatenate([self._pending, seg])
            self._pending = np.zeros(0, dtype=np.float32)

        self._idle_frames = 0
        if seg.size >= self.merge_frames * FRAME:
            return [seg]
        self._pending = seg
        return []

    def _take_pending(self) -> np.ndarray:
        seg = self._pending
        self._pending = np.zeros(0, dtype=np.float32)
        self._idle_frames = 0
        return seg

    def _close_segment(self) -> np.ndarray | None:
        seg = self._seg
        frames = self._frames_in_seg
        silence = self._silence_run
        self._reset()

        if silence > self.pad_frames:
            drop = (silence - self.pad_frames) * FRAME
            if 0 < drop < seg.size:
                seg = seg[:-drop]

        return seg if frames >= self.min_seg_frames else None

    def _reset(self) -> None:
        """只清「当前这一句」的状态；_pending 要留着跨句累积。"""
        self._speaking = False
        self._seg = np.zeros(0, dtype=np.float32)
        self._frames_in_seg = 0
        self._speech_run = 0
        self._silence_run = 0
        self._preroll = np.zeros(0, dtype=np.float32)
