"""WASAPI 环回（loopback）音频捕获。

抓的是「扬声器正在播放的声音」，所以 YouTube / ニコニコ / X / B 站 / 本地播放器
全部通用，不需要挑平台。

实现说明：sounddevice（PortAudio）**不提供** WASAPI 环回能力，所以这里用 soundcard
直接调 WASAPI 来抓扬声器输出。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import AudioConfig

logger = logging.getLogger(__name__)

__all__ = [
    "LoopbackCapture",
    "DeviceInfo",
    "AudioError",
    "list_devices",
    "resolve_capture_device",
    "resample_to",
]


class AudioError(RuntimeError):
    """音频设备打开或解析失败。"""


def _soundcard():  # noqa: ANN201 - 返回 soundcard 模块
    try:
        import soundcard as sc
    except ImportError as exc:
        raise AudioError(
            "缺少 soundcard 库（WASAPI 环回靠它实现）。装一下：\n"
            '    .venv\\Scripts\\python.exe -m pip install soundcard'
        ) from exc
    return sc


@dataclass
class DeviceInfo:
    name: str
    channels: int
    kind: str  # "speaker" | "microphone"
    is_loopback: bool
    is_default: bool


def _is_loopback(device: object) -> bool:
    if getattr(device, "isloopback", False):
        return True
    return "loopback" in str(getattr(device, "name", "")).lower()


def list_devices() -> list[DeviceInfo]:
    """列出所有播放设备，以及可以拿来抓声音的环回输入设备。"""
    sc = _soundcard()
    out: list[DeviceInfo] = []

    default_id = None
    try:
        default_id = sc.default_speaker().id
    except Exception:  # noqa: BLE001 - 没有默认播放设备时也把列表列出来
        logger.debug("拿不到默认播放设备", exc_info=True)

    for spk in sc.all_speakers():
        out.append(
            DeviceInfo(
                name=str(spk.name),
                channels=int(getattr(spk, "channels", 0) or 0),
                kind="speaker",
                is_loopback=False,
                is_default=(getattr(spk, "id", None) == default_id),
            )
        )

    for mic in sc.all_microphones(include_loopback=True):
        out.append(
            DeviceInfo(
                name=str(mic.name),
                channels=int(getattr(mic, "channels", 0) or 0),
                kind="microphone",
                is_loopback=_is_loopback(mic),
                is_default=False,
            )
        )
    return out


def resolve_capture_device(spec: str = ""):  # noqa: ANN201 - 返回 soundcard 设备对象
    """把配置里的 device 解析成一个「环回输入设备」。

    空字符串 => 系统默认播放设备对应的环回；
    否则按名字做子串匹配（优先匹配环回设备）。
    """
    sc = _soundcard()
    text = (spec or "").strip()

    mics = list(sc.all_microphones(include_loopback=True))
    loopbacks = [m for m in mics if _is_loopback(m)]

    if text:
        wanted = text.lower()
        for pool in (loopbacks, mics):
            hit = next((m for m in pool if wanted in str(m.name).lower()), None)
            if hit is not None:
                if not _is_loopback(hit):
                    raise AudioError(
                        f"“{hit.name}” 是普通录音设备，不是环回设备，抓不到扬声器里的声音。\n"
                        "环回设备的名字里通常带 “(Loopback)”，请重新选一个。"
                    )
                return hit
        raise AudioError(
            f"没找到名字包含 “{text}” 的环回设备。跑 `python -m livetrans --devices` 看可用列表。"
        )

    if not loopbacks:
        raise AudioError(
            "系统里没有任何环回设备。请确认：\n"
            "  1) 有能正常出声的播放设备\n"
            "  2) soundcard 装好了（pip install soundcard）\n"
            "  3) 在 Windows 声音设置里至少启用一个输出设备"
        )

    try:
        speaker_name = str(sc.default_speaker().name).lower()
    except Exception:  # noqa: BLE001
        return loopbacks[0]

    for mic in loopbacks:
        if speaker_name and speaker_name in str(mic.name).lower():
            return mic
    return loopbacks[0]


def _to_mono(data: np.ndarray) -> np.ndarray:
    """soundcard 的 record() 返回 (frames, channels)，这里降成单声道 float32。"""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1:
        return np.ascontiguousarray(arr[:, 0])
    return arr.mean(axis=1, dtype=np.float32)


def resample_to(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """把单声道 float32 音频重采样到目标采样率。

    整数倍降采样走盒式平均（自带抗混叠），其余情况走线性插值。块很小，够语音识别用。
    """
    if src_rate == dst_rate or audio.size == 0:
        return np.asarray(audio, dtype=np.float32)

    if src_rate > dst_rate and src_rate % dst_rate == 0:
        factor = src_rate // dst_rate
        usable = (audio.size // factor) * factor
        if usable == 0:
            return np.zeros(0, dtype=np.float32)
        return audio[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32, copy=False)

    n_out = int(round(audio.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.arange(audio.size, dtype=np.float64)
    dst_idx = np.arange(n_out, dtype=np.float64) * (src_rate / dst_rate)
    return np.interp(dst_idx, src_idx, audio).astype(np.float32)


class LoopbackCapture:
    """后台线程持续抓扬声器输出，按目标采样率产出单声道 float32 块。

    用法::

        cap = LoopbackCapture(cfg.audio)
        cap.start()                     # 打不开会抛 AudioError
        block = cap.read(timeout=1.0)   # np.ndarray 或 None
        cap.stop()
    """

    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self.device_name: str = ""
        self.native_rate: int = 0
        self.target_rate: int = int(cfg.sample_rate)
        self.channels: int = 1
        self.dropped_blocks: int = 0
        self.status_messages: list[str] = []

        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str = ""

    # ---- 生命周期 --------------------------------------------------------

    def start(self, timeout: float = 10.0) -> None:
        self._stop.clear()
        self._ready.clear()
        self._error = ""

        self._thread = threading.Thread(target=self._run, name="audio-capture", daemon=True)
        self._thread.start()

        # 等后台线程真的把设备打开，别让 start() 静默成功
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._ready.is_set() or self._error:
                break
            time.sleep(0.02)

        if self._error:
            raise AudioError(self._error)
        if not self._ready.is_set():
            raise AudioError(f"打开环回设备超时（{timeout:.0f}s）。设备：{self.device_name or '未确定'}")

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=3.0)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # ---- 采集线程 --------------------------------------------------------

    def _blocksize(self, rate: int) -> int:
        return max(64, int(rate * max(0.01, self.cfg.block_seconds)))

    def _run(self) -> None:
        try:
            mic = resolve_capture_device(self.cfg.device)
            self.device_name = str(mic.name)
            self.channels = int(getattr(mic, "channels", 1) or 1)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self._ready.set()
            return

        # 16k 是 whisper 想要的；不支持就退到常见采样率，再自己重采样
        rates = [self.target_rate]
        for fallback in (48000, 44100):
            if fallback != self.target_rate:
                rates.append(fallback)

        last_error: Exception | None = None
        for rate in rates:
            try:
                # channels=None 表示录设备的全部通道：soundcard 在 WASAPI 下只录单声道会拿到垃圾数据，
                # 所以录多声道再自己降成单声道。blocksize 要比 numframes 大，否则容易欠载丢帧。
                with mic.recorder(
                    samplerate=rate, channels=None, blocksize=self._blocksize(rate) * 2
                ) as rec:
                    self.native_rate = rate
                    self._ready.set()
                    if rate != self.target_rate:
                        self.status_messages.append(
                            f"设备不支持 {self.target_rate} Hz，改用 {rate} Hz 并在本地重采样"
                        )
                    self._pump(rec, rate)
                return
            except Exception as exc:  # noqa: BLE001 - 逐个采样率降级
                last_error = exc
                self._ready.clear()
                logger.warning("以 %s Hz 打开环回失败：%s", rate, exc)

        self._error = (
            f"打不开环回设备 {self.device_name!r}（试过 {rates} Hz）。\n"
            f"最后一次错误：{last_error}"
        )
        self._ready.set()

    def _pump(self, rec, rate: int) -> None:  # noqa: ANN001
        numframes = self._blocksize(rate)
        while not self._stop.is_set():
            data = rec.record(numframes=numframes)
            mono = _to_mono(data)
            if mono.size == 0:
                continue
            if self.cfg.gain != 1.0:
                mono = mono * self.cfg.gain
                np.clip(mono, -1.0, 1.0, out=mono)
            self._offer(resample_to(mono, rate, self.target_rate))

    def _offer(self, chunk: np.ndarray) -> None:
        try:
            self._queue.put_nowait(chunk)
            return
        except queue.Full:
            pass
        # 消费端跟不上就丢最旧的一块：实时字幕宁可漏，也不要越积越延迟
        try:
            self._queue.get_nowait()
            self._queue.put_nowait(chunk)
        except (queue.Empty, queue.Full):
            pass
        self.dropped_blocks += 1

    # ---- 数据通路 --------------------------------------------------------

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """取一块音频。超时返回 None。"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __enter__(self) -> "LoopbackCapture":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
