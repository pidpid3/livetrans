"""设备诊断 CLI：`python -m livetrans --devices`

列出播放设备和环回输入，然后真的录几秒钟，报告有没有抓到声音。排障先跑这个。
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .audio import AudioError, LoopbackCapture, list_devices, resolve_capture_device
from .config import load_config


def _bar(level: float, width: int = 28) -> str:
    filled = int(max(0.0, min(1.0, level)) * width)
    return "█" * filled + "·" * (width - filled)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    seconds = float(argv[0]) if argv else 5.0

    try:
        devices = list_devices()
    except AudioError as exc:
        print(f"✗ {exc}")
        return 1

    speakers = [d for d in devices if d.kind == "speaker"]
    loopbacks = [d for d in devices if d.kind == "microphone" and d.is_loopback]
    plain_mics = [d for d in devices if d.kind == "microphone" and not d.is_loopback]

    print("播放设备：")
    for dev in speakers:
        print(f"  {'★默认' if dev.is_default else '     '}  {dev.name}  ({dev.channels}ch)")
    if not speakers:
        print("  （一个都没有）")

    print("\n环回输入（能抓到扬声器声音，就是用它）：")
    for dev in loopbacks:
        print(f"         {dev.name}  ({dev.channels}ch)")
    if not loopbacks:
        print("  （一个都没有 —— 环回不可用，见下面提示）")

    if plain_mics:
        print("\n普通麦克风（抓不到直播声音，别选这些）：")
        for dev in plain_mics:
            print(f"         {dev.name}  ({dev.channels}ch)")

    cfg = load_config()
    print()
    try:
        mic = resolve_capture_device(cfg.audio.device)
    except AudioError as exc:
        print(f"✗ 选择捕获设备失败：{exc}")
        return 1

    print(f"将要捕获：{mic.name}")
    print(f"开始录 {seconds:.0f} 秒 —— 现在请让直播发出声音（人声、音乐都行）。\n")

    cap = LoopbackCapture(cfg.audio)
    try:
        cap.start()
    except AudioError as exc:
        print(f"✗ 打开环回录音失败：\n{exc}")
        return 1

    print(f"实际参数：{cap.native_rate} Hz → {cap.target_rate} Hz，声道 {cap.channels}")
    for msg in cap.status_messages:
        print(f"⚠ {msg}")

    peak = 0.0
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            block = cap.read(timeout=0.5)
            if block is None or block.size == 0:
                continue
            peak = max(peak, float(np.max(np.abs(block))))
            rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
            print(f"\r电平 {_bar(rms * 4):<28} rms={rms:.4f} peak={peak:.4f}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()

    print()
    if cap.dropped_blocks:
        print(f"⚠ 丢块 {cap.dropped_blocks} 次（消费端太慢，可调大 [audio] block_seconds）")

    if peak < 1e-4:
        print("\n✗ 没抓到任何声音。按顺序检查：")
        print("  1) 直播窗口的声音是不是从这个设备出的 —— 右键任务栏音量图标")
        print("     →「音量合成器」，看它是哪个输出设备")
        print("  2) 系统音量和播放器音量是否静音 / 过小")
        print("  3) 在 config.toml 的 [audio] device 里直接写上面列出的环回设备名的一部分")
        print("  4) 蓝牙耳机切到免手操作(HFP)模式时环回会失效，换 A2DP 或用有线设备验证")
        return 1

    print(f"\n✓ 捕获正常，峰值 {peak:.4f}。下一步：python -m livetrans --terminal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
