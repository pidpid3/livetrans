"""命令行入口。

    python -m livetrans               # 悬浮字幕窗（默认）
    python -m livetrans --terminal    # 终端滚动字幕，先验证链路
    python -m livetrans --devices     # 音频设备诊断
    python -m livetrans -c my.toml    # 指定配置文件
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import load_config
from .pipeline import Pipeline, PipelineCallbacks

logger = logging.getLogger("livetrans")


# 这些库在 INFO 级别会刷屏（每次 HTTP 请求、每段音频都打一行），非 -v 时压到 WARNING
_NOISY_LOGGERS = (
    "faster_whisper",
    "httpx",
    "httpcore",
    "huggingface_hub",
    "urllib3",
    "filelock",
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


def _fix_console_encoding() -> None:
    """Windows 控制台默认不是 UTF-8，日文会变问号。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def _run_terminal(cfg) -> int:  # noqa: ANN001
    def on_line(line) -> None:  # noqa: ANN001
        print()
        print(f"  日  {line.ja}")
        if line.zh:
            print(f"  中  {line.zh}")
        if line.error:
            print(f"  ⚠   {line.error}")
        print(
            f"      [识别 {line.asr_latency * 1000:.0f}ms · 翻译 {line.translate_latency * 1000:.0f}ms"
            f" · 音频 {line.audio_seconds:.1f}s]"
        )

    callbacks = PipelineCallbacks(
        on_line=on_line,
        on_status=lambda msg: print(f"[状态] {msg}", flush=True),
        on_error=lambda msg: print(f"[错误] {msg}", flush=True),
    )
    pipeline = Pipeline(cfg, callbacks)
    pipeline.start()

    try:
        while not pipeline.stopped:
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n正在停止…")
    finally:
        pipeline.stop()
    return 0


def _run_gui(cfg) -> int:  # noqa: ANN001
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from .ui import UiBridge, run_ui

    app = QApplication(sys.argv[:1])
    app.setApplicationName("livetrans")
    # 关掉字幕窗不退出进程，靠托盘菜单退出
    app.setQuitOnLastWindowClosed(False)

    bridge = UiBridge()
    pipeline = Pipeline(
        cfg,
        PipelineCallbacks(
            on_line=bridge.line.emit,
            on_status=bridge.status.emit,
            on_error=bridge.error.emit,
            on_level=bridge.level.emit,
        ),
    )
    # 托盘里改了 VAD 参数就直接作用到分段器上，立即生效且不用重新加载模型
    window = run_ui(cfg, bridge, on_vad_change=pipeline.apply_vad_config)

    # 先让窗口画出来，再开始加载模型（加载会卡住几百毫秒到几分钟）
    QTimer.singleShot(80, pipeline.start)

    # 空转定时器：让 Qt 事件循环有机会处理 Ctrl+C
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start(200)

    app.aboutToQuit.connect(pipeline.stop)
    try:
        return app.exec()
    except KeyboardInterrupt:
        # 在 Qt 事件循环里按 Ctrl+C 会抛到这里，不打难看的 traceback
        print("\n正在停止…")
        return 0
    finally:
        pipeline.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="livetrans",
        description="日本 VTuber 直播实时翻译字幕（系统音频环回 → 本地识别 → 云端翻译）",
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=None,
        help="配置文件路径，默认读当前目录的 config.toml",
    )
    parser.add_argument(
        "--terminal", action="store_true",
        help="不开字幕窗，直接在终端滚动输出（排查问题时用）",
    )
    parser.add_argument(
        "--devices", nargs="?", const=5.0, type=float, default=None,
        help="列出音频设备并实测环回捕获，可跟一个秒数（默认 5 秒），然后退出",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="只做配置校验和依赖自检，不抓音频",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    _fix_console_encoding()
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    if args.devices is not None:
        from .devices import main as devices_main

        return devices_main([str(args.devices)])

    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 配置读取失败：{exc}", file=sys.stderr)
        return 2

    if args.check:
        return _run_check(cfg)

    if args.terminal:
        return _run_terminal(cfg)
    return _run_gui(cfg)


def _run_check(cfg) -> int:  # noqa: ANN001
    """不碰音频的静态自检，用来确认环境和配置。"""
    ok = True
    print("配置检查")
    print(f"  识别模型    : {cfg.asr.model} (device={cfg.asr.device}, compute={cfg.asr.compute_type})")
    print(f"  翻译后端    : {cfg.translate.provider}")
    print(f"  目标语言    : {cfg.translate.target_language}")
    print(f"  API key     : {'已配置' if cfg.resolve_api_key() else '未配置'}")

    try:
        import numpy  # noqa: F401
        import soundcard  # noqa: F401
        from faster_whisper import WhisperModel  # noqa: F401

        print("  依赖          : numpy / soundcard / faster-whisper 均可用")
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 依赖缺失：{exc}")
        return 2

    try:
        from .translate import create_translator

        translator = create_translator(cfg.translate)
        print(f"  翻译实例    : {translator.name if translator else '未启用'}")
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 翻译后端创建失败：{exc}")
        ok = False

    from .audio import AudioError, resolve_capture_device

    try:
        mic = resolve_capture_device(cfg.audio.device)
        print(f"  捕获设备    : {mic.name}")
    except AudioError as exc:
        print(f"✗ {exc}")
        ok = False

    print("\n✓ 自检通过，可以运行 python -m livetrans" if ok else "\n✗ 自检发现问题，请按上面提示处理")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
