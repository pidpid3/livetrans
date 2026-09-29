"""`livetrans.bat` 双击后用的交互菜单。

菜单为什么放在 Python 里而不是 .bat 里：cmd 解析批处理文件用的是系统 ANSI 代码页
（中文 Windows 上是 GBK），而源码文件是 UTF-8。中文经 GBK 解码后字节数对不齐，
会把批处理的行结构搞乱 —— 实测结果是菜单那几行被当成命令执行，报一堆
「'xxx' 不是内部或外部命令」，`goto` 标签也错位，最后在菜单上无限循环。
放在这里就没这个问题：输出编码由 Python 自己控制。
"""

from __future__ import annotations

import subprocess
import sys

_TITLE = """
==========================================================
    livetrans  —  日本 VTuber 直播实时翻译
==========================================================
"""

# (按键, 说明, 要执行的参数)
_OPTIONS: tuple[tuple[str, str, list[str]], ...] = (
    ("1", "启动悬浮字幕窗", ["-m", "livetrans"]),
    ("2", "终端模式（每句的延迟都打出来）", ["-m", "livetrans", "--terminal"]),
    ("3", "检查音频设备（试抓 8 秒，确认抓得到声音）", ["-m", "livetrans", "--devices", "8"]),
    ("4", "配置与依赖自检（不抓音频）", ["-m", "livetrans", "--check"]),
    ("5", "跑单元测试", ["-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]),
    ("0", "退出", []),
)


def _fix_console_encoding() -> None:
    """和 livetrans.main 里做的一样：Windows 控制台默认不是 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 重定向到文件等场景可能不支持
            pass


def _show_menu() -> None:
    print(_TITLE)
    for key, label, _ in _OPTIONS:
        print(f"   {key}. {label}")
    print()


def _pause() -> bool:
    """等用户回车。返回 False 表示输入结束或按了 Ctrl+C，该退出了。"""
    try:
        input("   —— 回车返回菜单 ——")
        return True
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def main(argv: list[str] | None = None) -> int:
    _fix_console_encoding()

    # 带参数就直通给 livetrans，命令行沿用同一套用法
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        return subprocess.call([sys.executable, "-m", "livetrans", *args])

    table = {key: command for key, _, command in _OPTIONS}
    while True:
        _show_menu()
        try:
            choice = input("   请选择（0-5）：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        command = table.get(choice)
        if command is None:
            continue
        if not command:
            return 0

        print()
        try:
            subprocess.call([sys.executable, *command])
        except KeyboardInterrupt:
            print("\n已中断")
        print()
        if not _pause():
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
