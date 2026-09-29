"""测试包。

这里处理两件影响「跑测试观感」的事：

* Windows 控制台默认不是 UTF-8，unittest 直接往 stdout 写，中文会变乱码；
* IsolatedAsyncioTestCase 会刷一堆 asyncio 慢回调警告，把结果淹没。

编码部分的逻辑和 livetrans.main._fix_console_encoding 一致。
"""

from __future__ import annotations

import logging
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 某些环境（重定向到文件）不支持就算了
        pass


class _DropSlowCallbackNoise(logging.Filter):
    """滤掉 asyncio 的「Executing <Task ...> took ... seconds」。

    IsolatedAsyncioTestCase 默认开 asyncio 的 debug 模式，每个超过 100ms 的回调都会
    打一条这种 warning，把测试结果淹没。它属于事件循环层面的诊断，正常跑业务代码时不会
    出现，而测试里没人在意毫秒级调度延迟 —— 所以只滤这一种，其余 asyncio 警告照常输出。

    （试过调 loop.slow_callback_duration，也试过猴补丁 unittest 的 `_setupAsyncioRunner`：
      后者是私有方法，3.11 和 3.12 之间签名就不一样（3.12 起它没有返回值，靠副作用给
      self._asyncioRunner 赋值），照老写法打补丁会让它崩在 tearDown 的 runner.close() 上。）
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (message.startswith("Executing ") and "took" in message)


logging.getLogger("asyncio").addFilter(_DropSlowCallbackNoise())
