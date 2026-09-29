"""桌面悬浮字幕窗（PySide6）。

一个无边框、置顶、半透明的字幕板，默认鼠标穿透，看全屏直播时直接浮在上面。

托盘菜单里能直接调两类东西：

* **外观** —— 字号、背景透明度、宽度、同屏行数、位置、是否显示日文原文
* **VAD** —— 断句快慢、攒句阈值、攒句等待、语音灵敏度

两类都是改完立即生效，并记到 `.livetrans_state.json`，下次启动自动恢复。
VAD 那几项会直接作用到正在跑的分段器上，**不用重启，也不用重新加载模型**。
非穿透模式下还能拖着挪位置、滚轮调字号。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMenu,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from .config import Config, UiConfig, VadConfig
from .pipeline import SubtitleLine

logger = logging.getLogger(__name__)

__all__ = ["UiBridge", "SubtitleWindow", "run_ui"]

_ZH_COLOR = "#ffffff"
_JA_COLOR = "#cfd6dd"
_STATUS_COLOR = "#8fa4b8"
_ERROR_COLOR = "#ff8a8a"

_FONTS = ["Microsoft YaHei UI", "Microsoft JhengHei UI", "Yu Gothic UI", "Meiryo", "Segoe UI"]

# 状态记忆文件（跟着启动目录走）；_LEGACY 是老版本只存外观时用的名字
_STATE_FILE = ".livetrans_state.json"
_LEGACY_STATE_FILE = ".livetrans_ui.json"

_UI_FIELDS = (
    "font_size",
    "original_font_size",
    "opacity",
    "width_ratio",
    "show_original",
    "click_through",
    "position",
    "margin",
    "max_lines",
)

# 托盘里能调的 VAD 参数（其余参数仍然走 config.toml）
_VAD_FIELDS = (
    "min_silence_ms",
    "merge_seconds",
    "hold_ms",
    "threshold",
)

_MIN_FONT = 12
_MAX_FONT = 72
# 日文原文字号 = 中文字号 × 这个比例
_LINE_FONT_RATIO = 0.7

# (字段, 菜单标题, 调小的按钮, 调大的按钮, 步进, 下限, 上限)
_VAD_MENU_SPEC = (
    ("min_silence_ms", "断句快慢", "更快", "更慢", 150, 200, 2500),
    ("merge_seconds", "攒句阈值", "更短", "更长", 0.5, 0.0, 8.0),
    ("hold_ms", "攒句等待", "更短", "更长", 200, 200, 3000),
    ("threshold", "语音灵敏度", "更灵敏", "更保守", 0.05, 0.2, 0.9),
)


class UiBridge(QObject):
    """工作线程 → UI 线程的桥。所有跨线程更新都走 Qt 信号。"""

    line = Signal(object)
    status = Signal(str)
    error = Signal(str)
    level = Signal(float)


# ---- 状态记忆 ------------------------------------------------------------


def _state_path() -> Path:
    return Path.cwd() / _STATE_FILE


def _read_state() -> dict:
    for name in (_STATE_FILE, _LEGACY_STATE_FILE):
        path = Path.cwd() / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 记忆文件坏掉不该拦住启动
            logger.warning("状态文件读不出来，忽略：%s", path, exc_info=True)
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _save_state(ui: UiConfig, vad: VadConfig) -> None:
    data: dict = {name: getattr(ui, name) for name in _UI_FIELDS}
    data["vad"] = {name: getattr(vad, name) for name in _VAD_FIELDS}
    try:
        _state_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        logger.warning("保存设置失败", exc_info=True)


def _load_state(ui: UiConfig, vad: VadConfig | None = None) -> None:
    """把上次调好的东西套回配置上。没有文件、或文件坏了，都静默跳过。"""
    data = _read_state()
    if not data:
        return
    for name in _UI_FIELDS:
        if name in data:
            setattr(ui, name, data[name])
    vad_data = data.get("vad")
    if vad is not None and isinstance(vad_data, dict):
        for name in _VAD_FIELDS:
            if name in vad_data:
                setattr(vad, name, vad_data[name])
    logger.info("已恢复上次调好的设置")


def _make_icon() -> QIcon:
    """代码画一个托盘图标，省得带资源文件。"""
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(28, 32, 40, 235))
    painter.setPen(QColor(120, 200, 255, 255))
    painter.drawRoundedRect(4, 4, 56, 56, 14, 14)
    painter.setPen(QColor(235, 245, 255))
    font = QFont("Segoe UI", 26, QFont.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "译")
    painter.end()
    return QIcon(pixmap)


class SubtitleWindow(QWidget):
    def __init__(self, cfg: Config, bridge: UiBridge, on_vad_change=None) -> None:  # noqa: ANN001
        super().__init__(None)

        self.vad = cfg.vad
        self._on_vad_change = on_vad_change

        # 先记下 config.toml 里的原始值，菜单里的「恢复默认」回到这里
        self._base_ui = {name: getattr(cfg.ui, name) for name in _UI_FIELDS}
        self._base_vad = {name: getattr(cfg.vad, name) for name in _VAD_FIELDS}
        _load_state(cfg.ui, cfg.vad)

        self.cfg = cfg.ui
        self.bridge = bridge

        self._rows: list[QWidget] = []
        self._history: list[SubtitleLine] = []
        self._drag_offset = None
        self._screen = QApplication.primaryScreen()
        self._custom_pos = None
        self._vad_menus: dict[str, tuple[QMenu, str]] = {}

        # 连续调整（比如滚轮）时不每次都落盘
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(lambda: _save_state(self.cfg, self.vad))

        self.setWindowTitle("livetrans 直播翻译")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._apply_flags()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._panel = QFrame(self)
        self._panel.setObjectName("panel")
        outer.addWidget(self._panel)

        panel_layout = QVBoxLayout(self._panel)
        panel_layout.setContentsMargins(22, 12, 22, 14)
        panel_layout.setSpacing(6)

        self._status = QLabel("正在启动…", self._panel)
        self._status.setWordWrap(True)
        self._status.setFont(self._font(13))
        self._status.setStyleSheet(f"color: {_STATUS_COLOR};")
        panel_layout.addWidget(self._status)

        self._lines_box = QVBoxLayout()
        self._lines_box.setContentsMargins(0, 0, 0, 0)
        self._lines_box.setSpacing(8)
        panel_layout.addLayout(self._lines_box)

        self._zh_font = self._font(self.cfg.font_size, bold=True)
        self._ja_font = self._font(self.cfg.original_font_size)
        self._err_font = self._font(max(10, self.cfg.original_font_size - 4))

        self._refresh_panel_style()

        self.bridge.line.connect(self._on_line)
        self.bridge.status.connect(self._on_status)
        self.bridge.error.connect(self._on_error)

        self._tray = self._build_tray()
        self._reposition()

    # ---- 外观 ------------------------------------------------------------

    @staticmethod
    def _font(size: int, bold: bool = False) -> QFont:
        font = QFont()
        font.setFamilies(_FONTS)
        font.setPixelSize(size)
        font.setBold(bold)
        return font

    def _refresh_fonts(self) -> None:
        self._zh_font = self._font(self.cfg.font_size, bold=True)
        self._ja_font = self._font(self.cfg.original_font_size)
        self._err_font = self._font(max(10, self.cfg.original_font_size - 4))

    def _refresh_panel_style(self) -> None:
        alpha = int(max(0.0, min(1.0, self.cfg.opacity)) * 255)
        self._panel.setStyleSheet(
            f"""
            QFrame#panel {{
                background-color: rgba(10, 13, 18, {alpha});
                border-radius: 14px;
            }}
            """
        )

    def _apply_flags(self) -> None:
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        if self.cfg.click_through:
            flags |= Qt.WindowTransparentForInput
        self.setWindowFlags(flags)

    def _reposition(self) -> None:
        if self._screen is None:
            return
        area = self._screen.availableGeometry()

        width = int(area.width() * max(0.2, min(1.0, self.cfg.width_ratio)))
        self.setFixedWidth(width)

        hint = self.sizeHint()
        height = max(60, min(hint.height(), int(area.height() * 0.6)))
        self.resize(width, height)

        x = area.x() + (area.width() - width) // 2
        if self._custom_pos is not None:
            x, y = self._custom_pos
        elif self.cfg.position == "top":
            y = area.y() + self.cfg.margin
        else:
            y = area.y() + area.height() - height - self.cfg.margin

        # 夹回屏幕内，避免改配置后窗口跑到屏幕外
        x = max(area.x(), min(x, area.x() + area.width() - width))
        y = max(area.y(), min(y, area.y() + area.height() - height))
        self.move(x, y)

    # ---- 内容更新 --------------------------------------------------------

    def _on_status(self, message: str) -> None:
        logger.info("状态：%s", message)
        self._status.setText(message)
        self._status.setStyleSheet(
            f"color: {_ERROR_COLOR if message.startswith(('✗', '⚠')) else _STATUS_COLOR};"
        )
        self._reposition()

    def _on_error(self, message: str) -> None:
        logger.error("错误：%s", message)
        self._status.setText(f"✗ {message}")
        self._status.setStyleSheet(f"color: {_ERROR_COLOR};")
        self._reposition()

    def _on_line(self, line: SubtitleLine) -> None:
        logger.info("[%.2fs] %s → %s", line.total_latency, line.ja, line.zh or "(未翻译)")
        self._status.setText(
            f"识别 {line.asr_latency * 1000:.0f}ms · 翻译 {line.translate_latency * 1000:.0f}ms "
            f"· 音频 {line.audio_seconds:.1f}s"
        )
        self._status.setStyleSheet(f"color: {_STATUS_COLOR};")

        self._history.append(line)
        if len(self._history) > 50:
            self._history = self._history[-50:]

        self._add_row(line)
        self._trim_rows()
        self._reposition()

    def _add_row(self, line: SubtitleLine) -> None:
        row = QWidget(self._panel)
        box = QVBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)

        if line.zh:
            if self.cfg.show_original and line.ja:
                box.addWidget(self._label(line.ja, self._ja_font, _JA_COLOR))
            box.addWidget(self._label(line.zh, self._zh_font, _ZH_COLOR))
        elif line.ja:
            box.addWidget(self._label(line.ja, self._zh_font, _ZH_COLOR))

        if line.error:
            box.addWidget(self._label(f"⚠ {line.error}", self._err_font, _ERROR_COLOR))

        self._rows.append(row)
        self._lines_box.addWidget(row)

    def _trim_rows(self) -> None:
        while len(self._rows) > max(1, self.cfg.max_lines):
            old = self._rows.pop(0)
            self._lines_box.removeWidget(old)
            old.deleteLater()

    def _label(self, text: str, font: QFont, color: str) -> QLabel:
        label = QLabel(text, self._panel)
        label.setWordWrap(True)
        label.setFont(font)
        label.setStyleSheet(f"color: {color}; background: transparent;")
        label.setTextInteractionFlags(Qt.NoTextInteraction)
        return label

    def _drop_rows(self) -> None:
        for row in self._rows:
            self._lines_box.removeWidget(row)
            row.deleteLater()
        self._rows = []

    def _rebuild_lines(self) -> None:
        """外观变了（字号、行数、是否显示原文）之后，按新设置重画已有字幕。"""
        self._drop_rows()
        for line in self._history[-max(1, self.cfg.max_lines) :]:
            self._add_row(line)
        self._reposition()

    def clear_lines(self) -> None:
        self._history = []
        self._drop_rows()
        self._reposition()

    # ---- 运行时可调的外观 -------------------------------------------------

    def _save(self) -> None:
        self._save_timer.start()

    def _change_font(self, delta: int = 0, absolute: int | None = None) -> None:
        base = absolute if absolute is not None else self.cfg.font_size + delta
        new_size = max(_MIN_FONT, min(_MAX_FONT, base))
        if new_size == self.cfg.font_size and absolute is None:
            return
        self.cfg.font_size = new_size
        self.cfg.original_font_size = max(10, int(round(new_size * _LINE_FONT_RATIO)))
        self._refresh_fonts()
        self._rebuild_lines()
        self._save()

    def _change_opacity(self, delta: float = 0.0, absolute: float | None = None) -> None:
        base = absolute if absolute is not None else self.cfg.opacity + delta
        self.cfg.opacity = max(0.0, min(1.0, round(base, 2)))
        self._refresh_panel_style()
        self._save()

    def _change_width(self, delta: float = 0.0, absolute: float | None = None) -> None:
        base = absolute if absolute is not None else self.cfg.width_ratio + delta
        self.cfg.width_ratio = max(0.3, min(1.0, round(base, 2)))
        self._custom_pos = None
        self._reposition()
        self._save()

    def _change_lines(self, delta: int = 0, absolute: int | None = None) -> None:
        base = absolute if absolute is not None else self.cfg.max_lines + delta
        self.cfg.max_lines = max(1, min(10, base))
        self._rebuild_lines()
        self._save()

    def _toggle_original(self, checked: bool) -> None:
        self.cfg.show_original = bool(checked)
        self._rebuild_lines()
        self._save()

    def _set_position(self, position: str) -> None:
        self.cfg.position = position
        self._custom_pos = None
        self._reposition()
        self._sync_tray_actions()
        self._save()

    def _reset_position(self) -> None:
        self._custom_pos = None
        self._reposition()
        self._save()

    def _reset_appearance(self) -> None:
        for name in _UI_FIELDS:
            setattr(self.cfg, name, self._base_ui[name])
        self._refresh_fonts()
        self._refresh_panel_style()
        self._apply_flags()
        self.show()
        self._sync_tray_actions()
        self._rebuild_lines()
        self._save()

    # ---- 运行时可调的 VAD -------------------------------------------------

    @staticmethod
    def _format_vad(name: str, value) -> str:  # noqa: ANN001
        if name.endswith("_ms"):
            return f"{int(value)}ms"
        if name == "threshold":
            return f"{float(value):.2f}"
        return f"{float(value):g}s"

    def _change_vad(self, name: str, delta: float, low: float, high: float) -> None:
        current = getattr(self.vad, name)
        target = max(low, min(high, current + delta))
        new_value = int(round(target)) if isinstance(current, int) else round(target, 2)
        if new_value == current:
            return
        setattr(self.vad, name, new_value)
        self._after_vad_change()

    def _reset_vad(self, name: str) -> None:
        if getattr(self.vad, name) == self._base_vad[name]:
            return
        setattr(self.vad, name, self._base_vad[name])
        self._after_vad_change()

    def _reset_vad_all(self) -> None:
        for name in _VAD_FIELDS:
            setattr(self.vad, name, self._base_vad[name])
        self._after_vad_change()

    def _after_vad_change(self) -> None:
        """改完 VAD 要做三件事：作用到分段器、刷新菜单上的当前值、记盘。"""
        if self._on_vad_change is not None:
            try:
                self._on_vad_change()
            except Exception:  # noqa: BLE001 - 分段器还没起来也不该崩
                logger.warning("把 VAD 参数应用到分段器时出错", exc_info=True)
        self._sync_tray_actions()
        self._save()

    # ---- 托盘与交互 ------------------------------------------------------

    def _build_tray(self) -> QSystemTrayIcon | None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("系统托盘不可用，没法用托盘调整；请用 Ctrl+C 结束进程")
            return None

        tray = QSystemTrayIcon(_make_icon(), self)
        tray.setToolTip("livetrans 直播翻译")

        menu = QMenu()

        toggle_visible = QAction("显示 / 隐藏字幕", menu)
        toggle_visible.triggered.connect(lambda: self.setVisible(not self.isVisible()))
        menu.addAction(toggle_visible)
        menu.addSeparator()

        size_menu = menu.addMenu("字幕大小")
        self._pair(size_menu, "更大", lambda: self._change_font(2), "更小", lambda: self._change_font(-2))
        size_menu.addSeparator()
        size_default = QAction("恢复默认", size_menu)
        size_default.triggered.connect(lambda: self._change_font(absolute=self._base_ui["font_size"]))
        size_menu.addAction(size_default)

        opacity_menu = menu.addMenu("背景深浅")
        self._pair(
            opacity_menu,
            "更明显", lambda: self._change_opacity(0.1),
            "更透明", lambda: self._change_opacity(-0.1),
        )
        opacity_menu.addSeparator()
        opacity_default = QAction("恢复默认", opacity_menu)
        opacity_default.triggered.connect(lambda: self._change_opacity(absolute=self._base_ui["opacity"]))
        opacity_menu.addAction(opacity_default)

        width_menu = menu.addMenu("字幕宽度")
        self._pair(width_menu, "更宽", lambda: self._change_width(0.06), "更窄", lambda: self._change_width(-0.06))
        width_menu.addSeparator()
        width_default = QAction("恢复默认", width_menu)
        width_default.triggered.connect(lambda: self._change_width(absolute=self._base_ui["width_ratio"]))
        width_menu.addAction(width_default)

        lines_menu = menu.addMenu("同屏行数")
        self._pair(lines_menu, "多一行", lambda: self._change_lines(1), "少一行", lambda: self._change_lines(-1))
        lines_menu.addSeparator()
        lines_default = QAction("恢复默认", lines_menu)
        lines_default.triggered.connect(lambda: self._change_lines(absolute=self._base_ui["max_lines"]))
        lines_menu.addAction(lines_default)

        self._original_action = QAction("显示日文原文", menu)
        self._original_action.setCheckable(True)
        self._original_action.setChecked(self.cfg.show_original)
        self._original_action.triggered.connect(self._toggle_original)
        menu.addAction(self._original_action)

        position_menu = menu.addMenu("位置")
        self._position_group = QActionGroup(position_menu)
        self._position_group.setExclusive(True)
        for label, value in (("贴屏幕底部", "bottom"), ("贴屏幕顶部", "top")):
            action = QAction(label, position_menu)
            action.setCheckable(True)
            action.setChecked(self.cfg.position == value)
            action.triggered.connect(lambda _checked=False, v=value: self._set_position(v))
            self._position_group.addAction(action)
            position_menu.addAction(action)
        position_menu.addSeparator()
        reset_pos = QAction("重置到默认位置", position_menu)
        reset_pos.triggered.connect(self._reset_position)
        position_menu.addAction(reset_pos)

        menu.addSeparator()

        # ---- VAD 调节：改完立即作用到分段器，不用重启也不用重载模型 ----
        vad_menu = menu.addMenu("VAD 调节")
        for name, label, down_label, up_label, step, low, high in _VAD_MENU_SPEC:
            sub = vad_menu.addMenu(label)
            self._vad_menus[name] = (sub, label)
            self._pair(
                sub,
                down_label,
                lambda n=name, s=step, lo=low, hi=high: self._change_vad(n, -s, lo, hi),
                up_label,
                lambda n=name, s=step, lo=low, hi=high: self._change_vad(n, s, lo, hi),
            )
            sub.addSeparator()
            reset_one = QAction("恢复默认", sub)
            reset_one.triggered.connect(lambda _checked=False, n=name: self._reset_vad(n))
            sub.addAction(reset_one)

        vad_menu.addSeparator()
        vad_reset_all = QAction("全部恢复默认", vad_menu)
        vad_reset_all.triggered.connect(self._reset_vad_all)
        vad_menu.addAction(vad_reset_all)

        menu.addSeparator()

        self._through_action = QAction("鼠标穿透", menu)
        self._through_action.setCheckable(True)
        self._through_action.setChecked(self.cfg.click_through)
        self._through_action.triggered.connect(self.set_click_through)
        menu.addAction(self._through_action)

        clear = QAction("清空字幕", menu)
        clear.triggered.connect(self.clear_lines)
        menu.addAction(clear)

        reset_appearance = QAction("恢复默认外观", menu)
        reset_appearance.triggered.connect(self._reset_appearance)
        menu.addAction(reset_appearance)

        menu.addSeparator()
        quit_action = QAction("退出", menu)
        quit_action.triggered.connect(QApplication.quit)
        menu.addAction(quit_action)

        tray.setContextMenu(menu)
        tray.activated.connect(self._on_tray_activated)
        tray.show()
        self._sync_tray_actions()
        return tray

    @staticmethod
    def _pair(menu: QMenu, first_label: str, first_slot, second_label: str, second_slot) -> None:  # noqa: ANN001
        for label, slot in ((first_label, first_slot), (second_label, second_slot)):
            action = QAction(label, menu)
            action.triggered.connect(lambda _checked=False, fn=slot: fn())
            menu.addAction(action)

    def _sync_tray_actions(self) -> None:
        """程序内部改了状态之后，把菜单上的勾选和当前值同步过来。"""
        for name, value in (
            ("_through_action", self.cfg.click_through),
            ("_original_action", self.cfg.show_original),
        ):
            action = getattr(self, name, None)
            if action is not None:
                action.setChecked(value)

        group = getattr(self, "_position_group", None)
        if group is not None:
            for action in group.actions():
                action.setChecked((action.text() == "贴屏幕顶部") == (self.cfg.position == "top"))

        for name, (sub, label) in self._vad_menus.items():
            current = self._format_vad(name, getattr(self.vad, name))
            sub.setTitle(f"{label}（{current}）")

    def _on_tray_activated(self, reason) -> None:  # noqa: ANN001
        if reason == QSystemTrayIcon.DoubleClick:
            self.setVisible(not self.isVisible())

    def set_click_through(self, enabled: bool) -> None:
        self.cfg.click_through = bool(enabled)
        self._apply_flags()
        self.show()
        self._sync_tray_actions()
        self._reposition()
        self._save()

    # 非穿透模式下：拖着挪位置、滚轮调字号
    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self._custom_pos = (event.globalPosition().toPoint() - self._drag_offset).toTuple()
            self.move(*self._custom_pos)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        self._drag_offset = None
        self._save()
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: ANN001
        self._change_font(2 if event.angleDelta().y() > 0 else -2)
        event.accept()


def run_ui(cfg: Config, bridge: UiBridge, on_vad_change=None) -> SubtitleWindow:  # noqa: ANN001
    """创建并显示字幕窗（调用方需要已经有 QApplication）。

    ``on_vad_change`` 在托盘里改完 VAD 参数后调用，把新参数作用到正在跑的分段器上。
    """
    window = SubtitleWindow(cfg, bridge, on_vad_change=on_vad_change)
    window.show()
    return window
