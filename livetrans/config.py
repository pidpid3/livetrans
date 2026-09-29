"""配置加载：TOML 文件 + 环境变量覆盖。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_NAME = "config.toml"


@dataclass
class AudioConfig:
    # 空字符串 = 使用默认播放设备的环回（WASAPI loopback）
    device: str = ""
    sample_rate: int = 16000
    # 每次从音频流读取的块长（秒）。越小延迟越低、CPU 唤醒越频繁。
    block_seconds: float = 0.1
    # 输入增益，环回音量太小的时候可以调大
    gain: float = 1.0


@dataclass
class VadConfig:
    # 语音概率阈值（0~1）
    threshold: float = 0.5
    # 连续多少毫秒语音才认为“开始说话”
    min_speech_ms: int = 160
    # 连续多少毫秒静音才认为“说完一句”。调大 = 更愿意把停顿算作句内，句子更完整
    min_silence_ms: int = 700
    # 一句话最长多少秒，超过就强制切段送识别
    max_segment_seconds: float = 12.0
    # 每段语音前后各保留多少毫秒，避免吃掉字头字尾
    padding_ms: int = 240
    # 比这个还短的碎片直接丢掉（咳嗽、键盘、鼠标声）
    min_segment_seconds: float = 0.4
    # 短于这个秒数的语音段先攒着不送，等下一句接上再一起翻译。
    # 这是不把一句话切碎的关键：主播换气、思考的停顿不该算句末。
    merge_seconds: float = 2.5
    # 攒着的短句最多再等多少毫秒，超过就认定确实说完了，单独送出去
    hold_ms: int = 1200


@dataclass
class AsrConfig:
    # faster-whisper 模型名或本地目录；推荐 large-v3-turbo / large-v3 / medium
    model: str = "large-v3-turbo"
    # auto | cuda | cpu
    device: str = "auto"
    # auto | float16 | int8_float16 | int8 | float32
    compute_type: str = "auto"
    language: str = "ja"
    beam_size: int = 1
    # 塞给 whisper 的提示词，可以用来固定人名、术语的写法
    initial_prompt: str = ""
    # 国内网络可设为 https://hf-mirror.com
    hf_endpoint: str = ""
    # 同时最多排队多少个待识别片段，音频跟不上时丢最旧的
    max_queue: int = 8


@dataclass
class TranslateConfig:
    # openai = 任何 OpenAI 兼容接口（DeepSeek/OpenAI/Kimi/智谱/通义/Grok/Mistral/Groq/
    #          OpenRouter/硅基流动/火山方舟/Azure，以及本地 ollama、vLLM、LM Studio）
    # anthropic = Anthropic Claude（接口格式和 OpenAI 不同，单独实现）
    # gemini = Google Gemini
    # none = 不翻译，只显示日文原文
    provider: str = "openai"
    # 目标语言，写清楚一点对翻译质量有帮助
    target_language: str = "简体中文"
    # 留空则用 provider 的默认接口地址（DeepSeek / Gemini 等）
    base_url: str = ""
    # 留空则用 provider 的默认模型
    model: str = ""
    # 直接写 key（不推荐提交到 git）；留空则从 api_key_env 指定的环境变量读
    api_key: str = ""
    api_key_env: str = "LIVETRANS_API_KEY"
    timeout: float = 20.0
    # 带上前面几句原文作为上下文，翻译更连贯
    context_lines: int = 2
    # 额外追加到系统提示词后面的内容
    extra_prompt: str = ""
    # 失败重试次数
    retries: int = 2


@dataclass
class UiConfig:
    font_size: int = 26
    original_font_size: int = 18
    # 0~1，字幕板背景不透明度
    opacity: float = 0.85
    # 字幕窗宽度占屏幕宽度的比例
    width_ratio: float = 0.86
    show_original: bool = True
    # 鼠标穿透：开启后点击会穿透到下面的直播窗口
    click_through: bool = True
    # bottom | top
    position: str = "bottom"
    # 屏幕底部/顶部留白（像素）
    margin: int = 80
    # 同屏保留多少行字幕
    max_lines: int = 4


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    ui: UiConfig = field(default_factory=UiConfig)

    def resolve_api_key(self) -> str:
        """配置里写了就用配置的，否则读环境变量。"""
        if self.translate.api_key:
            return self.translate.api_key
        return os.environ.get(self.translate.api_key_env, "")


_SECTIONS: dict[str, type] = {
    "audio": AudioConfig,
    "vad": VadConfig,
    "asr": AsrConfig,
    "translate": TranslateConfig,
    "ui": UiConfig,
}


def _build_section(section_name: str, cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"[{section_name}] 存在未知配置项: {', '.join(sorted(unknown))}")
    return cls(**data)


def _coerce(raw: str, current: Any) -> Any:
    """按字段当前值的类型转换环境变量字符串。"""
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _apply_env_overrides(cfg: Config) -> None:
    """支持 LIVETRANS_<段名>_<字段名> 覆盖单项配置，方便临时切换不改文件。"""
    for section_name, cls in _SECTIONS.items():
        section = getattr(cfg, section_name)
        for f in fields(cls):
            env_key = f"LIVETRANS_{section_name.upper()}_{f.name.upper()}"
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            setattr(section, f.name, _coerce(raw, getattr(section, f.name)))


def load_config(path: str | Path | None = None) -> Config:
    """读取 TOML 配置。path 为 None 时在项目目录找 config.toml，找不到就用默认值。"""
    cfg = Config()

    candidate = Path(path) if path else Path.cwd() / DEFAULT_CONFIG_NAME
    if candidate.is_file():
        with candidate.open("rb") as fh:
            raw = tomllib.load(fh)
        unknown = set(raw) - set(_SECTIONS)
        if unknown:
            raise ValueError(f"配置文件中存在未知的段: {', '.join(sorted(unknown))}")
        for name, cls in _SECTIONS.items():
            if name in raw:
                setattr(cfg, name, _build_section(name, cls, raw[name]))

    _apply_env_overrides(cfg)
    return cfg
