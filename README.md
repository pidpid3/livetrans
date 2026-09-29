# livetrans — 日本 VTuber 直播实时翻译字幕

看日语 VTuber 直播时，在屏幕上浮一个字幕条，实时显示日文原文 + 中文翻译。

- **抓的是系统播放的声音**（WASAPI 环回），所以 YouTube / ニコニコ / X / B站 / 任何播放器都能用，不用挑平台
- **识别在本地跑**（faster-whisper + 你的显卡），音频不出本机
- **只把识别出的文字发给翻译 API**，延迟低、质量好

```
扬声器输出 ──WASAPI 环回──▶ VAD 切句 ──▶ faster-whisper 日语识别 ──▶ 云端 LLM 翻译 ──▶ 桌面字幕窗
                              (Silero)      (本地 GPU)              (DeepSeek/Gemini…)
```

端到端延迟通常 **1.5 ~ 3 秒**（识别 0.3~0.8s + 翻译 0.6~1.5s + VAD 等待断句 0.5s）。

---

## 1. 环境要求

| 项目 | 要求 |
| --- | --- |
| 系统 | Windows 10 / 11（依赖 WASAPI 环回，Linux/macOS 需要改音频采集层） |
| Python | 3.11 ~ 3.12（3.13+ 部分依赖还没有 wheel） |
| 显卡 | 可选。有 NVIDIA 显卡会快很多；没有就用 CPU 跑小模型 |
| 其它 | 一个翻译 API key（DeepSeek / OpenAI / Gemini / 本地 ollama 都行） |

本机环境：RTX 5070 Laptop 8GB、Python 3.12、ffmpeg 8.1.1。

## 2. 安装

```bash
# 建虚拟环境（推荐 uv，没有就用 python -m venv .venv）
uv venv --python 3.12 .venv

# 装依赖
uv pip install --python .venv -r requirements.txt
```

> 音频采集用的是 **soundcard** 而不是 sounddevice —— `sounddevice`（PortAudio）不提供 WASAPI 环回能力，
> 用它只能录音箱外放的声音。soundcard 直接调 WASAPI，才能干净地拿到「扬声器正在播放的声音」。

## 3. 配置 API key

打开 `config.toml`，翻译后端默认是 DeepSeek（中文好、便宜、国内直连）。

**推荐用环境变量放 key，不要写进配置文件**：

```powershell
# PowerShell（当前窗口有效）
$env:LIVETRANS_API_KEY = "sk-xxxxxxxx"
```

```bash
# bash / git bash
export LIVETRANS_API_KEY=sk-xxxxxxxx
```

想永久生效：`setx LIVETRANS_API_KEY "sk-xxxx"`（重开终端后有效）。

### 换别的翻译后端

只有四种 `provider`，但**国内外绝大多数厂商都属于第一类**（都提供 OpenAI 兼容入口），换个 `base_url` + `model` 就行：

| 厂商 | `provider` | `base_url` | 示例 `model` |
| --- | --- | --- | --- |
| DeepSeek（默认） | `openai` | 留空即用 | 留空即用 |
| OpenAI | `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 月之暗面 Kimi | `openai` | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| 智谱 GLM | `openai` | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| 通义千问 | `openai` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| 硅基流动 | `openai` | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` |
| OpenRouter | `openai` | `https://openrouter.ai/api/v1` | `google/gemini-2.0-flash-001` |
| Grok (xAI) | `openai` | `https://api.x.ai/v1` | `grok-2-latest` |
| Groq | `openai` | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Mistral | `openai` | `https://api.mistral.ai/v1` | `mistral-small-latest` |
| 火山方舟 / 豆包 | `openai` | `https://ark.cn-beijing.volces.com/api/v3` | 你的接入点 ID |
| **Anthropic Claude** | `anthropic` | 留空即用 | 留空即用 |
| **Google Gemini** | `gemini` | 留空即用 | 留空即用 |
| 本地 ollama（离线） | `openai` | `http://localhost:11434/v1` | `qwen2.5:7b` |
| 只要日文原文 | `none` | — | — |

```toml
# 写法示例
[translate]
provider = "openai"
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
```

> 碰到不认识的新模型也不用改代码：如果它不接受 `temperature`、或者把 `max_tokens` 换成了
> `max_completion_tokens`，程序会自动去掉被拒的参数重试一次。

## 4. 跑起来

最省事的办法：**双击 `livetrans.bat`** —— 会出一个菜单（启动字幕窗 / 终端模式 / 检查设备 / 自检 / 跑测试）。
带参数时不开菜单，直接透传：

```bat
livetrans.bat --terminal
livetrans.bat --devices 8
```

> ⚠ `livetrans.bat` 里**不要写中文**：cmd 用系统 ANSI 代码页（GBK）解析 .bat，而文件是 UTF-8，
> 中文会让字节流错位、批处理行结构崩坏，菜单行会被当成命令执行。菜单因此放在 `livetrans/menu.py` 里。

不想用 bat 就照旧敲命令。第一步先确认能抓到直播的声音（这一步最重要）：

```bash
.venv\Scripts\python.exe -m livetrans --devices 8
```

播放一段日语视频，看有没有电平条在动。抓到了就继续：

```bash
# 悬浮字幕窗（正常使用）
.venv\Scripts\python.exe -m livetrans

# 终端滚动模式（调试链路用，能直接看到延迟数字）
.venv\Scripts\python.exe -m livetrans --terminal

# 只做配置和依赖自检，不抓音频
.venv\Scripts\python.exe -m livetrans --check
```

**首次运行会下载 whisper 模型**（`large-v3-turbo` 约 1.6GB），国内网络慢的话在 `config.toml` 里解除注释：

```toml
[asr]
hf_endpoint = "https://hf-mirror.com"
```

想先快点确认链路通不通，可以临时换小模型（不用改文件）：

```bash
set LIVETRANS_ASR_MODEL=small
.venv\Scripts\python.exe -m livetrans --terminal
```

### 字幕窗操作

- 默认**鼠标穿透**，点字幕不会挡住下面的直播窗口
- **托盘图标右键**可以实时调整外观，改完立即生效并自动记住：
  - `字幕大小` / `背景深浅` / `字幕宽度` / `同屏行数`（各有「恢复默认」）
  - `显示日文原文` 勾选开关
  - `位置` → 贴屏幕底部 / 贴屏幕顶部 / 重置到默认位置
  - `鼠标穿透` 勾选开关、`清空字幕`、`恢复默认外观`、`退出`
- **想挪位置**：托盘里先关掉「鼠标穿透」，拖动窗口到合适位置（松手记住）
- **临时改字号**：同样关掉穿透后，鼠标滚轮在字幕上滚动即可
- 这些调整存在 `.livetrans_ui.json`，下次启动自动恢复；`config.toml` 里的 `[ui]`
  是出厂默认值，不会被程序改动，「恢复默认」就是回到它

## 5. 调优

### 想更快 / 更省显存

```toml
[asr]
model = "medium"        # 或 "small"，显存 <4GB 时用
compute_type = "int8_float16"

[vad]
min_silence_ms = 320    # 更早断句，字幕出得更快，但可能把一句话切碎
```

### 想更准

```toml
[asr]
model = "large-v3"
beam_size = 3           # 慢一些，但准确率略高
initial_prompt = "ホロライブ、兎田ぺこら、さくらみこ、星街すいせい。"   # 固化主播名/术语写法
```

### 翻译风格

```toml
[translate]
target_language = "繁體中文"     # 想要繁体就改这里
extra_prompt = "这是游戏实况，游戏名是「原神」，角色名保留日文。"
context_lines = 3                # 多带几句前文，译文更连贯
```

## 6. 排障

| 现象 | 原因 / 处理 |
| --- | --- |
| 报 `缺少 soundcard 库` | `pip install soundcard` |
| 报 `系统里没有任何环回设备` | Windows 声音设置里至少要有一个启用的输出设备；蓝牙耳机独占时也会没有 |
| 报 `是普通录音设备` | `[audio] device` 选到了真麦克风。要选名字带 `(Loopback)` 的那个 |
| `--devices` 电平一直是 0 | 声音不是从默认播放设备出的。右键任务栏音量图标 →「音量合成器」，确认直播走的是哪个输出设备，然后把它的名字填进 `[audio] device` |
| 蓝牙耳机抓不到 | 蓝牙切到 HFP（免手操作）时环回会失效。试试 A2DP 模式，或先用有线设备验证 |
| CUDA 加载失败自动退 CPU | 正常兜底。想用显卡：`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`，或把 `[asr] device` 显式设为 `"cuda"` 看具体报错 |
| 字幕全是「ご視聴ありがとうございました」 | whisper 的幻觉，代码里已过滤常见几句。还多的话把 `[vad] threshold` 调到 0.6，或调高 `min_segment_seconds` |
| 翻译报 401 / 缺少 key | 环境变量名要和 `[translate] api_key_env` 一致（默认 `LIVETRANS_API_KEY`），且要重开终端 |
| 字幕出得比说话慢很多 | 降 `[vad] min_silence_ms`，或换更小的 `[asr] model` |
| 字幕显示成方框 | 缺日文字体。Windows 装「Yu Gothic UI」或用 `[ui] font_size` 调大些 |

## 7. 项目结构

```
livetrans/
├── __main__.py           python -m livetrans 入口
├── main.py               CLI：--devices / --check / --terminal / 默认 GUI
├── config.py             配置数据类 + TOML 加载 + 环境变量覆盖
├── audio.py              WASAPI 环回捕获（soundcard）、重采样、设备解析
├── devices.py            音频设备诊断（--devices）
├── vad.py                Silero VAD 流式语音分段
├── asr.py                faster-whisper 封装 + 幻觉过滤
├── pipeline.py           三线程流水线（capture → asr → translate）
├── ui.py                 PySide6 悬浮字幕窗 + 系统托盘
└── translate/
    ├── base.py           提示词构造、重试、输出清洗
    ├── openai_compat.py  OpenAI 兼容后端（DeepSeek/OpenAI/ollama/…）
    └── gemini.py         Gemini 后端
config.toml               配置
tests/
├── test_core.py          配置、翻译工厂、提示词、幻觉过滤
└── test_vad.py           VAD 分段（假模型 + 人工音频，精确断言）
```

线程模型：三个守护线程用有界队列串联，任一级跟不上就丢最旧的片段——实时字幕宁可漏一句，也不要越积越延迟。

## 8. 运行测试

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

不碰音频设备、不联网、不加载 whisper 模型，一两秒跑完。覆盖：

- **VAD 分段**（`tests/test_vad.py`）：用假 VAD 模型配人工构造的音频，精确断言短句合并、
  断句超时、长句直送、强制切片、噪音过滤、帧对齐
- **配置与翻译**（`tests/test_core.py`）：模块导入、配置读取与环境变量覆盖、
  翻译后端工厂、提示词构造、输出清洗、whisper 幻觉过滤

改完 `vad.py` 或 `config.py` 之后，先跑这个再开直播。

## 9. 已知限制

- 只支持 Windows（WASAPI 环回）
- 识别是「整句」模式：要等一句话说完（`[vad] min_silence_ms`，默认 0.7s）才出字幕，不是逐字流式
- 短句还会被 `[vad] hold_ms` 再压最多 1.2s，等后续内容接上——这是换句子完整度的代价
- 长句会被 `[vad] max_segment_seconds` 强制切断，中文译文可能分成两条
- 翻译质量取决于所选的云模型；本地小模型翻日语口语会比较生硬

## License

MIT — 见 [LICENSE](LICENSE)。
