"""VAD 分段逻辑的单元测试。

用一个「看能量说话」的假模型顶替 Silero，这样测试音频可以完全人工构造，
断言就能精确验证分段行为 —— 尤其是新增的短句缓冲（不然一句话会被翻译拆散）。
"""

from __future__ import annotations

import unittest

import numpy as np

from livetrans.config import VadConfig
from livetrans.vad import FRAME, SpeechSegmenter

SR = 16000


class EnergyVad:
    """假 VAD：幅度大就算语音。接口和真模型一致（输入必须是 512 的整数倍）。"""

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        assert audio.ndim == 1, "输入应该是 1D"
        assert audio.shape[0] % FRAME == 0, "输入长度应该是 512 的倍数"
        frames = audio.reshape(-1, FRAME)
        rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
        return (rms > 0.01).astype(np.float32)


def build_audio(spec, sr: int = SR) -> np.ndarray:
    """spec: [(秒数, 是不是语音), ...]"""
    chunks = []
    for seconds, speech in spec:
        n = int(round(sr * seconds))
        chunks.append(np.full(n, 0.5 if speech else 0.0, dtype=np.float32))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def feed(segmenter: SpeechSegmenter, audio: np.ndarray, chunk_s: float = 0.1) -> list[np.ndarray]:
    """按 100ms 一块喂进去，模拟实时采集。"""
    step = int(SR * chunk_s)
    out: list[np.ndarray] = []
    for start in range(0, audio.size, step):
        out.extend(segmenter.process(audio[start : start + step]))
    return out


def make_segmenter(**overrides) -> SpeechSegmenter:
    return SpeechSegmenter(VadConfig(**overrides), EnergyVad())


# 三段短语音，被 0.8s 停顿分开 —— 实际上是一句话被换气切了三截
CHOPPY = [
    (0.5, True), (0.8, False),
    (0.5, True), (0.8, False),
    (0.5, True), (3.0, False),
]


class TestShortSegmentBuffering(unittest.TestCase):
    def test_short_segments_get_merged(self):
        """三段短语音应该合并成一段送出去，而不是切三刀。"""
        segments = feed(make_segmenter(), build_audio(CHOPPY))
        self.assertEqual(len(segments), 1, f"期望合并成 1 段，实际 {len(segments)} 段")

    def test_without_buffering_it_splits(self):
        """对照：把缓冲关掉，同样的音频就该被切成三段。"""
        segments = feed(make_segmenter(merge_seconds=0.0), build_audio(CHOPPY))
        self.assertEqual(len(segments), 3, f"期望 3 段，实际 {len(segments)} 段")

    def test_hold_timeout_releases_short_segment(self):
        """一句短话之后长时间没人说话，缓冲要按时放出去，不能一直压着。"""
        segments = feed(make_segmenter(), build_audio([(0.5, True), (4.0, False)]))
        self.assertEqual(len(segments), 1)

    def test_short_segment_released_promptly(self):
        """静音刚过 hold 就该出来：0.5 语音 + 0.7 断句 + 1.2 hold ≈ 2.4s 内。"""
        segmenter = make_segmenter()
        audio = build_audio([(0.5, True), (4.0, False)])
        step = int(SR * 0.1)
        produced_at = None
        for start in range(0, audio.size, step):
            if segmenter.process(audio[start : start + step]):
                produced_at = start / SR
                break
        self.assertIsNotNone(produced_at, "短句一直没放出来")
        self.assertLess(produced_at, 3.0, f"放得太晚了：{produced_at:.2f}s")


class TestLongSegments(unittest.TestCase):
    def test_long_speech_goes_out_without_buffering(self):
        """够长的语音不该被攒着，说完就送。"""
        segments = feed(make_segmenter(), build_audio([(4.0, True), (3.0, False)]))
        self.assertEqual(len(segments), 1)

    def test_very_long_speech_is_forced_out(self):
        """一口气说个没完时要强制切片，不能让字幕一直不出来。"""
        segments = feed(
            make_segmenter(max_segment_seconds=3.0),
            build_audio([(8.0, True), (3.0, False)]),
        )
        self.assertGreaterEqual(len(segments), 2)


class TestNoise(unittest.TestCase):
    def test_pure_silence_produces_nothing(self):
        self.assertEqual(feed(make_segmenter(), build_audio([(5.0, False)])), [])

    def test_tiny_blips_are_dropped(self):
        """极短的爆音（咳嗽、鼠标点击）不该产生字幕。"""
        audio = build_audio([(0.1, True), (1.0, False), (0.1, True), (3.0, False)])
        self.assertEqual(feed(make_segmenter(min_segment_seconds=0.4), audio), [])


class TestFrameAlignment(unittest.TestCase):
    def test_probabilities_align_with_new_frames(self):
        """_speech_probs 只能取本批新帧的概率，不能把 context 也算进去。

        否则进入语音状态会晚几帧，整段的起止都会偏。
        """
        segmenter = make_segmenter()
        segmenter.process(np.zeros(FRAME * 20, dtype=np.float32))
        segmenter.process(np.full(FRAME * 5, 0.5, dtype=np.float32))
        self.assertTrue(segmenter._speaking, "5 帧语音（160ms）就该进入说话状态了")

    def test_partial_frame_is_carried_over(self):
        """不足一帧的尾巴要留着和下次拼起来，不能丢。"""
        segmenter = make_segmenter()
        odd = np.full(FRAME * 3 + 100, 0.5, dtype=np.float32)
        segmenter.process(odd)
        self.assertEqual(segmenter._incoming.size, 100)


class TestRuntimeReconfigure(unittest.TestCase):
    """托盘菜单里改 VAD 参数走的就是 apply_config()，这里验证它真的立即生效。"""

    def test_apply_config_recomputes_thresholds(self):
        cfg = VadConfig(min_silence_ms=700, merge_seconds=2.5, hold_ms=1200)
        segmenter = SpeechSegmenter(cfg, EnergyVad())
        old_silence = segmenter.min_silence_frames

        cfg.min_silence_ms = 200
        cfg.merge_seconds = 0.0
        cfg.hold_ms = 400
        segmenter.apply_config()

        self.assertLess(segmenter.min_silence_frames, old_silence)
        self.assertEqual(segmenter.merge_frames, 1, "merge_seconds=0 等价于关掉缓冲")
        self.assertLess(segmenter.hold_frames, int(round(1200 / segmenter.frame_ms)))

    def test_new_thresholds_apply_to_the_next_segment(self):
        """门槛调小之后同样的音频会被切得更碎 —— 说明确实作用到了分段行为上，不只是改了个数。"""
        audio = build_audio([(0.5, True), (0.8, False), (0.5, True), (3.0, False)])
        cfg = VadConfig(merge_seconds=0.0, min_segment_seconds=0.4)
        segmenter = SpeechSegmenter(cfg, EnergyVad())

        cfg.min_silence_ms = 2000  # 0.8s 静音够不着门槛
        segmenter.apply_config()
        self.assertEqual(len(feed(segmenter, audio)), 1, "该连成一段")
        segmenter.flush()  # 清掉内部残留，别影响下一轮

        cfg.min_silence_ms = 200  # 现在 0.8s 静音足够断句
        segmenter.apply_config()
        self.assertEqual(len(feed(segmenter, audio)), 2, "该切成两段")


if __name__ == "__main__":
    unittest.main()
