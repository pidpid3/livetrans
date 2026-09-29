"""不依赖音频设备和网络的检查：模块能导入、配置读取、提示词与输出清洗。"""

from __future__ import annotations

import importlib
import os
import pathlib
import tempfile
import unittest

MODULES = (
    "livetrans",
    "livetrans.audio",
    "livetrans.asr",
    "livetrans.config",
    "livetrans.devices",
    "livetrans.main",
    "livetrans.pipeline",
    "livetrans.ui",
    "livetrans.vad",
    "livetrans.translate",
    "livetrans.translate.base",
    "livetrans.translate.gemini",
    "livetrans.translate.openai_compat",
)


class TestImports(unittest.TestCase):
    def test_all_modules_import(self):
        """逐个导入，任何语法错误或名字写错都会在这里炸出来。"""
        for name in MODULES:
            with self.subTest(module=name):
                importlib.import_module(name)


class TestConfig(unittest.TestCase):
    def test_defaults_load(self):
        from livetrans.config import load_config

        cfg = load_config()
        self.assertEqual(cfg.asr.language, "ja")
        self.assertGreater(cfg.vad.merge_seconds, 0, "短句缓冲默认应该是开着的")

    def test_env_override_coerces_type(self):
        from livetrans.config import load_config

        os.environ["LIVETRANS_ASR_BEAM_SIZE"] = "4"
        os.environ["LIVETRANS_UI_SHOW_ORIGINAL"] = "false"
        try:
            cfg = load_config()
            self.assertEqual(cfg.asr.beam_size, 4)
            self.assertIsInstance(cfg.asr.beam_size, int)
            self.assertIs(cfg.ui.show_original, False)
        finally:
            os.environ.pop("LIVETRANS_ASR_BEAM_SIZE", None)
            os.environ.pop("LIVETRANS_UI_SHOW_ORIGINAL", None)

    def test_unknown_field_is_rejected(self):
        from livetrans.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "c.toml"
            path.write_text("[asr]\nnot_a_field = 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_missing_file_falls_back_to_defaults(self):
        from livetrans.config import load_config

        cfg = load_config(pathlib.Path(tempfile.gettempdir()) / "definitely-not-here.toml")
        self.assertEqual(cfg.translate.provider, "openai")


class TestTranslatorFactory(unittest.TestCase):
    def test_none_disables_translation(self):
        from livetrans.config import TranslateConfig
        from livetrans.translate import create_translator

        self.assertIsNone(create_translator(TranslateConfig(provider="none")))

    def test_aliases_pick_openai_compatible(self):
        from livetrans.config import TranslateConfig
        from livetrans.translate import create_translator

        for alias in ("openai", "deepseek", "ollama"):
            with self.subTest(provider=alias):
                translator = create_translator(TranslateConfig(provider=alias))
                self.assertEqual(translator.name, "openai-compatible")
                translator.cfg  # 只是确保对象可用

    def test_gemini_backend(self):
        from livetrans.config import TranslateConfig
        from livetrans.translate import create_translator

        self.assertEqual(create_translator(TranslateConfig(provider="gemini")).name, "gemini")

    def test_unknown_provider_raises(self):
        from livetrans.config import TranslateConfig
        from livetrans.translate import create_translator

        with self.assertRaises(ValueError):
            create_translator(TranslateConfig(provider="not-a-thing"))


class TestPromptBuilding(unittest.TestCase):
    def test_output_cleaning(self):
        from livetrans.translate.base import clean_output

        self.assertEqual(clean_output("翻译：你好"), "你好")
        self.assertEqual(clean_output("译文: 你好"), "你好")
        self.assertEqual(clean_output("「你好」"), "你好")
        self.assertEqual(clean_output("  你好  "), "你好")
        self.assertEqual(clean_output("你好。"), "你好。")

    def test_context_is_included(self):
        from livetrans.config import TranslateConfig
        from livetrans.translate.base import build_messages

        cfg = TranslateConfig(target_language="简体中文", context_lines=2)
        system, user = build_messages(cfg, "こんにちは", [("おはよう", "早上好")])
        self.assertIn("简体中文", system)
        self.assertIn("こんにちは", user)
        self.assertIn("早上好", user)

    def test_context_lines_zero_drops_context(self):
        from livetrans.config import TranslateConfig
        from livetrans.translate.base import build_messages

        cfg = TranslateConfig(context_lines=0)
        _, user = build_messages(cfg, "はい", [("おはよう", "早上好")])
        self.assertNotIn("早上好", user)

    def test_extra_prompt_appended(self):
        from livetrans.config import TranslateConfig
        from livetrans.translate.base import build_messages

        cfg = TranslateConfig(extra_prompt="角色名保留日文")
        system, _ = build_messages(cfg, "テスト")
        self.assertIn("角色名保留日文", system)


class TestHallucinationFilter(unittest.TestCase):
    def test_common_hallucinations_are_dropped(self):
        from livetrans.asr import _looks_like_hallucination

        for text in ("ご視聴ありがとうございました", "チャンネル登録お願いします", "字幕", "wwww", "ん", "", "   "):
            with self.subTest(text=text):
                self.assertTrue(_looks_like_hallucination(text))

    def test_normal_speech_is_kept(self):
        from livetrans.asr import _looks_like_hallucination

        for text in (
            "こんにちは、今日も配信していきます",
            "ゲームで遊ばせていただこうと思いまして",
            "あ、後ろに人いる",
        ):
            with self.subTest(text=text):
                self.assertFalse(_looks_like_hallucination(text), f"不该被当成幻觉：{text}")


if __name__ == "__main__":
    unittest.main()
