# -*- coding: utf-8 -*-
"""测试 AI 助教模块"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.llm_assistant import (
    set_llm_config, get_llm_config, apply_preset_provider,
    get_provider_names, set_api_key, get_api_key,
    PRESET_PROVIDERS, PRESET_QUESTIONS,
)


class TestSetLlmConfig:
    def test_set_all(self):
        set_llm_config(base_url="https://example.com/v1", model="test-model", api_key="sk-test123456789")
        cfg = get_llm_config()
        assert cfg["base_url"] == "https://example.com/v1"
        assert cfg["model"] == "test-model"

    def test_set_partial(self):
        original = get_llm_config()
        set_llm_config(model="new-model")
        cfg = get_llm_config()
        assert cfg["model"] == "new-model"
        # Restore
        set_llm_config(base_url=original["base_url"], model=original["model"])

    def test_trailing_slash_stripped(self):
        set_llm_config(base_url="https://example.com/v1/")
        cfg = get_llm_config()
        assert not cfg["base_url"].endswith("/")


class TestGetLlmConfig:
    def test_api_key_masked(self):
        set_llm_config(api_key="sk-abcdefghijklmnopqrstuvwxyz")
        cfg = get_llm_config()
        assert "api_key" not in cfg
        assert "api_key_masked" in cfg
        assert "****" in cfg["api_key_masked"]

    def test_empty_api_key(self):
        set_llm_config(api_key="")
        # Also clear env var that might have been set
        os.environ.pop("DASHSCOPE_API_KEY", None)
        cfg = get_llm_config()
        assert cfg["api_key_masked"] == ""


class TestApplyPresetProvider:
    def test_known_provider(self):
        info = apply_preset_provider("通义千问 (DashScope)")
        assert "base_url" in info
        assert "model" in info

    def test_unknown_provider(self):
        info = apply_preset_provider("不存在的提供商")
        assert "未知提供商" in info.get("hint", "")


class TestGetProviderNames:
    def test_returns_list(self):
        names = get_provider_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "通义千问 (DashScope)" in names


class TestApiKeyCompat:
    def test_set_and_get(self):
        set_api_key("sk-test-key-1234567890")
        key = get_api_key()
        assert "****" in key

    def test_empty_key(self):
        set_api_key("")
        key = get_api_key()
        assert key == ""


class TestPresetQuestions:
    def test_not_empty(self):
        assert len(PRESET_QUESTIONS) > 0

    def test_questions_are_strings(self):
        for q in PRESET_QUESTIONS:
            assert isinstance(q, str)
