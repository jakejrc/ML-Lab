# -*- coding: utf-8 -*-
"""测试版本模块"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.version import VERSION, FULL_NAME


def test_version_not_empty():
    """版本号不应为空"""
    assert VERSION, "VERSION 不应为空"
    assert len(VERSION) > 0


def test_full_name_format():
    """FULL_NAME 应以前缀 v 开头"""
    assert FULL_NAME.startswith("v"), f"FULL_NAME 应以 v 开头: {FULL_NAME}"
    assert FULL_NAME == f"v{VERSION}"


def test_version_semver_format():
    """版本号应符合语义化版本格式 x.y.z"""
    parts = VERSION.split(".")
    assert len(parts) == 3, f"版本号应为三段式: {VERSION}"
    for p in parts:
        assert p.isdigit(), f"版本号各部分应为数字: {p}"
