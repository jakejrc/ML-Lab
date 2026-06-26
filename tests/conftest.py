# -*- coding: utf-8 -*-
"""ML-Lab 测试配置"""

import sys
import os
import pytest

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def project_root():
    """返回项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
