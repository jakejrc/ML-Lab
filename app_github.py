# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.2 — 应用入口（转发模块）

此文件为向后兼容的转发模块，实际实现在 ml_lab/app_builder.py 中。
所有 UI 构建逻辑已拆分到 app_builder.py（1841 行）。

历史：
  v3.8.2 — 从 6446 行单体文件拆分为转发模块（2026-06-27）
  v3.8.1 — 首次模块化拆分
  v3.7.x — 单体架构
"""
from ml_lab.app_builder import create_app, main

if __name__ == "__main__":
    main()
