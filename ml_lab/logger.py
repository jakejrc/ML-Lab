# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.4 - 日志配置模块

统一日志管理，替代分散的 print 语句。
支持控制台输出和文件日志两级记录。
"""

import logging
import sys
import os

LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

_LOG_FORMAT = "[%(asctime)s] [%(levelname)-5s] %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logger(name="ml_lab", level="info", log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(LEVEL_MAP.get(level, logging.INFO))
    if logger.handlers:
        return logger
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATE_FORMAT))
    logger.addHandler(console)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATE_FORMAT))
        logger.addHandler(fh)
    return logger


logger = setup_logger()
