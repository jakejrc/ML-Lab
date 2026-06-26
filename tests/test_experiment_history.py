# -*- coding: utf-8 -*-
"""测试实验历史模块"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_lab.experiment_history import (
    log_experiment, get_experiments, get_experiment_count,
    get_statistics, delete_experiment
)


def test_log_and_retrieve():
    """测试记录和查询实验"""
    exp_id = log_experiment(
        task_type="classification",
        algorithm="决策树",
        dataset_name="Iris",
        parameters={"max_depth": 5},
        metrics={"accuracy": 0.95},
        tags=["测试"]
    )
    assert exp_id > 0, "实验 ID 应为正整数"

    experiments = get_experiments(limit=10)
    assert len(experiments) > 0, "应能查询到实验记录"

    # 清理
    delete_experiment(exp_id)


def test_statistics():
    """测试统计摘要"""
    stats = get_statistics()
    assert "total" in stats
    assert "by_type" in stats
    assert "top_algorithms" in stats


def test_filter_by_type():
    """测试按任务类型过滤"""
    exp_id = log_experiment(
        task_type="regression",
        algorithm="线性回归",
        dataset_name="Boston",
        tags=["测试"]
    )

    count = get_experiment_count(task_type="regression")
    assert count > 0

    # 清理
    delete_experiment(exp_id)
