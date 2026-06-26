# -*- coding: utf-8 -*-
"""ML-Lab 学习路径进度追踪。

使用 SQLite 持久化存储学习进度，支持：
- 7 个阶段的学习完成状态
- 每个阶段的完成时间和实验关联
- 学习统计（完成率、用时等）
"""

import os
import sqlite3
import threading
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "learning_progress.db")
_lock = threading.Lock()

# 学习路径 7 个阶段定义
LEARNING_STAGES = [
    {
        "id": "lp_stage1",
        "title": "数据探索与预处理",
        "desc": "加载数据集，查看数据分布与统计信息，掌握数据预处理流程",
        "tasks": ["加载内置数据集", "查看数据分布图表", "执行数据预处理", "理解缺失值处理策略"],
        "badge": "data",
        "module": "data"
    },
    {
        "id": "lp_stage2",
        "title": "特征工程实践",
        "desc": "掌握特征缩放、选择、降维与构造方法",
        "tasks": ["执行特征缩放", "完成特征选择", "运行 PCA 降维", "尝试特征构造"],
        "badge": "data",
        "module": "feature_engineering"
    },
    {
        "id": "lp_stage3",
        "title": "分类算法实验",
        "desc": "训练分类模型，理解混淆矩阵、ROC曲线与学习曲线",
        "tasks": ["训练至少2种分类算法", "理解混淆矩阵", "分析 ROC 曲线", "对比不同算法性能"],
        "badge": "sup",
        "module": "classification"
    },
    {
        "id": "lp_stage4",
        "title": "回归算法实验",
        "desc": "训练回归模型，掌握残差分析与正则化技术",
        "tasks": ["训练线性回归模型", "分析残差分布", "对比正则化效果", "理解多项式回归"],
        "badge": "sup",
        "module": "regression"
    },
    {
        "id": "lp_stage5",
        "title": "聚类与降维",
        "desc": "探索无监督学习方法，理解 K-Means 与层次聚类",
        "tasks": ["运行 K-Means 聚类", "分析肘部法则", "查看 PCA 降维效果", "尝试 DBSCAN"],
        "badge": "unsup",
        "module": "clustering"
    },
    {
        "id": "lp_stage6",
        "title": "关联规则挖掘",
        "desc": "掌握 Apriori 与 FP-Growth 算法，理解支持度、置信度与提升度",
        "tasks": ["运行 Apriori 算法", "分析频繁项集", "理解关联规则评估指标", "查看 FP-Growth 树"],
        "badge": "unsup",
        "module": "association"
    },
    {
        "id": "lp_stage7",
        "title": "代码实践与报告",
        "desc": "在代码沙箱中编写 ML 代码，并生成实验报告",
        "tasks": ["使用代码沙箱编写 Python 代码", "运行并验证代码结果", "与 AI 助教互动学习"],
        "badge": "tool",
        "module": "code_sandbox"
    },
]


def _get_conn():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS learning_progress (
                stage_id TEXT PRIMARY KEY,
                completed INTEGER DEFAULT 0,
                completed_at TEXT,
                experiments_count INTEGER DEFAULT 0,
                notes TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS stage_activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage_id TEXT NOT NULL,
                activity TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (stage_id) REFERENCES learning_progress(stage_id)
            );
        """)
        # 确保所有阶段都有记录
        for stage in LEARNING_STAGES:
            conn.execute(
                "INSERT OR IGNORE INTO learning_progress (stage_id) VALUES (?)",
                (stage["id"],)
            )
        conn.commit()
        conn.close()


_init_db()


def get_all_progress():
    """获取所有阶段的学习进度。

    Returns:
        list[dict]: 每个阶段的进度信息，包含完成状态和 LEARNING_STAGES 元数据
    """
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM learning_progress ORDER BY stage_id"
        ).fetchall()
        conn.close()

        progress_map = {r["stage_id"]: dict(r) for r in rows}
        result = []
        for stage in LEARNING_STAGES:
            pid = stage["id"]
            record = progress_map.get(pid, {"completed": 0, "completed_at": None,
                                            "experiments_count": 0, "notes": ""})
            result.append({
                **stage,
                "completed": bool(record["completed"]),
                "completed_at": record["completed_at"],
                "experiments_count": record["experiments_count"],
                "notes": record["notes"],
            })
        return result


def mark_stage_completed(stage_id, notes=""):
    """标记某个学习阶段为已完成。

    Args:
        stage_id: 阶段 ID（如 "lp_stage1"）
        notes: 备注（可选）
    """
    with _lock:
        conn = _get_conn()
        conn.execute(
            """UPDATE learning_progress
               SET completed = 1, completed_at = ?, notes = ?
               WHERE stage_id = ?""",
            (datetime.now().isoformat(), notes, stage_id)
        )
        conn.execute(
            "INSERT INTO stage_activities (stage_id, activity, timestamp) VALUES (?, ?, ?)",
            (stage_id, "completed", datetime.now().isoformat())
        )
        conn.commit()
        conn.close()


def mark_stage_incomplete(stage_id):
    """取消完成标记"""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE learning_progress SET completed = 0, completed_at = NULL WHERE stage_id = ?",
            (stage_id,)
        )
        conn.commit()
        conn.close()


def record_activity(stage_id, activity):
    """记录阶段活动（如训练了某个算法）"""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO stage_activities (stage_id, activity, timestamp) VALUES (?, ?, ?)",
            (stage_id, activity, datetime.now().isoformat())
        )
        # 增加实验计数
        conn.execute(
            "UPDATE learning_progress SET experiments_count = experiments_count + 1 WHERE stage_id = ?",
            (stage_id,)
        )
        conn.commit()
        conn.close()


def get_progress_summary():
    """获取学习进度摘要。

    Returns:
        dict: 包含完成率、总阶段数、已完成数等
    """
    progress = get_all_progress()
    total = len(progress)
    completed = sum(1 for p in progress if p["completed"])
    return {
        "total_stages": total,
        "completed_stages": completed,
        "completion_rate": f"{completed / total * 100:.0f}%" if total > 0 else "0%",
        "stages": progress,
        "in_progress": total - completed,
    }


# 阶段与实验模块的映射（用于自动记录活动）
STAGE_MODULE_MAP = {s["module"]: s["id"] for s in LEARNING_STAGES}
