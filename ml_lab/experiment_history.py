# -*- coding: utf-8 -*-
"""ML-Lab 实验历史记录模块。

使用 SQLite 持久化存储用户的实验记录，支持：
- 实验元数据（算法、参数、数据集、时间戳）
- 评估指标（准确率、MSE、轮廓系数等）
- 实验备注和标签
"""

import os
import json
import sqlite3
import threading
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "experiment_history.db")
_lock = threading.Lock()


def _get_conn():
    """获取数据库连接（线程安全）"""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """初始化数据库表"""
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                task_type TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                parameters TEXT DEFAULT '{}',
                metrics TEXT DEFAULT '{}',
                notes TEXT DEFAULT '',
                tags TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_task_type ON experiments(task_type);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON experiments(timestamp);
            CREATE INDEX IF NOT EXISTS idx_algorithm ON experiments(algorithm);
        """)
        conn.commit()
        conn.close()


# 启动时初始化
_init_db()


def log_experiment(task_type, algorithm, dataset_name, parameters=None,
                   metrics=None, notes="", tags=None):
    """记录一次实验。

    Args:
        task_type: "classification" | "regression" | "unsupervised" | "association"
        algorithm: 算法名称（如"随机森林"）
        dataset_name: 数据集名称
        parameters: 参数字典
        metrics: 评估指标字典
        notes: 备注
        tags: 标签列表

    Returns:
        int: 实验记录 ID
    """
    with _lock:
        conn = _get_conn()
        cursor = conn.execute(
            """INSERT INTO experiments
               (timestamp, task_type, algorithm, dataset_name, parameters, metrics, notes, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(),
             task_type,
             algorithm,
             dataset_name,
             json.dumps(parameters or {}, ensure_ascii=False),
             json.dumps(metrics or {}, ensure_ascii=False),
             notes,
             json.dumps(tags or [], ensure_ascii=False))
        )
        conn.commit()
        exp_id = cursor.lastrowid
        conn.close()
        return exp_id


def get_experiments(limit=50, task_type=None, algorithm=None, dataset_name=None):
    """查询实验历史。

    Returns:
        list[dict]: 实验记录列表
    """
    with _lock:
        conn = _get_conn()
        conditions = []
        params = []
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        if algorithm:
            conditions.append("algorithm = ?")
            params.append(algorithm)
        if dataset_name:
            conditions.append("dataset_name = ?")
            params.append(dataset_name)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"SELECT * FROM experiments{where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()

        return [{
            "id": r["id"],
            "timestamp": r["timestamp"],
            "task_type": r["task_type"],
            "algorithm": r["algorithm"],
            "dataset_name": r["dataset_name"],
            "parameters": json.loads(r["parameters"]),
            "metrics": json.loads(r["metrics"]),
            "notes": r["notes"],
            "tags": json.loads(r["tags"]),
        } for r in rows]


def get_experiment_count(task_type=None):
    """获取实验总数"""
    with _lock:
        conn = _get_conn()
        if task_type:
            count = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE task_type = ?",
                (task_type,)
            ).fetchone()[0]
        else:
            count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        conn.close()
        return count


def delete_experiment(exp_id):
    """删除指定实验记录"""
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
        conn.commit()
        conn.close()


def get_statistics():
    """获取实验统计摘要。

    Returns:
        dict: 包含总实验数、各任务类型数量、常用算法等
    """
    with _lock:
        conn = _get_conn()
        stats = {}

        row = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
        stats["total"] = row[0]

        rows = conn.execute(
            "SELECT task_type, COUNT(*) as cnt FROM experiments GROUP BY task_type"
        ).fetchall()
        stats["by_type"] = {r["task_type"]: r["cnt"] for r in rows}

        rows = conn.execute(
            "SELECT algorithm, COUNT(*) as cnt FROM experiments "
            "GROUP BY algorithm ORDER BY cnt DESC LIMIT 10"
        ).fetchall()
        stats["top_algorithms"] = [(r["algorithm"], r["cnt"]) for r in rows]

        conn.close()
        return stats
