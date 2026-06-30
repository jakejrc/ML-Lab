# -*- coding: utf-8 -*-
"""
ML-Lab v3.8.4 — 全局状态管理器

同时支持 dict 式访问 (_g["key"]) 和属性式访问 (_g.key)。
向后兼容: 所有 _g["key"] 和 _sync() 调用继续有效。
"""

import copy
from typing import Any, Optional


class MLState(dict):
    """ML-Lab 全局状态管理器"""

    DEFAULTS = {
        # 数据
        "X": None, "y": None,
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "feature_names": None, "target_names": None,
        # 模型
        "model": None,
        "dataset_name": "未加载", "model_name": "未训练",
        # 报告
        "report_images": {},
        "last_eval_report": "",
        "last_cluster_report": "",
        "last_algo_name": "",
        "last_task_type": "",
        # 实验记录
        "fe_results": {},
        "experiment_log": [],
        # 关联规则
        "last_assoc_model": None,
        "last_assoc_rules": None,
        # 生成代码
        "last_generated_code": "",
    }

    def __init__(self):
        super().__init__()
        self.reset()

    # ---- 生命周期 ----

    def reset(self):
        """重置所有状态为默认值（深拷贝避免共享可变对象）"""
        self.clear()
        for k, v in self.DEFAULTS.items():
            self[k] = copy.deepcopy(v)

    # ---- 属性式访问 ----

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'MLState' object has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)

    # ---- 便捷属性 ----

    @property
    def has_data(self) -> bool:
        """是否已加载数据"""
        return self["X_train"] is not None

    @property
    def has_model(self) -> bool:
        """是否已训练模型"""
        return self["model"] is not None

    @property
    def task_type(self) -> str:
        """当前任务类型"""
        dtype = self.get("last_task_type", "")
        return dtype if dtype else "unknown"

    # ---- 便捷方法 ----

    def sync(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        fnames,
        tnames,
        X: Optional = None,
        y: Optional = None,
    ):
        """同步训练/测试数据到全局状态"""
        self["X_train"] = X_train
        self["X_test"] = X_test
        self["y_train"] = y_train
        self["y_test"] = y_test
        self["feature_names"] = fnames
        self["target_names"] = tnames
        if X is not None:
            self["X"] = X
        if y is not None:
            self["y"] = y

    def update_model(self, model, name: str):
        """保存训练好的模型"""
        self["model"] = model
        self["model_name"] = name

    def log_experiment(self, entry: dict):
        """记录一次实验到日志"""
        self["experiment_log"].append(entry)

    def to_dict(self) -> dict:
        """导出可序列化的状态快照"""
        d = dict(self)
        d.pop("model", None)
        d.pop("report_images", None)
        d.pop("fe_results", None)
        d.pop("experiment_log", None)
        d.pop("last_assoc_model", None)
        return d


# 全局单例（向后兼容）
_g: MLState = MLState()


def _sync(*args, **kwargs):
    """向后兼容的同步函数"""
    _g.sync(*args, **kwargs)
