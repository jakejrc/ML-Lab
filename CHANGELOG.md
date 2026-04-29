# ML-Lab 变更日志

## v3.1.0 (2026-04-28)

### 新增功能

- 数据工作台新增 **波士顿房价 (Boston)** 数据集（regression 类型，13 个特征 + MEDV 房价目标值）
- 通过 `fetch_openml` 加载（兼容 sklearn >= 1.2，替代已移除的 `load_boston`）
- 数据集总数扩展至 **9 个**

---

## v3.0.0 (2026-04-28)

机器学习可视化实验平台，Gradio 单文件架构。

### 功能模块

| 模块 | 说明 |
|------|------|
| 数据工作台 | 8 个公开数据集，PCA 降维可视化，训练/测试集分割 |
| 特征工程工作台 | 数据标准化、特征选择（单变量/RFE/模型驱动）、PCA 降维、特征构造（多项式/交互/统计）、分箱离散化、相关性分析 |
| 分类实验 | KNN / 逻辑回归 / SVM / 随机森林 / 梯度提升 / 神经网络，含混淆矩阵、ROC 曲线、学习曲线 |
| 回归实验 | 线性回归 / 岭回归 / 决策树回归 / 随机森林回归，含残差分析 |
| 聚类实验 | K-Means / DBSCAN / 层次聚类 / PCA 降维，含肘部法则、eps 分析、树状图 |
| 代码沙箱 | Python 代码编辑、执行、输出 |
| AI 助教 | 大语言模型对话辅助 |

### UI 系统

- 深色渐变 sidebar + 自定义 Radio 导航
- 顶部 nav-tag 标签栏 + CSS 动态页面切换
- 蓝白科技风配色 + 状态卡片 + 数据伦理提醒

### 技术栈

- Gradio 5.x / scikit-learn / matplotlib / numpy / pandas
- Python 3.10+

### 已修复 Bug

1. `visualization.py` — `plot_clustering_scatter` 中 `centers_2d` 未初始化 (UnboundLocalError)
2. `algorithms.py` — `HierarchicalModel` 使用废弃的 `affinity` 参数 (sklearn >= 1.2)
3. `app.py` — PCA 聚类的 `predict` 返回 2D ndarray 被误作 labels (TypeError)
4. `app.py` — 聚类算法 if/elif 结构导致 `img1` 未赋值 (UnboundLocalError)
