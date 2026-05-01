# ML-Lab v3.4 更新日志

发布日期：2026-05-01

## 新增功能

### 全面表格化展示
- **评估结果表格化**：分类/回归/聚类三种实验的评估结果从纯文本改为结构化 HTML 表格，含指标名称、数值、说明列；分类评估额外包含等级徽章（优秀/良好/一般）和指标分组标题
- **数据信息表格化**：数据工作台的"数据信息"和"预处理结果"改为键值对 HTML 表格展示
- **缩放统计信息表格化**：特征缩放的统计输出改为特征级 HTML 表格（含元信息行和各特征统计列）
- **特征选择结果表格化**：支持三种选择方法（F值/互信息/RFE/RF重要性）的不同列结构
- **PCA 分析结果表格化**：主成分方差解释率改为三列表格（主成分/方差解释率/累计解释率）+ 底部累计方差说明
- **特征构造结果表格化**：三种构造方法（多项式/交互/统计）的输出改为键值对表格
- **分箱结果表格化**：策略/分箱数/特征数/分箱边界改为键值对表格
- **相关性分析结果表格化**：元信息表 + 高相关性特征对表格（特征1/特征2/相关系数）

### 聚类参数面板优化
- **动态参数显隐**：选择不同聚类算法（K-Means/DBSCAN/层次聚类）时，仅显示对应算法的参数控件，避免参数混淆
- **左右布局**：算法选择框与参数面板改为左右并排，各占 50%

### 样式优化
- **评估表格表头**：背景从渐变色改为透明，与整体界面风格统一
- **统一 CSS 样式**：新增 `.info-table` / `.eval-table` 样式类，所有表格共享一致视觉风格

## 技术细节
- `evaluation.py`：新增 `format_evaluation_table()` 函数，生成带样式的 HTML 表格
- `app.py`：6 个特征工程子模块回调（`on_fe_scaling`/`on_preprocess`/`on_fe_feature_selection`/`on_fe_pca`/`on_fe_construct`/`on_fe_discretize`/`on_fe_correlation`）输出改为 HTML 表格
- UI 组件：`data_info`/`pp_info`/`fe_scale_info`/`fe_sel_info`/`fe_pca_info`/`fe_const_info`/`fe_disc_info`/`fe_corr_info` 从 `gr.Textbox` 改为 `gr.HTML`
- 保留 `format_evaluation_report()` 纯文本函数，供 PDF 报告生成使用

---

# ML-Lab v3.3 更新日志

发布日期：2026-04-30

## Bug 修复

### 严重
- **修复回归条件图片不可见**：`reg_cmp_img`、`poly_cmp_img`、`model_cmp_img` 组件初始 `visible=False`，训练成功返回图片时未切换为 `visible=True`，导致正则化对比图、多项式阶数对比图、回归模型对比图有数据但不显示

### 中等
- **`on_preprocess` 添加异常捕获**：预处理回调缺少 try/except，异常时整个回调崩溃无提示
- **`on_chat` 添加异常捕获**：AI 助教 API 调用失败时无容错，现在返回友好错误提示

### 低
- **`INT_PARAMS` 移除 `hidden_layer_sizes`**：该参数为字符串类型（如 `"64,32"`），误列入整数参数集会导致类型转换错误
- **聚类添加轮廓系数可视化**：新增 `cluster_img3` 轮廓系数图（Silhouette Plot），动态显示/隐藏

---

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
