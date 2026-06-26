# ML-Lab v3.7.0 更新日志

发布日期：2026-05-24

## 新增功能

### 关联规则挖掘模块
- 新增「🔗 关联规则」页面，支持 **Apriori** 和 **FP-Growth** 两种经典关联规则算法
- 频繁项集挖掘与关联规则生成，支持配置最小支持度/置信度/提升度/最大项集长度
- 4 种可视化图表：频繁项集支持度条形图、规则散点图（置信度 vs 提升度）、项集长度分布图、规则热力图
- **FP-Growth 树结构可视化**：出版级风格，含贝塞尔曲线连线、节点投影阴影与 Header Table 卡片式设计
- 算法切换时动态显示对应说明文案
- 代码复现：一键生成并复制 Python 代码

### Float Chat - 右下角浮动 AI 对话
- 新增页面右下角浮动 AI 助教按钮，点击弹出对话面板
- 气泡式对话 UI，支持 Markdown 粗体和行内代码渲染
- 默认使用智谱 GLM-4-Flash 模型，通过 `FLOAT_CHAT_API_KEY` / `FLOAT_CHAT_BASE_URL` / `FLOAT_CHAT_MODEL` 环境变量配置
- 内容限制：仅回答机器学习/数据分析/Python 编程/数学基础相关问题

### 代码沙箱 - 自定义数据上传
- 代码沙箱新增文件上传组件，支持直接上传 CSV/Excel 数据集
- 数据自动加载到代码沙箱命名空间（X, y, X_train, y_test 等变量）
- 鲁棒分割：类别样本不足时自动降级为随机分割

## 界面优化
- Banner 品牌标题「ML-Lab」字号从 17px 提升至 24px
- 修复顶部蓝色 Banner 未铺满页面宽度的白边问题
- 首页学习路径布局优化

## 工程改进
- 代码沙箱模板扩展至 **11 个**（v3.6.5）
- 修复 nav-tag 标签栏点击无效
- 修复代码沙箱模板 SyntaxError
- 修复分类实验特征重要性误警告

---

# ML-Lab v3.6.5 更新日志

发布日期：2026-05-18

## 新增功能

### 代码沙箱一键下载
- 新增「💾 下载 .py」按钮，点击即可将代码编辑器内容导出为 `.py` 文件
- 基于 MutationObserver 监控 gr.File 组件 DOM 变化，文件链接生成后自动触发浏览器下载
- 导出文件自动添加 UTF-8 编码头和生成时间戳注释

### 代码模板扩展
- 代码沙箱模板从 1 个扩展至 **11 个**：默认查看数据、折线图、柱状图、散点图、饼图、直方图+箱线图、热力图、3D散点图、子图组合、分类ML流程、回归ML流程

## 界面优化

### Banner 优化
- 品牌标题「ML-Lab」字号从 17px 提升至 24px
- 修复顶部蓝色 Banner 未铺满页面宽度的白边问题

### 导航交互修复
- 修复顶部 nav-tag 标签栏点击无效的问题（launch js= 注入客户端 IIFE + 300ms 轮询注册 click 事件）
- 修复 launch(js=...) 参数中两处预存 JavaScript 语法错误（引号内换行符 + 重复的 `})();`）

## Bug 修复

- **代码沙箱模板 SyntaxError**：修复"默认（查看数据）"模板 print 语句中 `\n` 转义错误
- **分类实验警告误显示**：修复特征重要性图表在非随机森林/梯度提升算法时仍显示"可能不准确"警告

---

# ML-Lab v3.6.0 更新日志

发布日期：2026-05-15

## 新增功能

### 自定义数据集上传
- **数据工作台 Tab 切换**：新增 `gr.Tabs` / `gr.Tab` 控件，区分"内置数据集"和"自定义上传"两个 Tab
- **文件上传**：支持拖放或点击上传 CSV / Excel 文件（`gr.File`）
- **文件路径输入**：备选方案，支持直接输入本地文件绝对路径（`gr.Textbox`）
- **自动类型检测**：自动识别数值特征列和目标列，支持 `pd.StringDtype` / `object` / `category` 等多种字符串类型
- **任务类型自动判断**：根据目标列唯一值数量自动区分分类（≤20类）和回归任务
- **配置选项**：目标列指定（列名或索引，-1=最后一列）、任务类型（auto/classification/regression）、测试集比例

### 功能验证
- 通过 `gradio_client` API 和 Playwright 浏览器双重验证自定义数据集加载完整可用
- 验证加载后的自定义数据集可正常用于分类实验、特征工程等后续模块

## Bug 修复

### 严重
- **修复 `preprocessing.py` 缺少 `import os`**：`load_custom_dataset` 函数使用 `os.path.splitext` / `os.path.basename` 但文件头部未导入 `os` 模块，导致运行时 `NameError`
- **修复字符串类型检测不兼容 pandas 3.0+**：原条件 `df[col].dtype == object` 无法匹配 pandas 3.0 的 `StringDtype`（dtype 为 `str`），改为使用 `pd.api.types.is_string_dtype()` 和 `pd.api.types.is_object_dtype()` 统一检测

### 低
- **版本号同步**：`app.py` 中 5 处硬编码 `v3.5.2` 全局替换为 `v3.6.0`

---

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
| 代码沙箱 | Python 代码编辑、执行、输出，matplotlib 图形自动捕获显示 |
| AI 助教 | 大语言模型对话辅助 |

### UI 系统

- 深色渐变 sidebar + 自定义 Radio 导航
- 顶部 nav-tag 标签栏 + CSS 动态页面切换
- 蓝白科技风配色 + 状态卡片 + 数据伦理提醒

### 技术栈

- Gradio 6.x / scikit-learn / matplotlib / numpy / pandas
- Python 3.10+

### 已修复 Bug

1. `visualization.py` — `plot_clustering_scatter` 中 `centers_2d` 未初始化 (UnboundLocalError)
2. `algorithms.py` — `HierarchicalModel` 使用废弃的 `affinity` 参数 (sklearn >= 1.2)
3. `app.py` — PCA 聚类的 `predict` 返回 2D ndarray 被误作 labels (TypeError)
4. `app.py` — 聚类算法 if/elif 结构导致 `img1` 未赋值 (UnboundLocalError)
