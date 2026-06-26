# ML-Lab v3.8.1 更新日志

发布日期：2026-06-26

## 修复

- **修正 Footer 实验室名称**：将"南通市新质技术重点实验室"更正为"南通市人工智能新质技术重点实验室"
- **版本号升级**：v3.8.0 → v3.8.1

---

<div align="center">

# ML-Lab

### 机器学习可视化实验平台

**Interactive Machine Learning Visualization Platform for Education**

[![Docker Pulls](https://img.shields.io/docker/pulls/jakejrc/ml-lab?style=flat-square&logo=docker&label=Docker%20Pulls)](https://hub.docker.com/r/jakejrc/ml-lab)
[![Docker Image Size](https://img.shields.io/docker/image-size/jakejrc/ml-lab/latest?style=flat-square&logo=docker&label=Image%20Size)](https://hub.docker.com/r/jakejrc/ml-lab)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

18 种算法 · 9 个数据集 · 自定义上传 · 9 大功能模块 · 代码沙箱直传数据 · AI 助教 · Docker 一键部署

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/home.png" alt="ML-Lab 首页" width="800">

</div>

---

## v3.8.0 更新日志 (2026-05-24)

### 代码沙箱 - 自定义数据上传

代码沙箱新增 **直接上传数据集** 功能，无需切换到数据工作台即可加载数据：

- **文件上传组件**：支持 CSV / Excel 拖放或点击上传，自动识别特征列和目标列（默认最后一列）
- **一键加载**：点击「加载数据」按钮后，`X`, `y`, `X_train`, `y_test` 等变量自动注入代码沙箱命名空间
- **鲁棒分割**：当类别样本不足导致分层分割失败时，自动降级为随机分割，避免报错中断
- **预导入扩展**：新增 `os`, `json` 模块预导入，方便文件操作和数据处理
- **兼容原有流程**：通过数据工作台加载的数据同样可用，两处加载互不冲突

### Bug 修复

- **stratify 降级**：修复自定义数据集中类别样本过少（< 2）时 `split_dataset(stratify=True)` 抛出 `ValueError` 的问题

---

## v3.6.5 更新日志 (2026-05-18)

### 界面优化

- **Banner 品牌标题字号调整**：ML-Lab 品牌标识字号从 17px 提升至 24px，视觉层级更突出
- **Banner 宽度铺满**：修复顶部蓝色 Banner 未铺满页面宽度的白边问题

### 代码沙箱增强

- **一键下载 .py 文件**：新增下载按钮，点击即可将代码编辑器内容导出为 `.py` 文件
- **代码模板扩展**：代码沙箱模板从 1 个扩展至 **11 个**，覆盖折线图、柱状图、散点图、饼图、直方图、热力图、3D 散点图、子图组合、分类/回归机器学习流程

### 导航交互修复

- **顶部标签导航修复**：修复 `nav-tag` 顶部标签栏点击无效的问题

---

## v3.8.0-p1 更新日志 (2026-06-13)

### 新增数据集

- **糖尿病 (Diabetes)**：442 样本, 10 特征, 回归任务 — 糖尿病进展量化指标预测
- **Olivetti 人脸 (Faces)**：400 样本, 4096 特征 (64x64) — 人脸图像降维与聚类演示
- **农作物分类 (Crops)**：500 样本, 5 特征, 3 分类 — 农业遥感模拟数据 (NDVI/EVI/NDWI/SAVI/纹理方差)
- 三个数据集均以 `.npz` 格式存储在 `data/` 目录，**零网络依赖**，瞬间加载

### HTML 实验报告导出

- 分类、回归、聚类、关联规则四个页面各增加「导出 HTML 报告」按钮
- 报告含实验概要、评估结果表格、可视化图表 (base64 内嵌)、代码复现块
- 自包含 HTML 文件，可直接打印或另存为 PDF
- 支持批量实验对比报告

### 学习进度追踪

- 7 阶段引导式学习路径，SQLite 持久化存储完成状态
- 训练实验时自动记录学习活动
- 进度统计：完成率、总阶段数、各阶段状态

### 实验历史记录

- SQLite 持久化存储所有实验记录 (算法/参数/指标/时间戳)
- 支持按任务类型、算法、数据集过滤查询
- 实验统计摘要：总实验数、各类型分布、常用算法排行

### 工程改进

- **代码重构**：app.py 从 3408 行瘦身至 ~3224 行，CSS 样式、代码模板独立为 `ui_styles.py` / `sandbox_templates.py`
- **版本管理**：统一版本源 `VERSION` 文件，各处引用自动同步
- **安全修复**：Float Chat API Key 从硬编码改为环境变量 `FLOAT_CHAT_API_KEY`
- **移动端适配**：768px / 480px 两处断点响应式 CSS，支持手机和平板访问
- **单元测试**：pytest 框架，覆盖版本号、实验历史、学习进度模块
- **根目录清理**：92 项备份/日志/调试文件归档至 `_archive_v3.8.0/`

### Bug 修复

- 修复标题渲染为 `ML-Lab {FULL_NAME}` 占位符的问题
- 修复文字渲染红蓝重影 (移除继承的 text-shadow, 添加 font-smoothing)
- 修复 Olivetti 面孔数据集重复下载超时 -> 本地 `.npz` 加载
- 修复 HTML 报告导出按钮置灰/报错的问题


## v3.6.0 更新日志 (2026-05-15)

### 自定义数据集上传

数据工作台新增 **Tab 切换**布局（内置数据集 / 自定义上传），支持用户上传 CSV/Excel 文件：

- `gr.File` 文件上传组件 + 手动路径输入，适配本地和 Docker 部署
- 自动检测任务类型（分类/回归/聚类）和目标列，支持 8 种编码自动识别
- PCA 降维可视化、数据摘要表格、训练/测试集自动划分
- 上传数据集与全流程实验（分类、回归、聚类、代码沙箱）无缝联动

### 代码沙箱图形输出

代码沙箱新增 **matplotlib 图形自动捕获**功能：

- 通过 `plt.get_fignums()` 记录 `exec()` 前后 figure 编号差异，自动保存新增图形
- 输出区域上下布局：文本输出在上，图形输出在下

### 字体方案升级

- 项目自带 **SimHei** 中文字体（`fonts/SimHei.ttf`），Docker 构建时自动安装
- Docker 镜像同时安装 `fonts-noto-cjk`（从 TTC 预提取 SC OTF）+ SimHei 双重保障

---

## 平台简介

ML-Lab 是一款面向高职教育的**机器学习可视化实验平台**，旨在解决传统机器学习教学中 **环境搭建门槛高、算法过程黑箱化、个性化答疑不足** 三大痛点。

平台基于 Gradio 构建，集成 scikit-learn 的 18 种经典机器学习算法，提供从数据探索、特征工程、模型训练到结果评估的**全流程可视化实验环境**，并内置支持多种大语言模型的 **AI 智能助教**，为学生的自主学习提供即时支持。

> 江苏工程职业技术学院 · 南通市人工智能新质技术重点实验室

---

## 功能总览

| 模块 | 功能说明 |
|:---:|:---|
| 📖 **学习路径** | 7 阶段引导式实验流程，含核心知识点讲解、实验任务与思考题，形成"学-练-测"闭环 |
| 📊 **数据工作台** | 9 个公开学术数据集 + **自定义上传**（CSV/Excel），支持数据分布可视化、统计信息表格展示与预处理对比 |
| ⚙️ **特征工程** | 6 大子模块：特征缩放（4种）、特征选择（F值/互信息/RFE/RF）、PCA 降维、特征构造、分箱离散化、相关性分析 |
| 🏷️ **分类实验** | 8 种算法：逻辑回归、KNN、SVM、决策树、朴素贝叶斯、随机森林、GBDT、神经网络。含评估表格、混淆矩阵、ROC 曲线、学习曲线与多模型对比 |
| 📈 **回归实验** | 6 种算法：线性回归、Ridge、Lasso、多项式回归、SVR、决策树回归。含残差分析、正则化与多项式对比实验 |
| 🔵 **聚类实验** | 4 种算法：K-Means、DBSCAN、层次聚类、谱聚类。含 PCA 降维可视化、轮廓系数评估与肘部法则分析 |
| 🔗 **关联规则** | Apriori + FP-Growth 频繁项集挖掘、关联规则生成、FP-Growth 树可视化 |
| 💻 **代码沙箱** | 在线 Python 代码编辑与执行，**支持直接上传 CSV/Excel 数据集**，预导入 numpy/pandas/sklearn/matplotlib，matplotlib 图形自动显示，11 个代码模板，一键下载 .py |
| 🤖 **AI 助教** | 支持多种大模型 API（通义千问/OpenRouter/硅基流动/Ollama/智谱/DeepSeek 等），算法原理讲解、代码逐行解释与实验指导 |

---

## 快速开始

### 方式一：Docker 部署（推荐）

```bash
# 基础部署（AI 助教使用内置知识库）
docker run -d -p 7860:7860 --name ml-lab jakejrc/ml-lab:latest
```

启用 AI 助教（配置自定义大模型）：

```bash
# 方式 A：通过环境变量传入
docker run -d -p 7860:7860 \
  -e LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  -e LLM_MODEL=qwen-turbo \
  -e LLM_API_KEY=your_api_key \
  --name ml-lab jakejrc/ml-lab:latest

# 方式 B：通过 .env 文件挂载（推荐）
echo "LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1" > .env
echo "LLM_MODEL=qwen-turbo" >> .env
echo "LLM_API_KEY=your_api_key" >> .env
docker run -d -p 7860:7860 --env-file .env --name ml-lab jakejrc/ml-lab:latest
```

> **Ollama 本地模式**：`LLM_API_KEY` 可留空，仅需设置 `LLM_BASE_URL=http://host.docker.internal:11434/v1`

停止与清理：

```bash
docker stop ml-lab && docker rm ml-lab
```

### 方式二：本地安装

**环境要求**

- Python 3.10+
- 无需 GPU

**安装步骤**

```bash
# 1. 克隆项目
git clone https://github.com/jakejrc/ML-Lab.git
cd ML-Lab

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 AI 助教（可选）
cp .env.example .env
# 编辑 .env 文件，填入 LLM_BASE_URL、LLM_MODEL、LLM_API_KEY

# 4. 启动平台
python app.py
```

浏览器访问 [http://localhost:7860](http://localhost:7860) 即可使用。

> 环境变量优先级：系统环境变量 > `.env` 文件 > 代码默认值

---

## 界面预览

### 首页 · 学习路径

引导式实验流程设计，从数据探索到报告撰写，覆盖机器学习完整学习路线。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/home.png" alt="学习路径" width="800">

### 分类实验

8 种分类算法，支持参数调节、模型训练与多维度评估（混淆矩阵、ROC 曲线、学习曲线）。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/classify.png" alt="分类实验" width="800">

### 关联规则 · FP-Growth 树可视化

出版级风格的 FP-Growth 树可视化，含贝塞尔曲线连线、节点投影阴影与 Header Table 卡片式设计。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/association.png" alt="关联规则" width="800">

### AI 助教

支持多种大模型 API，提供智能问答、代码解释与实验指导服务。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/ai_tutor.png" alt="AI助教" width="800">

---

## 技术栈

| 技术 | 说明 |
|:---:|:---|
| [Gradio](https://gradio.app/) | Web 交互框架 |
| [Scikit-learn](https://scikit-learn.org/) | 机器学习算法引擎（18 种算法） |
| [Matplotlib](https://matplotlib.org/) | 数据可视化（支持中文） |
| [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) | 数据处理与科学计算 |
| OpenAI 兼容 API | AI 助教（支持通义千问 / OpenRouter / 硅基流动 / Ollama / 智谱 / DeepSeek 等） |
| [Docker](https://www.docker.com/) | 容器化部署 |

---

## 项目结构

```
ML-Lab/
├── app.py                      # Gradio 主界面
├── .env.example                # 配置文件模板
├── requirements.txt            # Python 依赖
├── Dockerfile                  # Docker 构建（多阶段构建 + CJK 字体）
├── VERSION                     # 当前版本号
├── fonts/                      # 中文字体（SimHei，Docker 中文渲染保障）
│   └── SimHei.ttf
├── ml_lab/                     # 核心算法包
│   ├── __init__.py
│   ├── algorithms.py           # 18 种 ML 算法实现
│   ├── evaluation.py           # 模型评估与可视化
│   ├── experiment_manager.py   # 实验管理
│   ├── feature_engineering.py  # 特征工程模块
│   ├── llm_assistant.py        # AI 助教（多模型 OpenAI 兼容格式）
│   ├── preprocessing.py        # 数据加载与预处理（含自定义数据集）
│   ├── code_generator.py       # 代码生成器
│   ├── association_rules.py    # 关联规则挖掘
│   ├── association_viz.py      # 关联规则可视化
│   ├── fptree_viz.py           # FP-Growth 树可视化
│   └── visualization.py        # 图表绘制工具（中文字体支持）
├── data/                       # 内置数据集
├── static/                     # 静态资源（Logo 等）
├── screenshots_v34/            # 界面截图
└── docs/                       # 项目文档
```

---

## 许可证

[MIT License](LICENSE)

---

## 作者

**姜荣昌** · 江苏工程职业技术学院

南通市人工智能新质技术重点实验室

<a href="mailto:jake_jrc@qq.com">jake_jrc@qq.com</a>
