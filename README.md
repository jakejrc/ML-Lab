<div align="center">

# ML-Lab

### 机器学习可视化实验平台

**Interactive Machine Learning Visualization Platform for Education**

[![Version](https://img.shields.io/badge/version-3.8.3-blue?style=flat-square)](https://github.com/jakejrc/ML-Lab)
[![Docker Pulls](https://img.shields.io/docker/pulls/jakejrc/ml-lab?style=flat-square&logo=docker&label=Docker%20Pulls)](https://hub.docker.com/r/jakejrc/ml-lab)
[![Docker Image Size](https://img.shields.io/docker/image-size/jakejrc/ml-lab/latest?style=flat-square&logo=docker&label=Image%20Size)](https://hub.docker.com/r/jakejrc/ml-lab)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

18 种算法 · 9 个数据集 · 自定义上传 · 10 大功能模块 · 代码沙箱直传数据 · AI 助教 · Docker 一键部署

</div>

---

## 更新日志

### v3.8.3 (2026-06-29)

#### 新增：四类实验"📋 生成并复制代码"按钮

为分类、回归、聚类、关联规则四类实验实现一键复制 Python 代码功能：

- **分类实验**：`cls-code-clipboard` 隐藏 Textbox 桥接 + `execCommand('copy')` 自动复制
- **回归实验**：`reg-code-clipboard` 桥接 + `execCommand('copy')` 自动复制
- **聚类实验**：`uns-code-clipboard` 桥接 + `execCommand('copy')` 自动复制
- **关联规则**：`assoc-code-clipboard` 桥接 + `execCommand('copy')` 自动复制

**统一行为**：
- 点击"📋 生成并复制代码" → 按钮变为"✅ 已复制剪贴板"（3 秒后自动恢复）
- 未训练时提示"⚠️ 请先训练模型，再复制代码"
- 使用 `execCommand('copy')` 兼容 HTTP 页面（`navigator.clipboard` 仅支持 HTTPS/localhost）
- 前端轮询 + 隐藏 Textbox 方案，不破坏 Gradio 6 Svelte 事件链

---

### v3.8.2 (2026-06-27)

#### 修复：页面切换闪现问题

平台采用 Gradio SPA 架构，页面面板通过 JS 控制显隐。此前使用 `style.setProperty` 设置内联样式，但 Gradio 在服务端事件后会重渲染组件、清除内联样式，导致切换页面时短暂显示错误面板。

**解决方案**：改用 CSS 类名控制显隐，CSS 规则始终生效、不受 Gradio 重渲染影响：

| 组件 | 旧方案（导致闪现） | 新方案（无闪现） |
|------|--------------------|------------------|
| CSS | `.page-panel` 默认 `display:flex` | `.page-panel` 默认 `display:none!important` |
| 显隐控制 | JS `style.setProperty` 设置 inline style | JS `classList.add/remove('ml-visible')` + `style.setProperty` 双保险 |
| 显示规则 | 无 | `.page-panel.ml-visible { display:flex!important }` |

> 即使 Gradio 重渲染移除所有 inline style，CSS 层级的 `display:none` 始终生效，不会出现面板短暂显示的情况。

#### 修复：导航按钮失效

**根因**：事件监听器直接绑定在 `<label>` 元素上，Gradio 重渲染时替换 DOM 节点导致监听器丢失。

**解决方案**：改用 `document.body` 事件委托，通过 `event.target.closest('.sidebar-nav label')` 捕获点击，body 永不被 Gradio 替换。

#### 修复：数据集加载字体注册失败

**根因**：`callbacks.py` 中 `matplotlib.font_manager.addfont()` 处理 `.ttc` 字体文件时无异常保护，抛出 `FileNotFoundError` 导致整个加载流程中断。

**解决方案**：三步回退策略，每步均有 try/except：
1. 优先使用项目自带的 `fonts/SimHei.ttf`（单文件 `.ttf`，兼容性最好）
2. 回退到系统 `wqy-microhei.ttc`
3. 最后兜底用 `DejaVu Sans`

#### 工程改进

- **代码重构**：`app.py` 从 6396 行精简至约 190 行，页面 UI 和事件逻辑拆分为独立模块
- **版本号升级**：v3.8.1 → **v3.8.2**

#### 测试情况

- **树莓派 5B**（100.66.1.15）：全部功能测试通过，页面切换、导航按钮、数据集加载均正常
- **Docker 镜像**（GitHub Actions 自动构建）：已修复中文乱码问题
  - 根因：Docker 使用 Noto Sans CJK SC 字体，但 `visualization.py` 和 `callbacks.py` 的字体注册逻辑未包含该字体路径
  - 修复：三步回退策略增加 Noto Sans CJK SC OTF（Dockerfile 预提取）支持
  - 字体优先级：SimHei.ttf → NotoSansCJKSC.otf → wqy-microhei.ttc → DejaVu Sans

平台采用 Gradio SPA 架构，页面面板通过 JS 控制显隐。此前使用 `style.setProperty` 设置内联样式，但 Gradio 在服务端事件后会重渲染组件、清除内联样式，导致切换页面时短暂显示错误面板。

**解决方案**：改用 CSS 类名控制显隐，CSS 规则始终生效、不受 Gradio 重渲染影响：

| 组件 | 旧方案（导致闪现） | 新方案（无闪现） |
|------|--------------------|------------------|
| CSS | `.page-panel` 默认 `display:flex` | `.page-panel` 默认 `display:none!important` |
| 显隐控制 | JS `style.setProperty` 设置 inline style | JS `classList.add/remove('ml-visible')` |
| 显示规则 | 无 | `.page-panel.ml-visible { display:flex!important }` |

> 即使 Gradio 重渲染移除所有 inline style，CSS 层级的 `display:none` 始终生效，不会出现面板短暂显示的情况。

#### 工程改进

- **代码重构**：`app.py` 从 6396 行精简至约 190 行，页面 UI 和事件逻辑拆分为独立模块
- **版本号升级**：v3.8.1 → **v3.8.2**

---

### 平台简介

ML-Lab 是一款面向高职教育的**机器学习可视化实验平台**，旨在解决传统机器学习教学中**环境搭建门槛高、算法过程黑箱化、个性化答疑不足**三大痛点。

平台基于 Gradio 构建，集成 scikit-learn 的 18 种经典机器学习算法，提供从数据探索、特征工程、模型训练到结果评估的**全流程可视化实验环境**，并内置支持多种大语言模型的 **AI 智能助教**，为学生的自主学习提供即时支持。

> 江苏工程职业技术学院 · 南通市人工智能新质技术重点实验室

---

## 功能总览

| 模块 | 功能说明 |
|:---:|:---|
| 🧠 **知识图谱** | 62 个机器学习核心概念 · 96 条关系 · 9 大类别 · 交互式力导向图 |
| 📖 **学习路径** | 7 阶段引导式实验流程，含核心知识点讲解与思考题 |
| 📊 **数据工作台** | 9 个内置数据集 + 自定义 CSV/Excel 上传，数据分布可视化 |
| ⚙️ **特征工程** | 6 大子模块：缩放、选择、降维、构造、分箱、相关性分析 |
| 🏷️ **分类实验** | 8 种算法，含混淆矩阵、ROC 曲线、学习曲线与多模型对比 |
| 📈 **回归实验** | 6 种算法，含残差分析、正则化与多项式对比 |
| 🔵 **聚类实验** | 4 种算法，含 PCA 降维可视化与轮廓系数评估 |
| 🔗 **关联规则** | Apriori + FP-Growth，FP-Growth 树可视化 |
| 💻 **代码沙箱** | 在线 Python 执行，支持文件上传与图形自动捕获 |
| 🤖 **AI 助教** | 支持多种大模型 API 的智能问答与实验指导 |

---

## 界面截图

### 首页 · 学习路径

引导式实验流程设计，从数据探索到报告撰写，覆盖机器学习完整学习路线。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/home.png" alt="学习路径" width="800">

### 数据工作台

内置 9 个公开学术数据集 + 自定义 CSV/Excel 上传，支持数据分布可视化。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/data_workbench.png" alt="数据工作台" width="800">

### 特征工程

6 大子模块：特征缩放、选择、降维、构造、分箱离散化与相关性分析。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/feature_eng.png" alt="特征工程" width="800">

### 分类实验

8 种分类算法，支持参数调节、模型训练与多维度评估。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/classify.png" alt="分类实验" width="800">

### 回归实验

6 种回归算法，含残差分析、正则化与多项式对比实验。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/regression.png" alt="回归实验" width="800">

### 聚类实验

4 种聚类算法，含 PCA 降维可视化、轮廓系数评估与肘部法则分析。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/cluster.png" alt="聚类实验" width="800">

### 代码沙箱

在线 Python 代码编辑执行，支持文件上传、图形自动捕获，11 个代码模板。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/code_sandbox.png" alt="代码沙箱" width="800">

---

## 快速开始

### 方式一：Docker 部署（推荐）

```bash
# 基础部署（AI 助教使用内置知识库）
docker run -d -p 7860:7860 --name ml-lab jakejrc/ml-lab:latest

# 启用 AI 助教
docker run -d -p 7860:7860 \
  -e LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  -e LLM_MODEL=qwen-turbo \
  -e LLM_API_KEY=your_api_key \
  --name ml-lab jakejrc/ml-lab:latest

# Ollama 本地模式
docker run -d -p 7860:7860 \
  -e LLM_BASE_URL=http://host.docker.internal:11434/v1 \
  --name ml-lab jakejrc/ml-lab:latest
```

### 方式二：本地安装

```bash
git clone https://github.com/jakejrc/ML-Lab.git
cd ML-Lab
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 文件配置 AI 助教
python app.py
```

浏览器访问 http://localhost:7860 即可使用。

---

## 技术栈

| 技术 | 说明 |
|:---:|:---|
| Gradio | Web 交互框架 |
| Scikit-learn | 机器学习算法引擎（18 种算法） |
| Matplotlib | 数据可视化（支持中文） |
| NumPy / Pandas | 数据处理与科学计算 |
| OpenAI 兼容 API | AI 助教 |
| Docker | 容器化部署 |

---

## 项目结构

```
ML-Lab/
├── app.py                      # Gradio 入口文件（~190 行）
├── ml_lab/                     # 核心模块
│   ├── pages.py                # 页面 UI 组件构建
│   ├── events.py               # 事件绑定与回调
│   ├── html_templates.py       # HTML 模板
│   ├── ui_styles.py            # CSS 样式表
│   ├── algorithms.py           # 18 种 ML 算法
│   ├── callbacks.py            # 各模块回调函数
│   ├── preprocessing.py        # 数据加载与预处理
│   ├── feature_engineering.py  # 特征工程
│   ├── code_generator.py       # 代码生成
│   ├── llm_assistant.py        # AI 助教
│   ├── knowledge_graph.py      # 知识图谱
│   ├── experiment_manager.py   # 实验管理
│   ├── experiment_history.py   # 实验历史
│   ├── learning_progress.py    # 学习进度
│   ├── report_generator.py     # HTML 报告
│   ├── evaluation.py           # 模型评估
│   ├── association_rules.py    # 关联规则
│   ├── association_viz.py      # 关联规则可视化
│   ├── fptree_viz.py           # FP-Growth 树可视化
│   ├── sandbox_templates.py    # 代码沙箱模板
│   ├── visualization.py        # 图表绘制
│   └── version.py              # 版本号
├── data/                       # 内置数据集
├── fonts/SimHei.ttf            # 中文字体
├── screenshots_v34/            # 界面截图
├── tests/                      # 单元测试
├── Dockerfile                  # Docker 构建
├── requirements.txt            # Python 依赖
└── .env.example                # 配置文件模板
---

## 许可证

[MIT License](LICENSE)

---

## 作者

**姜荣昌** · 江苏工程职业技术学院

南通市人工智能新质技术重点实验室

jake_jrc@qq.com
