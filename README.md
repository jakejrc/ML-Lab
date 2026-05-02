<div align="center">

# ML-Lab

### 机器学习可视化实验平台

**Interactive Machine Learning Visualization Platform for Education**

[![Docker Pulls](https://img.shields.io/docker/pulls/jakejrc/ml-lab?style=flat-square&logo=docker&label=Docker%20Pulls)](https://hub.docker.com/r/jakejrc/ml-lab)
[![Docker Image Size](https://img.shields.io/docker/image-size/jakejrc/ml-lab/latest?style=flat-square&logo=docker&label=Image%20Size)](https://hub.docker.com/r/jakejrc/ml-lab)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

18 种算法 · 9 个数据集 · 9 大功能模块 · AI 助教 · Docker 一键部署

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/home.png" alt="ML-Lab 首页" width="800">

</div>

---

## 平台简介

ML-Lab 是一款面向高职教育的**机器学习可视化实验平台**，旨在解决传统机器学习教学中 **环境搭建门槛高、算法过程黑箱化、个性化答疑不足** 三大痛点。

平台基于 Gradio 构建，集成 scikit-learn 的 18 种经典机器学习算法，提供从数据探索、特征工程、模型训练到结果评估的**全流程可视化实验环境**，并内置基于通义千问大语言模型的 **AI 智能助教**，为学生的自主学习提供即时支持。

> 江苏工程职业技术学院 · 南通市人工智能新质技术重点实验室

---

## 功能总览

| 模块 | 功能说明 |
|:---:|:---|
| 📖 **学习路径** | 7 阶段引导式实验流程，含核心知识点讲解、实验任务与思考题，形成"学-练-测"闭环 |
| 📊 **数据工作台** | 9 个公开学术数据集，支持数据分布可视化、统计信息表格展示与预处理对比 |
| ⚙️ **特征工程** | 6 大子模块：特征缩放（4种）、特征选择（F值/互信息/RFE/RF）、PCA 降维、特征构造、分箱离散化、相关性分析，结果均以结构化表格展示 |
| 🏷️ **分类实验** | 8 种算法：逻辑回归、KNN、SVM、决策树、朴素贝叶斯、随机森林、GBDT、神经网络。含评估表格、混淆矩阵、ROC 曲线、学习曲线与多模型横向对比 |
| 📈 **回归实验** | 6 种算法：线性回归、Ridge、Lasso、多项式回归、SVR、决策树回归。含残差分析、正则化与多项式对比实验 |
| 🔵 **聚类实验** | 4 种算法：K-Means、DBSCAN、层次聚类、谱聚类。含 PCA 降维可视化、轮廓系数评估与肘部法则分析 |
| 💻 **代码沙箱** | 在线 Python 代码编辑与执行，已预导入 numpy/pandas/sklearn/matplotlib，支持一键生成训练代码模板 |
| 🤖 **AI 助教** | 基于通义千问大语言模型，支持算法原理讲解、代码逐行解释、错误排查与实验指导 |

---

## 界面预览

### 首页 · 学习路径

引导式实验流程设计，从数据探索到报告撰写，覆盖机器学习完整学习路线。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/home.png" alt="学习路径" width="800">

### 数据工作台

内置 9 个公开学术数据集，支持数据分布可视化与预处理操作。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/data_workbench.png" alt="数据工作台" width="800">

### 特征工程

6 大子模块覆盖特征缩放、特征选择、PCA 降维、特征构造、分箱离散化与相关性分析，结果以结构化表格实时展示。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/feature_eng.png" alt="特征工程" width="800">

### 分类实验

8 种分类算法，支持参数调节、模型训练与多维度评估（混淆矩阵、ROC 曲线、学习曲线）。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/classify.png" alt="分类实验" width="800">

### 回归实验

6 种回归算法，含残差分析、正则化对比与多项式回归实验。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/regression.png" alt="回归实验" width="800">

### 聚类实验

4 种聚类算法，含 PCA 降维散点图、轮廓系数评估与肘部法则分析。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/cluster.png" alt="聚类实验" width="800">

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/cluster_results.png" alt="聚类结果" width="800">

### 代码沙箱

在线 Python 编程环境，预导入机器学习核心库，支持代码编辑、执行与模板加载。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/code_sandbox.png" alt="代码沙箱" width="800">

### AI 助教

基于通义千问大语言模型，提供 7×24 小时智能问答、代码解释与实验指导服务。

<img src="https://raw.githubusercontent.com/jakejrc/ML-Lab/main/screenshots_v34/ai_tutor.png" alt="AI助教" width="800">

---

## 快速开始

### 方式一：Docker 部署（推荐）

```bash
docker run -d -p 7860:7860 --name ml-lab jakejrc/ml-lab:latest
```

启用 AI 助教功能：

```bash
docker run -d -p 7860:7860 -e DASHSCOPE_API_KEY=your_api_key --name ml-lab jakejrc/ml-lab:latest
```

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

# 3. 启动平台
python app.py
```

浏览器访问 [http://localhost:7860](http://localhost:7860) 即可使用。

**配置 AI 助教（可选）**

```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key

# Linux / macOS
export DASHSCOPE_API_KEY=your_api_key
```

> 免费获取 API Key：[阿里云百炼平台](https://dashscope.aliyun.com/)

---

## 技术栈

| 技术 | 说明 |
|:---|:---|
| [Gradio](https://gradio.app/) | Web 交互框架 |
| [Scikit-learn](https://scikit-learn.org/) | 机器学习算法引擎（18 种算法） |
| [Matplotlib](https://matplotlib.org/) | 数据可视化 |
| [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) | 数据处理与科学计算 |
| [通义千问](https://dashscope.aliyun.com/) | AI 助教大语言模型 |
| [Docker](https://www.docker.com/) | 容器化部署 |

---

## 项目结构

```
ML-Lab/
├── app.py                      # Gradio 主界面
├── requirements.txt            # Python 依赖
├── Dockerfile                  # Docker 构建文件
├── VERSION                     # 当前版本号
├── ml_lab/                     # 核心算法包
│   ├── __init__.py
│   ├── algorithms.py           # 18 种 ML 算法实现
│   ├── evaluation.py           # 模型评估与可视化
│   ├── feature_engineering.py  # 特征工程模块
│   ├── llm_assistant.py        # AI 助教（通义千问）
│   ├── preprocessing.py        # 数据加载与预处理
│   ├── code_generator.py       # 代码生成器
│   └── visualization.py        # 图表绘制工具
├── data/                       # 内置数据集
├── static/                     # 静态资源（Logo 等）
├── screenshots_v34/            # 界面截图
└── docs/                       # 项目文档
    ├── 安装手册.html / .docx
    ├── 使用手册.html / .docx
    └── 开发记录.html / .docx
```

---

## 教学应用成效

平台已在江苏工程职业技术学院实际投入使用，取得显著教学效果：

| 指标 | 使用前 | 使用后 | 变化 |
|:---|:---:|:---:|:---:|
| 环境准备时间 | 120 min | 10 min | **↓ 92%** |
| 算法概念理解率 | 45% | 93% | **↑ 106%** |
| AI 助教满意度 | 30% | 91% | **↑ 203%** |

---

## 许可证

[MIT License](LICENSE)

---

## 作者

**姜荣昌** · 江苏工程职业技术学院

南通市人工智能新质技术重点实验室

<a href="mailto:jake_jrc@qq.com">jake_jrc@qq.com</a>
