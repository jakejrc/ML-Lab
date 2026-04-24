# ML-Lab：面向高职教学的机器学习可视化实验平台

> 基于通义千问国产大模型辅助开发 | 江苏工程职业技术学院

## 项目简介

ML-Lab 是一个面向高职"机器学习"课程教学的可视化实验平台，帮助学生通过交互式可视化直观理解机器学习算法原理。平台借助通义千问国产大模型辅助开发，集成了AI智能问答功能。

## 功能模块

### 1. 数据工作台
- 内置示例数据集：鸢尾花分类、乳腺癌诊断、葡萄酒分类
- 数据分布可视化：目标变量分布、特征分布
- 支持自定义CSV数据集

### 2. 算法实验室
支持6大经典机器学习算法，每个算法提供参数调节和实时可视化：
- **线性回归**：拟合直线、残差分析
- **逻辑回归**：决策边界、Sigmoid函数
- **决策树**：树结构生长、特征重要性
- **SVM**：支持向量、核函数对比
- **KNN**：K值影响、决策边界
- **神经网络(MLP)**：损失曲线、网络结构、学习率影响

### 3. 模型评估台
- 混淆矩阵（数值+百分比）
- ROC曲线 / PR曲线
- 分类指标汇总（准确率、精确率、召回率、F1值）

### 4. AI助教
- 基于通义千问大模型的智能问答
- 算法原理讲解
- 代码解释与错误排查
- 按算法分类的常见问题推荐

## 快速开始

### 环境要求
- Python 3.10+
- 无需GPU

### 安装步骤

1. 克隆项目：
```bash
git clone https://github.com/jakejrc/ML-Lab.git
cd ML-Lab
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置API Key（AI助教功能）：
```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key_here

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

4. 启动平台：
```bash
python app.py
```

5. 打开浏览器访问：http://localhost:7860

## 项目结构

```
ML-Lab/
├── app.py                  # Gradio Web主界面
├── requirements.txt        # 依赖包列表
├── README.md               # 项目说明
├── ml_lab/                 # 核心包
│   ├── __init__.py         # 包初始化
│   ├── algorithms.py       # 6大ML算法模块
│   ├── evaluation.py       # 模型评估模块
│   ├── llm_assistant.py    # AI助教模块（通义千问）
│   ├── preprocessing.py    # 数据预处理模块
│   └── data/               # 示例数据
└── docs/                   # 文档
```

## 作者

jake_jrc

江苏工程职业技术学院

## 许可证

MIT License

## 致谢

- 通义千问大模型（AI辅助开发）
- Scikit-learn、PyTorch、Gradio 等开源社区
