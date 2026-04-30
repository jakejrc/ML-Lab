 Stashed changes
---

## 技术栈

- **前端框架**：[Gradio](https://gradio.app/)
- **机器学习**：[Scikit-learn](https://scikit-learn.org/)
- **数据可视化**：[Matplotlib](https://matplotlib.org/)
- **AI 助教**：通义千问 (DashScope API)
- **语言**：Python 3.10+

---

## 快速开始

### 环境要求

- Python 3.10+
- 无需 GPU

### 安装步骤

1. **克隆项目**：

```bash
git clone https://github.com/jakejrc/ML-Lab.git
cd ML-Lab
```

2. **安装依赖**：

```bash
pip install -r requirements.txt
```

3. **配置 AI 助教 API Key**（可选，不配置不影响其他功能）：

```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key_here

# Linux / macOS
export DASHSCOPE_API_KEY=your_api_key_here
```

> 免费获取 API Key：[阿里云百炼平台](https://dashscope.aliyun.com/)

4. **启动平台**：

```bash
python app.py
```

5. **打开浏览器访问**：http://localhost:7860

---

## Docker 部署

### 一键启动

```bash
docker run -d -p 7860:7860 --name ml-lab jakejrc/ml-lab:latest
```

### 配置 AI 助教

```bash
docker run -d -p 7860:7860 -e DASHSCOPE_API_KEY=your_api_key_here --name ml-lab jakejrc/ml-lab:latest
```

### 停止与删除

```bash
docker stop ml-lab && docker rm ml-lab
```

---

## 项目结构

```
ML-Lab/
├── app.py                    # Gradio 主界面
├── requirements.txt          # Python 依赖
├── Dockerfile                # Docker 构建文件
├── CHANGELOG.md              # 变更日志
├── VERSION                   # 版本号
├── ml_lab/                   # 核心算法包
│   ├── __init__.py           # 包初始化
│   ├── algorithms.py         # 15 种 ML 算法实现
│   ├── evaluation.py         # 模型评估与可视化
│   ├── feature_engineering.py # 特征工程模块
│   ├── llm_assistant.py      # AI 助教（通义千问）
│   ├── preprocessing.py      # 数据加载与预处理
│   └── visualization.py      # 图表绘制工具
└── docs/                     # 项目文档
    ├── 安装手册.html/docx
    ├── 使用手册.html/docx
    └── 开发记录.html/docx
```

---

## 作者

**姜荣昌** (jake_jrc)

江苏工程职业技术学院 · 南通市人工智能新质技术重点实验室

---

## 许可证

[MIT License](LICENSE)

---

## 致谢

- 通义千问大模型（AI 辅助开发与 AI 助教功能）
- [Scikit-learn](https://scikit-learn.org/)、[Gradio](https://gradio.app/)、[Matplotlib](https://matplotlib.org/) 等开源社区
