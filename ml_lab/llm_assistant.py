# -*- coding: utf-8 -*-
"""
ML-Lab AI助教模块
集成通义千问（Qwen）大模型API，提供机器学习相关的智能问答功能。
支持原理讲解、代码解释、错误排查等场景。

使用前需配置 API Key:
    方法1: 设置环境变量 DASHSCOPE_API_KEY
    方法2: 调用 set_api_key("your-key") 进行设置

获取 API Key: https://dashscope.aliyun.com (免费额度)
"""

import os
import json
import time


# ---------------------------------------------------------------------------
# API Key 管理
# ---------------------------------------------------------------------------

_api_key = os.environ.get("DASHSCOPE_API_KEY", "")


def set_api_key(key: str):
    """设置通义千问 API Key

    Args:
        key (str): API Key 字符串
    """
    global _api_key
    _api_key = key
    os.environ["DASHSCOPE_API_KEY"] = key


def get_api_key() -> str:
    """获取当前 API Key（中间用****脱敏）"""
    if not _api_key:
        return ""
    return _api_key[:6] + "****" + _api_key[-4:] if len(_api_key) > 10 else "****"


# ---------------------------------------------------------------------------
# 系统提示词（预设AI助教角色）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一个名为"ML-Lab AI助教"的机器学习教学助手，面向高职院校学生。
你的职责是：
1. 用通俗易懂的语言解释机器学习的核心概念和算法原理
2. 帮助学生理解代码的功能和运行逻辑
3. 协助排查代码错误，但不要直接给出完整答案，而是引导学生思考
4. 回答应当简洁、准确、避免过于复杂的数学推导

回答要求：
- 使用中文回答
- 适当使用比喻和类比帮助理解
- 如果涉及代码，使用Python代码块并添加注释
- 回答末尾标注"（由通义千问大模型生成）"
"""


# ---------------------------------------------------------------------------
# 内置知识库（离线回退方案）
# ---------------------------------------------------------------------------

BUILTIN_KNOWLEDGE = {
    "梯度下降": """
梯度下降是机器学习中最常用的优化算法之一，核心思想是：**沿着梯度的反方向一步步走，直到找到损失函数的最小值**。

用生活中的例子理解：想象你站在山顶上，眼睛被蒙住，你想走到山谷最低点。你只能用脚感受地面的坡度，然后往最陡的下坡方向走一步。反复这样做，最终你就会到达山谷最低点。

三个关键超参数：
- **学习率 (learning_rate)**：每一步走多远。太大可能跳过最低点，太小走得太慢。
- **迭代次数 (n_iterations)**：总共走多少步。
- **批量大小 (batch_size)**：每次看多少个数据来决定方向。

代码示例：
```python
for i in range(n_iterations):
    gradient = compute_gradient(weights, X, y)
    weights -= learning_rate * gradient
```

（由通义千问大模型生成）
""",
    "过拟合": """
过拟合是指模型在训练数据上表现很好，但在新数据上表现很差。

理解方式：就像一个学生把考试题**死记硬背**了，做原题能拿满分，但换一道同类型的题就不会了。

常见原因和解决办法：
1. **模型太复杂** → 简化模型结构，减少参数
2. **数据太少** → 增加训练数据或使用数据增强
3. **训练太久** → 使用早停法 (Early Stopping)
4. **正则化** → 使用 L1/L2 正则化约束模型

（由通义千问大模型生成）
""",
    "决策树": """
决策树是一种像"20个问题"游戏一样的分类算法。

核心概念：
- **根节点**：树的起点，包含所有数据
- **内部节点**：根据某个特征的条件进行分裂
- **叶子节点**：最终的分类结果

分裂标准：
- **信息增益（基尼系数）**：分裂后数据"变纯"了多少
- **信息增益比**：考虑了分支数量的信息增益

关键超参数：
- `max_depth`：树的最大深度，太深容易过拟合
- `min_samples_split`：节点最少需要多少样本才能继续分裂

优点：直观、可解释性强
缺点：容易过拟合、对数据敏感

（由通义千问大模型生成）
""",
    "SVM": """
SVM（支持向量机）的核心思想：**找到一条线（或超平面），让两类数据之间的"间隔"最大化**。

关键概念：
- **超平面**：在高维空间中分隔数据的边界
- **支持向量**：距离超平面最近的那些数据点，它们"支撑"着超平面的位置
- **核技巧**：当数据线性不可分时，将数据映射到高维空间使其变得线性可分

常用核函数：
- `linear`：线性核，适合线性可分数据
- `rbf`：径向基函数核，最常用，适合大多数场景
- `poly`：多项式核，适合特定的非线性关系

超参数 C：控制对误分类的容忍度。C越大，越不允许误分类（可能过拟合）。

（由通义千问大模型生成）
""",
    "交叉验证": """
交叉验证是评估模型性能的一种更可靠的方法，比简单的"训练集-测试集"划分更稳定。

最常用的是 **K折交叉验证 (K-Fold Cross Validation)**：
1. 将数据分成K份（通常K=5或K=10）
2. 每次用其中1份作为测试集，其余K-1份作为训练集
3. 重复K次，每次使用不同的测试集
4. 将K次的评估结果取平均

为什么需要交叉验证？
- 单次划分可能碰巧训练集/测试集分布不均
- 交叉验证能更全面地评估模型的泛化能力

（由通义千问大模型生成）
""",
    "逻辑回归": """
逻辑回归虽然名字里有"回归"，但它实际上是一个**分类算法**！

核心思想：在线性回归的基础上，加一个 **Sigmoid 函数**，把输出压缩到 0~1 之间，表示属于某个类别的"概率"。

Sigmoid函数：σ(z) = 1 / (1 + e^(-z))

工作流程：
1. z = w₁x₁ + w₂x₂ + ... + b （线性组合）
2. p = σ(z) （转换为概率）
3. 如果 p ≥ 0.5，预测为类别1；否则预测为类别0

超参数：
- `C`：正则化强度的倒数。C越大，模型越复杂

适用场景：二分类问题（如邮件是否为垃圾邮件、客户是否流失）

（由通义千问大模型生成）
""",
}


# ---------------------------------------------------------------------------
# API 调用
# ---------------------------------------------------------------------------

def call_qwen_api(messages, model="qwen-turbo", max_tokens=1024,
                   temperature=0.7):
    """调用通义千问 API

    Args:
        messages (list): 对话消息列表
        model (str): 模型名称 (qwen-turbo / qwen-plus / qwen-max)
        max_tokens (int): 最大生成token数
        temperature (float): 温度参数 (0~1)

    Returns:
        str: 模型回复内容
    """
    global _api_key

    if not _api_key:
        return "[未配置API Key] 请先调用 ml_lab.llm_assistant.set_api_key('your-key') 设置API Key。\n\n获取免费API Key: https://dashscope.aliyun.com"

    try:
        import requests

        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(url, headers=headers,
                                 json=payload, timeout=30)
        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        elif "error" in result:
            return f"[API错误] {result['error'].get('message', '未知错误')}"
        else:
            return "[API错误] 未知响应格式"

    except ImportError:
        return "[错误] 请安装 requests 库: pip install requests"
    except Exception as e:
        return f"[错误] {str(e)}"


def _search_builtin(question):
    """在内置知识库中搜索匹配的回答"""
    question_lower = question.lower()
    for keyword, answer in BUILTIN_KNOWLEDGE.items():
        if keyword in question_lower:
            return answer
    return None


# ---------------------------------------------------------------------------
# 对话管理
# ---------------------------------------------------------------------------

class ConversationManager:
    """对话管理器，维护多轮对话上下文"""

    def __init__(self, max_history=10):
        """
        Args:
            max_history (int): 保留的历史对话轮数
        """
        self.max_history = max_history
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, question, use_api=True):
        """发送问题并获取回答

        Args:
            question (str): 用户问题
            use_api (bool): 是否使用API（False则仅用内置知识库）

        Returns:
            str: 回答内容
        """
        # 1. 先尝试内置知识库
        builtin_answer = _search_builtin(question)
        if builtin_answer and not use_api:
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": builtin_answer})
            return builtin_answer

        # 2. 调用API
        self.history.append({"role": "user", "content": question})

        # 截断历史以控制token数量
        if len(self.history) > self.max_history * 2 + 1:
            system_msg = self.history[0]
            recent = self.history[-(self.max_history * 2):]
            self.history = [system_msg] + recent

        if use_api and _api_key:
            answer = call_qwen_api(self.history)
        elif builtin_answer:
            answer = builtin_answer + "\n\n[提示：当前使用内置知识库回答，配置API Key后可获得更精准的AI回答]"
        else:
            answer = ("抱歉，内置知识库中未找到相关内容。\n"
                      "请配置通义千问API Key以获得AI智能问答功能。\n"
                      "获取免费Key: https://dashscope.aliyun.com\n"
                      "设置方法: ml_lab.llm_assistant.set_api_key('your-key')")

        self.history.append({"role": "assistant", "content": answer})
        return answer

    def clear(self):
        """清空对话历史"""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def get_history(self):
        """获取对话历史"""
        return self.history.copy()


# 全局对话管理器
_default_conversation = None


def get_default_conversation():
    """获取默认对话管理器"""
    global _default_conversation
    if _default_conversation is None:
        _default_conversation = ConversationManager()
    return _default_conversation


def ask(question):
    """快速问答接口（使用默认对话管理器）

    Args:
        question (str): 问题

    Returns:
        str: 回答
    """
    return get_default_conversation().chat(question)


def reset_conversation():
    """重置默认对话"""
    global _default_conversation
    _default_conversation = ConversationManager()


# ---------------------------------------------------------------------------
# 预设问题（常用ML概念）
# ---------------------------------------------------------------------------

PRESET_QUESTIONS = [
    "什么是梯度下降？如何理解学习率的作用？",
    "什么是过拟合？如何防止过拟合？",
    "决策树是如何进行分类的？",
    "SVM支持向量机的原理是什么？",
    "什么是交叉验证？为什么需要它？",
    "逻辑回归为什么叫'回归'却是分类算法？",
    "神经网络中的激活函数有什么作用？",
    "精确率和召回率有什么区别？",
]