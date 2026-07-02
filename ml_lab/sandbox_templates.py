# -*- coding: utf-8 -*-
"""ML-Lab 代码沙箱模板。

共 11 个预置代码模板，覆盖数据查看、基础图表和机器学习流程。
"""

SANDBOX_TEMPLATES = {
    "默认（查看数据）": """# ML-Lab 代码沙箱 — 在此编写和运行 Python 代码

# 已预导入: numpy (as np), pandas (as pd), sklearn, matplotlib, os, json
# 数据变量: X, y, X_train, y_train, X_test, y_test, feature_names, target_names
# 提示: 可直接在下方上传 CSV/Excel 文件，无需前往数据工作台

import numpy as np
import pandas as pd

# 查看数据形状
print("数据形状:", X.shape)
print("前5行:\\\\n", pd.DataFrame(X[:5], columns=feature_names))
""",

    "基础折线图": """import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制折线图
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y1, label='sin(x)', linewidth=2, color='#4F46E5')
ax.plot(x, y2, label='cos(x)', linewidth=2, color='#10B981', linestyle='--')
ax.set_title('三角函数图像', fontsize=16)
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('y', fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#FAFAFA')
plt.tight_layout()
plt.show()
""",

    "柱状图": """import matplotlib.pyplot as plt
import numpy as np

# 示例数据
categories = ['语文', '数学', '英语', '物理', '化学']
scores_A = [85, 92, 78, 88, 76]
scores_B = [90, 80, 85, 72, 82]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, scores_A, width, label='班级A', color='#4F46E5', edgecolor='white')
bars2 = ax.bar(x + width/2, scores_B, width, label='班级B', color='#F59E0B', edgecolor='white')

ax.set_title('各科成绩对比', fontsize=16)
ax.set_xlabel('科目', fontsize=13)
ax.set_ylabel('分数', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#FAFAFA')

# 数值标签
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
""",

    "散点图": """import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# 生成聚类数据
X_blob, y_blob = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
colors = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444']

fig, ax = plt.subplots(figsize=(8, 6))
for i, c in enumerate(colors):
    mask = y_blob == i
    ax.scatter(X_blob[mask, 0], X_blob[mask, 1], c=c, s=30, alpha=0.7,
               label=f'类别 {i}', edgecolors='white', linewidth=0.5)

ax.set_title('散点图 — 聚类数据可视化', fontsize=16)
ax.set_xlabel('特征 1', fontsize=13)
ax.set_ylabel('特征 2', fontsize=13)
ax.legend(fontsize=11, markerscale=1.5)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#FAFAFA')
plt.tight_layout()
plt.show()
""",

    "饼图": """import matplotlib.pyplot as plt

# 数据
labels = ['Python', 'JavaScript', 'Java', 'C/C++', 'Go', '其他']
sizes = [28, 22, 16, 12, 8, 14]
colors = ['#4F46E5', '#F59E0B', '#EF4444', '#10B981', '#8B5CF6', '#94A3B8']
explode = (0.05, 0, 0, 0, 0, 0)  # 突出Python

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', startangle=140, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
)

for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax.set_title('编程语言使用占比（环形图）', fontsize=16, pad=20)
plt.tight_layout()
plt.show()
""",

    "直方图 + 箱线图": """import matplotlib.pyplot as plt
import numpy as np

# 生成正态分布数据
np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 直方图
ax1.hist(data, bins=30, color='#4F46E5', edgecolor='white', alpha=0.85)
ax1.axvline(data.mean(), color='#EF4444', linestyle='--', linewidth=2, label=f'均值={data.mean():.1f}')
ax1.axvline(np.median(data), color='#10B981', linestyle='--', linewidth=2, label=f'中位数={np.median(data):.1f}')
ax1.set_title('成绩分布直方图', fontsize=14)
ax1.set_xlabel('分数', fontsize=12)
ax1.set_ylabel('人数', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_facecolor('#FAFAFA')

# 箱线图
bp = ax2.boxplot(data, patch_artist=True, widths=0.4,
                  boxprops=dict(facecolor='#DBEAFE', edgecolor='#4F46E5', linewidth=1.5),
                  medianprops=dict(color='#EF4444', linewidth=2),
                  whiskerprops=dict(color='#4F46E5'),
                  capprops=dict(color='#4F46E5'),
                  flierprops=dict(marker='o', markerfacecolor='#F59E0B', markersize=5))
ax2.set_title('成绩箱线图', fontsize=14)
ax2.set_ylabel('分数', fontsize=12)
ax2.set_xticklabels(['成绩'])
ax2.grid(axis='y', alpha=0.3)
ax2.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.show()
""",

    "热力图（相关系数矩阵）": """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 使用已加载的数据集
df = pd.DataFrame(X, columns=feature_names)
if hasattr(df.iloc[0, 0], '__float__') or np.issubdtype(df.iloc[0, 0].dtype, np.number):
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(corr.columns, fontsize=10)

    # 数值标注
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='相关系数')
    ax.set_title('特征相关系数热力图', fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print("当前数据集不适合绘制相关系数矩阵（非数值型特征）")
""",

    "3D 散点图": """import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D

# 生成瑞士卷数据
X_sr, t = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_sr[:, 0], X_sr[:, 1], X_sr[:, 2],
                     c=t, cmap='viridis', s=10, alpha=0.8, edgecolors='none')
ax.set_title('3D 散点图 — 瑞士卷 (Swiss Roll)', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.colorbar(scatter, ax=ax, shrink=0.6, label='t 参数')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.show()
""",

    "子图组合（多图排列）": """import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 50)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('子图组合示例', fontsize=16, fontweight='bold')

# 左上：折线图
axes[0, 0].plot(x, np.sin(x), color='#4F46E5', linewidth=2)
axes[0, 0].set_title('折线图')
axes[0, 0].grid(True, alpha=0.3)

# 右上：柱状图
axes[0, 1].bar(['A', 'B', 'C', 'D'], [15, 30, 45, 20], color=['#4F46E5', '#10B981', '#F59E0B', '#EF4444'])
axes[0, 1].set_title('柱状图')

# 左下：散点图
axes[1, 0].scatter(np.random.randn(50), np.random.randn(50), c='#8B5CF6', s=40, alpha=0.7)
axes[1, 0].set_title('散点图')
axes[1, 0].grid(True, alpha=0.3)

# 右下：饼图
axes[1, 1].pie([35, 25, 20, 20], labels=['Q1', 'Q2', 'Q3', 'Q4'],
               colors=['#4F46E5', '#10B981', '#F59E0B', '#EF4444'],
               autopct='%1.0f%%', startangle=90,
               wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1, 1].set_title('饼图')

plt.tight_layout()
plt.show()
""",

    "机器学习流程（分类）": """# 使用已加载数据集完成分类流程
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print("=== 分类报告 ===")
print(classification_report(y_test, y_pred, target_names=target_names))

print(f"训练集准确率: {model.score(X_train, y_train):.4f}")
print(f"测试集准确率: {model.score(X_test, y_test):.4f}")

# 特征重要性
if hasattr(model, 'feature_importances_'):
    import numpy as np
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=target_names, yticklabels=target_names)
    ax1.set_title('混淆矩阵', fontsize=14)
    ax1.set_xlabel('预测值')
    ax1.set_ylabel('真实值')
    
    # 特征重要性
    importance.plot(kind='barh', color='#4F46E5', ax=ax2)
    ax2.set_title('特征重要性 Top 10', fontsize=14)
    ax2.set_xlabel('重要性')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    plt.show()
""",

    "机器学习流程（回归）": """# 使用已加载数据集完成回归流程
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 训练两个模型
models = {
    '线性回归': LinearRegression(),
    '随机森林': RandomForestRegressor(n_estimators=100, random_state=42)
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'{name}: R²={r2:.4f}, RMSE={rmse:.4f}')
    
    # 预测值 vs 真实值
    ax = axes[idx]
    ax.scatter(y_test, y_pred, alpha=0.5, s=30, color='#4F46E5', edgecolors='white')
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
    ax.set_title(f'{name}\\nR²={r2:.4f}', fontsize=13)
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#FAFAFA')
    ax.set_aspect('equal', adjustable='box')

plt.suptitle('回归模型: 预测值 vs 真实值', fontsize=15)
plt.tight_layout()
plt.show()
""",
}
