# ======================== 多元统计实验：钢铁公司 =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 数据 ----------------------
companies = ['宝钢', '鞍钢', '武钢', '首钢', '浦项']
features = [
    '负债保障率', '长期负债倍数', '流动比率', '资产利润率', '收入利润率',
    '成本费用利润率', '净利润现金比率', '三年资产平均增长率', '三年销售平均增长率', '三年平均资本增长率'
]

data = np.array([
    [2.89, 5.16, 1.31, 21.71, 23.17, 30.23, 1.79, 1.48, 20.07, 11.04],
    [2.95, 9.15, 1.83, 17.34, 11.33, 12.76, 0.9, 7.28, 29.19, 10.5],
    [2.34, 6.07, 1.16, 24.77, 19.55, 24.81, 1.7, 63.3, 52.88, 48.95],
    [1.85, 2.63, 2.22, 11.89, 7.6, 8.05, 1.09, 11.76, 18.77, 7.63],
    [3.12, 6.96, 2.1, 25.34, 22.28, 28.52, 1.3, 13.18, 24.16, 17.51]
])

df = pd.DataFrame(data, index=companies, columns=features)
data_norm = (data - data.min(0)) / (data.max(0) - data.min(0))


# 1. 散点图矩阵
plt.figure(figsize=(16,14))
sns.pairplot(df, diag_kind='hist', corner=True)
plt.suptitle("五家钢铁公司经营指标散点图矩阵", y=1.02)
plt.show()


# 2. 脸谱图

fig, axs = plt.subplots(1, 5, figsize=(18, 5))  # 1行5列，只画5张脸
fig.suptitle("五家钢铁公司经营状况脸谱图", fontsize=16)

for idx, ax in enumerate(axs):
    g,d,f,p1,p2,p3,p4,a,s,c = data_norm[idx]
    # 脸
    ax.add_patch(plt.Circle((0.5,0.5), 0.25+0.15*g, fc='#FFEBB2', ec='k'))
    # 眼睛
    es = 0.03+0.06*d
    ax.add_patch(plt.Circle((0.4,0.58), es, fc='white', ec='k'))
    ax.add_patch(plt.Circle((0.6,0.58), es, fc='white', ec='k'))
    ax.add_patch(plt.Circle((0.4,0.58), 0.02, fc='k'))
    ax.add_patch(plt.Circle((0.6,0.58), 0.02, fc='k'))
    # 鼻子
    ax.plot([0.5,0.5],[0.5, 0.4-0.1*f],'k',lw=1.5)
    # 嘴巴
    mw = 0.1+0.12*p1
    my = 0.32 - 0.08*p2
    ax.plot([0.5-mw/2,0.5+mw/2],[my,my],'k',lw=2)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    ax.set_title(companies[idx])
    ax.axis('off')

plt.tight_layout()
plt.show()


# 3. 雷达图
angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(12,12), subplot_kw=dict(polar=True))
colors = ['#FF6B6B','#4ECDC4','#45B7D1','#FFA07A','#98D8C8']

for i in range(5):
    v = data_norm[i].tolist()
    v += v[:1]
    ax.plot(angles, v, lw=2, label=companies[i], c=colors[i])
    ax.fill(angles, v, alpha=0.1, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=11)
plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
plt.title("五家钢铁公司经营指标雷达图", fontsize=16, pad=20)
plt.tight_layout()
plt.show()