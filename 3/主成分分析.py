import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置中文显示（解决matplotlib中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 构建52名学生的6门课程成绩数据（x1数学,x2物理,x3化学,x4语文,x5历史,x6英语）
data = np.array([
    [65,61,72,84,81,79], [77,77,76,64,70,55], [67,63,49,65,67,57],
    [78,84,75,62,71,64], [66,71,67,52,65,57], [83,100,79,41,67,50],
    [86,94,97,51,63,55], [67,84,53,58,66,56], [69,56,67,75,94,80],
    [77,90,80,68,66,60], [84,67,75,60,70,63], [62,67,83,71,85,77],
    [91,74,97,62,71,66], [82,70,83,68,77,85], [66,61,77,62,73,64],
    [90,78,78,59,72,66], [77,89,80,73,75,70], [72,68,77,83,92,79],
    [72,67,61,92,92,88], [81,90,79,73,85,80], [68,85,70,84,89,86],
    [85,91,95,63,76,66], [91,85,100,70,65,76], [74,74,84,61,80,69],
    [88,100,85,49,71,66], [87,84,100,74,81,76], [64,79,64,72,76,74],
    [60,51,60,78,74,76], [59,75,81,82,77,73], [64,61,49,100,99,95],
    [56,48,61,85,82,80], [62,45,67,78,76,82], [86,78,92,87,87,77],
    [80,98,83,58,66,66], [83,71,81,63,77,73], [67,83,65,68,74,60],
    [71,58,45,83,77,73], [90,83,91,58,60,59], [73,80,64,75,80,78],
    [87,98,87,68,78,64], [69,72,79,89,82,73], [79,73,69,65,73,73],
    [87,86,88,70,73,70], [76,61,73,63,60,70], [99,100,99,53,63,60],
    [78,68,52,75,74,66], [72,90,73,76,80,79], [69,64,60,68,74,80],
    [52,62,65,100,96,100], [70,72,56,74,82,74], [72,74,75,88,91,86],
    [68,74,70,87,87,83]
])
# 转换为DataFrame，方便查看
df = pd.DataFrame(data, columns=['数学x1', '物理x2', '化学x3', '语文x4', '历史x5', '英语x6'])
print("原始成绩数据前5行：")
print(df.head())
print("\n数据基本统计信息：")
print(df.describe())

# 标准化处理
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)  # 标准化后的数组
print("\n标准化后数据前5行：")
print(pd.DataFrame(data_scaled, columns=df.columns).head())

# 构建PCA模型，提取所有6个主成分
pca = PCA(n_components=None)  # n_components=None表示保留所有主成分
pca_result = pca.fit_transform(data_scaled)  # 主成分分析结果

# 1. 主成分的方差贡献率
explained_variance_ratio = pca.explained_variance_ratio_
# 2. 累计方差贡献率
cum_explained_variance = np.cumsum(explained_variance_ratio)
# 3. 主成分系数（载荷矩阵）
loadings = pd.DataFrame(pca.components_.T,
                        columns=[f'主成分{i+1}' for i in range(6)],
                        index=df.columns)

# 输出主成分分析关键结果
print("\n===== 主成分载荷矩阵（系数） =====")
print(loadings.round(3))  # 保留3位小数
print("\n===== 各主成分方差贡献率 =====")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"主成分{i+1}：{ratio:.3f} ({ratio*100:.1f}%)")
print("\n===== 各主成分累计方差贡献率 =====")
for i, cum_ratio in enumerate(cum_explained_variance):
    print(f"前{i+1}个主成分：{cum_ratio:.3f} ({cum_ratio*100:.1f}%)")

# 提取主成分得分（每个样本在各主成分上的取值）
pca_score = pd.DataFrame(pca_result, columns=[f'主成分{i+1}得分' for i in range(6)])
print("\n主成分得分前5行：")
print(pca_score.head())

# 绘制方差贡献率与累计方差贡献率图
plt.figure(figsize=(10, 6))
# 柱状图：单个主成分方差贡献率
plt.bar(range(1,7), explained_variance_ratio, color='lightblue', label='单个方差贡献率')
# 折线图：累计方差贡献率
plt.plot(range(1,7), cum_explained_variance, color='red', marker='o', label='累计方差贡献率')
# 标注数值
for i, (ratio, cum_ratio) in enumerate(zip(explained_variance_ratio, cum_explained_variance)):
    plt.text(i+1, ratio+0.02, f'{ratio*100:.1f}%', ha='center')
    plt.text(i+1, cum_ratio-0.05, f'{cum_ratio*100:.1f}%', ha='center')
# 图表设置
plt.xlabel('主成分个数')
plt.ylabel('方差贡献率')
plt.title('主成分方差贡献率与累计方差贡献率')
plt.xticks(range(1,7))
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()