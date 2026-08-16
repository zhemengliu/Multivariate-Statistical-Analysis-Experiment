import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import warnings
warnings.filterwarnings('ignore')


# 解决matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 构建52名学生6门课程成绩数据（x1数学,x2物理,x3化学,x4语文,x5历史,x6英语）
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

# 转换为DataFrame，设置列名
df = pd.DataFrame(data, columns=['数学x1', '物理x2', '化学x3', '语文x4', '历史x5', '英语x6'])
print("原始成绩数据前5行：")
print(df.head())
print("\n数据基本统计信息：")
print(df.describe().round(2))

# 1. 数据标准化（Z-score，均值0，方差1）
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(data_scaled, columns=df.columns)
print("\n标准化后数据前5行：")
print(df_scaled.head().round(3))

# 2. 因子分析适用性检验
# KMO检验：取值0~1，>0.6表示适合因子分析，>0.8表示非常适合
kmo_all, kmo_model = calculate_kmo(df_scaled)
# 巴特利特球形检验：p<0.05表示拒绝原假设，变量间存在显著相关性
bartlett_stats, bartlett_p = calculate_bartlett_sphericity(df_scaled)

print("\n===== 因子分析适用性检验 =====")
print(f"KMO检验值：{kmo_model:.3f}")
print(f"巴特利特球形检验统计量：{bartlett_stats:.2f}，p值：{bartlett_p:.4f}")

# 第一步：先提取所有6个因子，查看特征值，确定最优因子个数
fa = FactorAnalyzer(n_factors=6, rotation=None, method='principal')  # 未旋转，主成分法
fa.fit(df_scaled)

# 查看特征值、方差贡献率、累计方差贡献率
eigen_values = fa.get_eigenvalues()[0]  # 特征值
factor_variance = fa.get_factor_variance()  # 方差贡献率、累计方差贡献率
variance_df = pd.DataFrame({
    '公因子': [f'因子{i+1}' for i in range(6)],
    '特征值': eigen_values.round(3),
    '方差贡献率': factor_variance[0].round(3),
    '累计方差贡献率': factor_variance[2].round(3)
})

print("\n===== 各公因子特征值与方差贡献率 =====")
print(variance_df)

# 绘制碎石图：直观判断因子个数（特征值>1为有效因子）
plt.figure(figsize=(10, 6))
plt.plot(range(1,7), eigen_values, marker='o', linestyle='-', color='red')
plt.axhline(y=1, color='blue', linestyle='--', label='特征值=1')
plt.xlabel('公因子个数')
plt.ylabel('特征值')
plt.title('因子分析碎石图（确定公因子个数）')
plt.xticks(range(1,7))
plt.legend()
plt.grid(alpha=0.7)
plt.show()

# 第二步：提取最优个数的公因子（根据特征值>1，本次为2个），并做正交旋转（方差最大旋转）
# 方差最大旋转：让因子载荷矩阵更清晰，便于解读因子含义
fa_final = FactorAnalyzer(n_factors=2, rotation='varimax', method='principal')
fa_final.fit(df_scaled)

# 1. 因子载荷矩阵（核心，反映变量与公因子的相关性）
loadings = fa_final.loadings_
loadings_df = pd.DataFrame(loadings, index=df.columns, columns=['公因子1', '公因子2'])
print("\n===== 旋转后因子载荷矩阵（保留3位小数） =====")
print(loadings_df.round(3))

# 2. 最终因子方差贡献率（旋转后）
final_variance = fa_final.get_factor_variance()
final_variance_df = pd.DataFrame({
    '指标': ['方差贡献率', '累计方差贡献率'],
    '公因子1': [final_variance[0][0].round(3), final_variance[2][0].round(3)],
    '公因子2': [final_variance[0][1].round(3), final_variance[2][1].round(3)]
})
print("\n===== 旋转后公因子方差贡献率 =====")
print(final_variance_df)

# 3. 计算每个样本的因子得分
factor_scores = fa_final.transform(df_scaled)
factor_scores_df = pd.DataFrame(factor_scores, columns=['公因子1得分', '公因子2得分'])
print("\n===== 样本因子得分前5行（保留3位小数） =====")
print(factor_scores_df.head().round(3))

# 4. 因子得分可视化：散点图展示52名学生在两个公因子上的分布
plt.figure(figsize=(10, 6))
plt.scatter(factor_scores_df['公因子1得分'], factor_scores_df['公因子2得分'], alpha=0.7, color='orange')
plt.xlabel('公因子1得分')
plt.ylabel('公因子2得分')
plt.title('52名学生成绩因子得分散点图')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.grid(alpha=0.7)
plt.show()