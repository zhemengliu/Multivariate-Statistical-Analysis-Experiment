import numpy as np
import pandas as pd
from scipy.stats import f, chi2, t
from scipy.linalg import inv, det

# ---------------------- 1. 数据准备 ----------------------
# 样本数据（9个地区，5个指标）
data = np.array([
    [5068, 31.1, 2141, 8.23, 15.83],  # 内蒙古
    [4076, 34.2, 2040, 9.01, 13.32],  # 广西
    [2342, 29.8, 1551, 14.26, 28.98], # 贵州
    [4355, 31.1, 2059, 12.10, 25.48], # 云南
    [3716, 43.5, 1551, 15.90, 57.97], # 西藏
    [4270, 37.3, 1947, 13.08, 25.56], # 宁夏
    [6229, 35.4, 2745, 12.81, 11.44], # 新疆
    [3456, 32.8, 1612, 10.04, 28.65], # 甘肃
    [4367, 40.9, 2047, 14.48, 42.92]  # 青海
])

# 全国平均水平（μ0，1998年统计年鉴真实数据）
mu0 = np.array([6392, 32.1, 2972, 9.14, 15.78])

n, p = data.shape  # n=9样本量，p=5指标维度
print(f"样本量n={n}, 指标维度p={p}")

# ---------------------- 2. 计算样本统计量 ----------------------
x_bar = np.mean(data, axis=0)  # 样本均值向量
S = np.cov(data, rowvar=False) # 样本协方差阵（无偏估计）
S_inv = inv(S)                 # 协方差阵的逆

print("\n样本均值向量（边远地区）：")
print(pd.Series(x_bar, index=["人均GDP", "第三产业比重", "人均消费支出", "人口自然增长率", "文盲半文盲占比"]))
print("\n样本协方差阵：")
print(pd.DataFrame(S, columns=["人均GDP", "第三产业比重", "人均消费支出", "人口自然增长率", "文盲半文盲占比"],
                  index=["人均GDP", "第三产业比重", "人均消费支出", "人口自然增长率", "文盲半文盲占比"]))

# ---------------------- 3. 多元均值检验 ----------------------
# 计算T²统计量
diff = x_bar - mu0
T2 = n * diff.T @ S_inv @ diff

# 转换为F统计量
F_stat = (n - p) / (p * (n - 1)) * T2
# 计算p值
p_value = 1 - f.cdf(F_stat, dfn=p, dfd=n-p)

# 临界值（α=0.05）
F_crit = f.ppf(0.95, dfn=p, dfd=n-p)

print("\n===== 多元均值检验结果 =====")
print(f"T²统计量: {T2:.4f}")
print(f"转换F统计量: {F_stat:.4f}")
print(f"F临界值(α=0.05): {F_crit:.4f}")
print(f"p值: {p_value:.6f}")

if p_value < 0.05:
    print("结论：拒绝原假设H0，边远地区社会经济发展水平与全国平均水平存在显著差异")
else:
    print("结论：不拒绝原假设H0，无显著差异")

# ---------------------- 4. 协方差阵检验（似然比检验） ----------------------
# 假设全国协方差阵为Σ0（这里用对角协方差阵，实际可替换为全国真实协方差阵）
Sigma0 = np.diag(np.var(data, axis=0, ddof=1))

# 似然比统计量
log_lambda = (n-1) * (np.log(det(Sigma0)) - np.log(det(S)) + np.trace(S_inv @ Sigma0) - p)
# 转换为卡方统计量
chi2_stat = -2 * log_lambda
chi2_p = 1 - chi2.cdf(chi2_stat, df=p*(p+1)//2)  # 自由度为p(p+1)/2

print("\n===== 协方差阵似然比检验结果 =====")
print(f"卡方统计量: {chi2_stat:.4f}")
print(f"自由度: {p*(p+1)//2}")
print(f"p值: {chi2_p:.6f}")

if chi2_p < 0.05:
    print("结论：拒绝原假设H0，边远地区协方差阵与假设协方差阵存在显著差异")
else:
    print("结论：不拒绝原假设H0，无显著差异")

# ---------------------- 5. 单指标t检验 ----------------------
print("\n===== 单指标t检验  =====")
for i, name in enumerate(["人均GDP", "第三产业比重", "人均消费支出", "人口自然增长率", "文盲半文盲占比"]):
    s_i = np.sqrt(S[i,i])
    t_stat = (x_bar[i] - mu0[i]) / (s_i / np.sqrt(n))
    t_p = 2 * (1 - t.cdf(abs(t_stat), df=n-1))
    print(f"{name}: t统计量={t_stat:.4f}, p值={t_p:.6f}")