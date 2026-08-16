# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ==============================
# 1. 数据输入
# ==============================
data = {
    "地区": ["北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海", "江苏",
             "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "广西",
             "海南", "重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"],
    "未上过学": [313, 333, 2752, 909, 1013, 782, 675, 1295, 651, 3908,
                 2916, 3549, 1953, 1830, 5230, 4323, 2690, 1984, 3194, 1697,
                 332, 1070, 5508, 3117, 3219, 982, 1685, 1793, 605, 350, 790],
    "小学": [1631, 1902, 14776, 5620, 4312, 6838, 5106, 6952, 2528, 14012,
             12079, 13091, 9407, 10875, 18766, 17933, 10885, 13315, 18686, 10350,
             1570, 7581, 21287, 9233, 14306, 786, 6847, 6813, 1602, 1394, 5455],
    "初中": [4070, 4169, 25765, 12675, 7552, 15399, 9333, 13491, 6199, 22540,
             15554, 20802, 10337, 13502, 30835, 32979, 17461, 20403, 33330, 16376,
             3197, 8112, 23637, 9951, 12135, 466, 11835, 6337, 1390, 1821, 6743],
    "高中": [3258, 2821, 8625, 5907, 3395, 5787, 3653, 4972, 4104, 12036,
             6890, 6226, 4570, 6011, 13025, 12638, 8563, 11213, 18179, 5616,
             1253, 4287, 8958, 2967, 4060, 146, 5701, 3310, 540, 844, 2618],
    "大专及以上": [7729, 3176, 5966, 3941, 3640, 6331, 3093, 4159, 5791, 10458,
                   6705, 4515, 3421, 3177, 9499, 5870, 6406, 6184, 11779, 2954,
                   685, 3038, 5869, 1905, 3210, 132, 3822, 2191, 444, 801, 2483]
}

df = pd.DataFrame(data)
df.set_index("地区", inplace=True)


# ==============================
# 2. 对应分析函数
# ==============================
def correspondence_analysis(df, n_components=2):
    """
    对应分析主函数
    df: 行是样本（省份），列是类别（教育程度）
    """
    # 总频数矩阵
    N = df.values
    total = np.sum(N)

    # 概率矩阵
    P = N / total

    # 行和、列和
    r = np.sum(P, axis=1)  # 行和
    c = np.sum(P, axis=0)  # 列和

    # 计算残差矩阵
    S = np.zeros_like(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            S[i, j] = (P[i, j] - r[i] * c[j]) / np.sqrt(r[i] * c[j])

    # 奇异值分解
    U, D, Vt = svd(S, full_matrices=False)

    # 行坐标（标准坐标）
    row_coords = np.zeros((P.shape[0], n_components))
    for k in range(n_components):
        row_coords[:, k] = U[:, k] * D[k] / np.sqrt(r)

    # 列坐标（标准坐标）
    col_coords = np.zeros((P.shape[1], n_components))
    for k in range(n_components):
        col_coords[:, k] = Vt[k, :] * D[k] / np.sqrt(c)

    # 惯量
    inertia = D ** 2
    total_inertia = np.sum(inertia)

    return row_coords, col_coords, inertia, total_inertia, r, c


# ==============================
# 3. 执行对应分析
# ==============================
row_coords, col_coords, inertia, total_inertia, r, c = correspondence_analysis(df)

# 输出总惯量
print(f"总惯量 (Total Inertia): {total_inertia:.6f}")
print("\n各维度惯量及解释比例:")
for i, inert in enumerate(inertia[:3]):
    print(f"维度 {i + 1}: 惯量 = {inert:.6f}, 解释比例 = {inert / total_inertia:.2%}")

# ==============================
# 4. 可视化
# ==============================
plt.figure(figsize=(12, 8))

# 绘制省份（行）
for i, province in enumerate(df.index):
    plt.scatter(row_coords[i, 0], row_coords[i, 1], marker='o', color='blue', alpha=0.7)
    plt.text(row_coords[i, 0], row_coords[i, 1], province, fontsize=9, ha='right', color='blue')

# 绘制教育程度（列）
edu_levels = df.columns
for j, edu in enumerate(edu_levels):
    plt.scatter(col_coords[j, 0], col_coords[j, 1], marker='s', color='red', alpha=0.7)
    plt.text(col_coords[j, 0], col_coords[j, 1], edu, fontsize=10, ha='left', color='red', weight='bold')

# 辅助线
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.title("对应分析图 (2016年各省份教育程度)", fontsize=14)
plt.xlabel(f"维度1 (惯量比例: {inertia[0] / total_inertia:.2%})")
plt.ylabel(f"维度2 (惯量比例: {inertia[1] / total_inertia:.2%})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# # ==============================
# # 5. 结果解释（对应课后思考题）
# # ==============================
# print("\n===== 对应分析思想与特点 =====")
# print("思想：将列联表的行和列同时投影到低维空间，揭示行与列之间的关联关系。")
# print("特点：")
# print("1. 同时分析行和列，便于发现类别间的关联。")
# print("2. 适用于频数表或列联表。")
# print("3. 结果可通过二维图直观展示。")
# print("\n总惯量意义：")
# print("总惯量表示行与列之间的总关联强度，类似于卡方检验中的总卡方值除以总频数。")
# print("惯量越大，说明行与列之间的关联越强。")