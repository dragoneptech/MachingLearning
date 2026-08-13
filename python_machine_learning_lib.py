# Python 机器学习库

# 机器学习中最核心的四个 Python 库：NumPy、Pandas、Matplotlib 和 Scikit-learn

# 四大核心库的角色
# Numpy：数值计算的基础，提供高效的数组操作
# Pandas：数据处理的利器，提供数据结构和分析工具
# Matplotlib：数据可视化的画笔，创建各种图表
# scikit-learn：机器学习的瑞士军刀，提供完整的 ML 工具链

# NumPy：数值计算的基础

# NumPy 的核心概念
# 数组（Array）

# NumPy 数组基础操作
# import numpy as np

# # 创建数组的不同方式
# print("=== NumPy 数组创建 ===")

# # 从列表创建
# arr1 = np.array([1, 2, 3, 4, 5])
# print(f"从列表创建：{arr1}")

# # 创建等差数组
# arr2 = np.arange(0, 10, 2)  # 0到10，步长为2
# print(f"等差数组：{arr2}")

# # 创建等间隔数组
# arr3 = np.linspace(0, 1, 5)  # 0到1，5个点
# print(f"等间隔数组：{arr3}")

# # 创建特殊数组
# zeros_arr = np.zeros((2, 3))  # 2行3列的零数组
# ones_arr = np.ones((2, 3))    # 2行3列的一数组
# identity_arr = np.eye(3)      # 3x3单位矩阵

# print(f"零数组：\n{zeros_arr}")
# print(f"一数组：\n{ones_arr}")
# print(f"单位矩阵：\n{identity_arr}")

# 数组操作

# 数组的基本操作
# import numpy as np
# print("\n=== 数组基本操作 ===")

# # 数组属性
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# print(f"数组：\n{arr}")
# print(f"形状：{arr.shape}")
# print(f"维度：{arr.ndim}")
# print(f"元素个数：{arr.size}")
# print(f"数据类型：{arr.dtype}")

# # 数组索引和切片
# print(f"第一行：{arr[0]}")
# print(f"第一列：{arr[:, 0]}")
# print(f"元素[1,2]：{arr[1, 2]}")

# # 数组运算
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 5, 6])

# print(f"加法：{arr1 + arr2}")
# print(f"乘法：{arr1 * arr2}")
# print(f"点积：{np.dot(arr1, arr2)}")

# # 统计函数
# data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# print(f"均值：{np.mean(data)}")
# print(f"标准差：{np.std(data)}")
# print(f"最大值：{np.max(data)}")
# print(f"最小值：{np.min(data)}")
# print(f"中位数：{np.median(data)}")

# NumPy 实际应用示例
# NumPy 实际应用：简单线性回归
# import numpy as np
# def numpy_linear_regression():
#     """使用 NumPy 实现简单线性回归"""
    
#     # 生成示例数据
#     np.random.seed(42)
#     X = 2 * np.random.rand(100, 1)  # 特征
#     y = 4 + 3 * X + np.random.randn(100, 1)  # 标签 + 噪声
    
#     # 添加 x0 = 1 到 X
#     X_b = np.c_[np.ones((100, 1)), X]  # 添加偏置项
    
#     # 使用正规方程求解：θ = (X^T * X)^(-1) * X^T * y
#     theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
#     print("=== NumPy 线性回归示例 ===")
#     print(f"学习到的参数：截距={theta_best[0][0]:.2f}, 斜率={theta_best[1][0]:.2f}")
    
#     # 预测
#     X_new = np.array([[0], [2]])
#     X_new_b = np.c_[np.ones((2, 1)), X_new]
#     y_predict = X_new_b.dot(theta_best)
    
#     print(f"预测结果：X=0 时 y={y_predict[0][0]:.2f}, X=2 时 y={y_predict[1][0]:.2f}")
    
#     return theta_best, X, y

# # 运行示例
# theta, X, y = numpy_linear_regression()

# Pandas：数据处理的利器

# Pandas 的核心数据结构

# Series（一维数据）

# Pandas Series 基础操作
# import pandas as pd

# print("=== Pandas Series ===")

# # 从列表创建 Series
# s1 = pd.Series([1, 2, 3, 4, 5])
# print(f"从列表创建：\n{s1}")

# # 带索引的 Series
# s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
# print(f"\n带索引的 Series：\n{s2}")

# # 从字典创建 Series
# s3 = pd.Series({'数学': 90, '英语': 85, '物理': 88})
# print(f"\n从字典创建：\n{s3}")

# # Series 操作
# print(f"\n访问元素：s2['b'] = {s2['b']}")
# print(f"切片：s2[0:2] =\n{s2[0:2]}")
# print(f"统计信息：\n{s2.describe()}")

# DataFrame（二维数据）
# Pandas DataFrame 基础操作
# import pandas as pd
# print("\n=== Pandas DataFrame ===")

# # 创建 DataFrame
# data = {
#     '姓名': ['张三', '李四', '王五', '赵六'],
#     '年龄': [25, 30, 35, 28],
#     '城市': ['北京', '上海', '广州', '深圳'],
#     '薪资': [15000, 20000, 18000, 22000]
# }

# df = pd.DataFrame(data)
# print("原始 DataFrame：")
# print(df)

# # DataFrame 基本操作
# print(f"\nDataFrame 形状：{df.shape}")
# print(f"\n列名：{list(df.columns)}")
# print(f"\n数据类型：\n{df.dtypes}")

# # 选择数据
# print(f"\n选择'姓名'列：\n{df['姓名']}")
# print(f"\n选择前两行：\n{df.head(2)}")
# print(f"\n选择年龄大于28的行：\n{df[df['年龄'] > 28]}")

# # 统计信息
# print(f"\n数值列的统计信息：\n{df.describe()}")

# # 添加新列
# df['年薪'] = df['薪资'] * 12
# print(f"\n添加年薪列后：\n{df}")

# Pandas 数据处理示例
# Pandas 数据处理完整示例
# import pandas as pd
# import numpy as np
# def pandas_data_processing():
#     """演示 Pandas 数据处理的完整流程"""
    
#     print("=== Pandas 数据处理示例 ===")
    
#     # 1. 创建示例数据
#     np.random.seed(42)
#     n_samples = 1000
    
#     data = {
#         '学生ID': range(1, n_samples + 1),
#         '姓名': [f'学生{i}' for i in range(1, n_samples + 1)],
#         '年龄': np.random.randint(18, 25, n_samples),
#         '性别': np.random.choice(['男', '女'], n_samples),
#         '数学成绩': np.random.normal(75, 15, n_samples),
#         '英语成绩': np.random.normal(80, 12, n_samples),
#         '物理成绩': np.random.normal(72, 18, n_samples),
#         '班级': np.random.choice(['一班', '二班', '三班'], n_samples)
#     }
    
#     df = pd.DataFrame(data)
    
#     # 2. 数据清洗
#     print("原始数据形状：", df.shape)
    
#     # 处理异常值（成绩应在 0-100 之间）
#     score_columns = ['数学成绩', '英语成绩', '物理成绩']
#     for col in score_columns:
#         df[col] = df[col].clip(0, 100)
    
#     # 3. 特征工程
#     # 计算总分和平均分
#     df['总分'] = df[score_columns].sum(axis=1)
#     df['平均分'] = df[score_columns].mean(axis=1)
    
#     # 添加等级
#     def get_grade(score):
#         if score >= 90:
#             return 'A'
#         elif score >= 80:
#             return 'B'
#         elif score >= 70:
#             return 'C'
#         elif score >= 60:
#             return 'D'
#         else:
#             return 'F'
    
#     df['等级'] = df['平均分'].apply(get_grade)
    
#     # 4. 数据分析
#     print("\n=== 数据分析结果 ===")
    
#     # 基本统计
#     print("各科平均分：")
#     print(df[score_columns].mean())
    
#     # 按班级分析
#     print("\n各班级平均分：")
#     class_avg = df.groupby('班级')['平均分'].mean()
#     print(class_avg)
    
#     # 按性别分析
#     print("\n性别分布：")
#     gender_count = df['性别'].value_counts()
#     print(gender_count)
    
#     # 等级分布
#     print("\n等级分布：")
#     grade_dist = df['等级'].value_counts().sort_index()
#     print(grade_dist)
    
#     # 5. 数据筛选
#     print("\n=== 特定数据筛选 ===")
    
#     # 优秀学生（平均分 > 85）
#     excellent_students = df[df['平均分'] > 85].sort_values('平均分', ascending=False).head(5)
#     print("优秀学生（前5名）：")
#     print(excellent_students[['姓名', '平均分', '等级']])
    
#     # 各班级最高分学生
#     print("\n各班级最高分学生：")
#     top_students = df.loc[df.groupby('班级')['平均分'].idxmax()]
#     print(top_students[['班级', '姓名', '平均分']])
    
#     return df

# # 运行示例
# student_df = pandas_data_processing()

# Matplotlib：数据可视化的画笔

# Matplotlib 基础图表

# Matplotlib 基础图表示例
# import matplotlib.pyplot as plt
# import numpy as np

# # 设置中文字体（防止中文显示为方框）
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# def matplotlib_basic_charts():
#     """演示 Matplotlib 基础图表"""
    
#     print("=== Matplotlib 基础图表示例 ===")
    
#     # 1. 折线图
#     plt.figure(figsize=(12, 8))
    
#     plt.subplot(2, 3, 1)
#     x = np.linspace(0, 10, 100)
#     y1 = np.sin(x)
#     y2 = np.cos(x)
#     plt.plot(x, y1, label='sin(x)')
#     plt.plot(x, y2, label='cos(x)')
#     plt.title('三角函数')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.grid(True)
    
#     # 2. 散点图
#     plt.subplot(2, 3, 2)
#     np.random.seed(42)
#     x = np.random.randn(100)
#     y = 2 * x + np.random.randn(100) * 0.5
#     plt.scatter(x, y, alpha=0.6, c='blue')
#     plt.title('散点图')
#     plt.xlabel('X')
#     plt.ylabel('Y')
    
#     # 3. 柱状图
#     plt.subplot(2, 3, 3)
#     categories = ['A', 'B', 'C', 'D', 'E']
#     values = [23, 45, 56, 78, 32]
#     plt.bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])
#     plt.title('柱状图')
#     plt.xlabel('类别')
#     plt.ylabel('数值')
    
#     # 4. 直方图
#     plt.subplot(2, 3, 4)
#     data = np.random.normal(100, 15, 1000)
#     plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
#     plt.title('直方图')
#     plt.xlabel('数值')
#     plt.ylabel('频数')
    
#     # 5. 饼图
#     plt.subplot(2, 3, 5)
#     sizes = [30, 25, 20, 15, 10]
#     labels = ['A', 'B', 'C', 'D', 'E']
#     colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'plum']
#     plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#     plt.title('饼图')
    
#     # 6. 箱线图
#     plt.subplot(2, 3, 6)
#     data1 = np.random.normal(0, 1, 100)
#     data2 = np.random.normal(2, 1, 100)
#     data3 = np.random.normal(-2, 1, 100)
#     plt.boxplot([data1, data2, data3], label=['组1', '组2', '组3'])
#     plt.title('箱线图')
#     plt.ylabel('数值')
    
#     plt.tight_layout()
#     plt.show()
    
#     print("图表已显示！")

# # 运行示例
# matplotlib_basic_charts()

# 高级可视化示例

# 高级可视化示例
import matplotlib.pyplot as plt
import numpy as np


def advanced_visualization():
    """演示高级可视化技巧"""

    # 设置中文字体（防止中文显示为方框）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    print("=== 高级可视化示例 ===")

    # 创建更复杂的数据
    np.random.seed(42)
    n_points = 200

    # 生成相关数据
    x = np.random.randn(n_points)
    y = 2 * x + np.random.randn(n_points) * 0.5
    colors = np.random.rand(n_points)
    sizes = 1000 * np.random.rand(n_points)

    # ============================================================
    # 修改1：创建 Figure，并建立外层 1×3 GridSpec
    # ============================================================
    fig = plt.figure(figsize=(15, 5))

    outer_gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1, 1, 1]
    )

    # ============================================================
    # 1. 气泡图
    # ============================================================
    ax_bubble = fig.add_subplot(outer_gs[0, 0])

    scatter = ax_bubble.scatter(
        x,
        y,
        c=colors,
        s=sizes,
        alpha=0.6,
        cmap='viridis'
    )

    fig.colorbar(
        scatter,
        ax=ax_bubble,
        label='颜色值'
    )

    ax_bubble.set_title('气泡图')
    ax_bubble.set_xlabel('X')
    ax_bubble.set_ylabel('Y')

    # ============================================================
    # 2. 热力图
    # ============================================================
    ax_heatmap = fig.add_subplot(outer_gs[0, 1])

    data = np.random.randn(10, 10)

    im = ax_heatmap.imshow(
        data,
        cmap='coolwarm',
        aspect='auto'
    )

    fig.colorbar(
        im,
        ax=ax_heatmap,
        label='数值'
    )

    ax_heatmap.set_title('热力图')

    # ============================================================
    # 3. 子图组合
    #
    # 修改2：在外层第3列中创建嵌套的 2×2 GridSpec
    # ============================================================
    inner_gs = outer_gs[0, 2].subgridspec(
        2,
        2,
        hspace=0.5,
        wspace=0.4
    )

    # ------------------------------------------------------------
    # 3.1 极坐标图
    # ------------------------------------------------------------
    ax1 = fig.add_subplot(
        inner_gs[0, 0],
        projection='polar'
    )

    theta = np.linspace(
        0,
        2 * np.pi,
        100
    )

    r = np.sin(
        3 * theta
    )

    ax1.plot(
        theta,
        r
    )

    ax1.set_title(
        '极坐标图'
    )

    # ------------------------------------------------------------
    # 3.2 柱状图
    # ------------------------------------------------------------
    ax2 = fig.add_subplot(
        inner_gs[0, 1]
    )

    categories = [
        'A',
        'B',
        'C',
        'D'
    ]

    values = [
        15,
        30,
        45,
        10
    ]

    ax2.bar(
        categories,
        values
    )

    ax2.set_title(
        '柱状图'
    )

    # ------------------------------------------------------------
    # 3.3 组合折线图
    #
    # 修改3：使用 inner_gs[1, :]
    # 让该图占据第二行的两列
    # ------------------------------------------------------------
    ax3 = fig.add_subplot(
        inner_gs[1, :]
    )

    x_line = np.linspace(
        0,
        10,
        100
    )

    y_line1 = np.sin(
        x_line
    )

    y_line2 = np.cos(
        x_line
    )

    ax3.plot(
        x_line,
        y_line1,
        label='sin'
    )

    ax3.plot(
        x_line,
        y_line2,
        label='cos'
    )

    ax3.set_title(
        '组合图'
    )

    ax3.legend()

    # ============================================================
    # 自动调整布局
    # ============================================================
    plt.tight_layout()

    plt.show()

    print("高级图表已显示！")


# 运行示例
advanced_visualization()

