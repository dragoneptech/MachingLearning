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











