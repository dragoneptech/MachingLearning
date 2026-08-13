# Python 入门机器学习

# 安装 Python 和必要的库
# 方法一：官方安装器
# 方法二：Anaconda 发行版
# 方法三：uv 管理 python 包

# 选择方法三
# 安装 uv 虚拟环境
# uv --version

# 为什么需要虚拟环境？
# 依赖隔离：不同项目使用不同版本的库
# 环境复现：方便在其他机器上重建相同环境
# 权限管理：避免污染系统 Python 环境
# 项目清理：删除项目时一并删除相关环境

# 使用 conda 管理环境
# 创建环境
# uv venv .venv --python 3.12 --seed
# 激活环境
# .venv\Scripts\activate (windows)
# source .venv/bin/activate (linux/mac)
# 安装包
# uv pip install numpy pandas scikit-learn
# 列出环境
# uv pip list
# 删除环境
# rm -rf .venv

# 1. 数据导入和探索
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------- 设置中文字体 start --------------------------
# # plt.rcParams['font.sans-serif'] = [  #设置无衬线字体
# #     # Windows 优先
# #     'SimHei', 'Microsoft YaHei',
# #     # macOS 优先
# #     'PingFang SC', 'Heiti TC',
# #     # Linux 优先
# #     'WenQuanYi Micro Hei', 'DejaVu Sans'
# # ]

# # 英文字体为新罗马，中文字体为宋体
# plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
# # 衬线字体
# plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]
# # 无衬线字体，与Latex相关
# plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun"]
# # for mathtext.fontset
# # supported values are ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
# # 设置数学公式字体为 custom，设置LaTeX字体为用户自定义
# plt.rcParams['mathtext.fontset'] = 'custom'  # 设置数学公式字体为 custom，设置LaTeX字体为用户自定义

# plt.rcParams['font.size'] = 15
# # bug？必须设置字体才能显示中英文混排

# # 修复负号显示为方块的问题
# plt.rcParams['axes.unicode_minus'] = False
# # -------------------------- 设置中文字体 end --------------------------

# 创建示例数据
# data = {
#     '姓名': ['张三', '李四', '王五', '赵六'],
#     '年龄': [25, 30, 35, 28],
#     '城市': ['北京', '上海', '广州', '深圳'],
#     '薪资': [15000, 20000, 18000, 22000]
# }
# df = pd.DataFrame(data)
# print("数据预览：")
# print(df.head())

# 2. 数据可视化
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.bar(df['姓名'], df['年龄'])
# plt.title('年龄分布')
# plt.xlabel('姓名')
# plt.ylabel('年龄')
# plt.subplot(1, 2, 2)
# plt.bar(df['姓名'], df['薪资'])
# plt.title('薪资分布')
# plt.xlabel('姓名')
# plt.ylabel('薪资')
# plt.tight_layout()
# plt.show()

# # 3. 简单统计分析
# print("\n基本统计信息：")
# print(df.describe())
# print("\n城市分布：")
# print(df['城市'].value_counts())

# 一个简单的机器学习例子：使用 Scikit-learn 做分类

# 步骤 1：导入库
# 导入需要的 Python 库：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 步骤 2：加载数据
# 加载鸢尾花数据集：
# 加载鸢尾花数据集
iris = load_iris()

# 将数据转化为 pandas DataFrame
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # 特征数据
y = pd.Series(iris.target)  # 标签数据

# 显示前五行数据
print(X.head())

# 步骤 3：数据集划分
# 将数据集划分为训练集和测试集，通常使用 70% 训练集和 30% 测试集的比例：
# 划分训练集和测试集（80% 训练集，20% 测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤 4：特征缩放（标准化）
# 许多机器学习算法都依赖于特征的尺度，特别是像 K 最近邻算法
# 为了确保每个特征的均值为 0，标准差为 1，我们使用标准化来处理数据：
# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 步骤 5：选择模型并训练
# 在这个例子中，我们选择 K-Nearest Neighbors（KNN） 算法来进行分类：
# 创建 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

# 步骤 6：评估模型
# 训练完成后，我们使用测试集评估模型的准确性：
# 预测测试集
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy:.2f}')

# 步骤 7：可视化结果（可选）
# 可视化 - 这里只是一个简单示例，具体可根据实际情况选择绘图方式
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o')
plt.title("KNN Classification Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()









