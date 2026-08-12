# 机器学习基础术语

# 机器学习: Maching Learning
# 数据: Data
# 特征: Feature
# 标签: Label
# 模型: Model
# 训练: Training
# 推理: Inference

# 数据的类型
# 1. 结构化数据: Structured Data

# import pandas as pd
# import json
# students_data = {
#     '姓名': ['张三', '李四', '王五'],
#     '年龄': [18, 19, 20],
#     '成绩': [85, 92, 78],
#     '班级': ['一班', '二班', '一班']
# }
# df = pd.DataFrame(students_data)
# print(df)
# print(json.dumps(students_data, indent=4, ensure_ascii=False))
# json.dumps() 将 Python 对象转换为 JSON 字符串
# indent 参数用于设置缩进
# ensure_ascii=False 用于确保中文字符不被转义

# 2. 非结构化数据: Unstructured Data

# 非结构化数据示例：文本和图像
# text_data = "这个产品质量很好，我很满意！"
# image_data = 一张产品的照片
# audio_data = 顾客的语音评价

# 数据质量的重要性
# 垃圾进，垃圾出（Garbage In, Garbage Out）是机器学习的重要原则
# 数据质量直接决定模型效果

# 数据质量问题示例
# import numpy as np
# import pandas as pd
# 创建包含各种问题的数据
# problematic_data = {
#     '价格': [100, 200, None, 300, -50],  # 缺失值和异常值
#     '评分': [4.5, '好', 3.8, 4.2, 5.0],  # 数据类型不一致
#     '销量': [1000, 1200, 800, 1500, '很多']  # 文本和数字混合
# }
# df = pd.DataFrame(problematic_data)
# print("有问题的数据：")
# print(df)
# print("\n数据问题分析：")
# print(f"缺失值数量：{df.isnull().sum().sum()}")
# print(f"数据类型：\n{df.dtypes}")

# 特征（Feature）
# 什么是特征？
# 特征是数据的"可观察属性"
# 在机器学习中，特征是用来做预测的依据

# 特征选择的重要性：
# 好的特征能让模型事半功倍
# 坏的特征会让模型事倍功半
# 特征工程往往是决定模型效果的关键

# 特征的类型
# 数值特征
# 特点：可以用数字表示，可以进行数学运算

# 数值特征示例
# import pandas as pd
# numerical_features = {
#     '年龄': [25, 30, 35, 40],
#     '收入': [5000, 8000, 12000, 15000],
#     '身高': [165, 170, 175, 180]
# }
# df_numerical = pd.DataFrame(numerical_features)
# print("数值特征示例：")
# print(df_numerical)

# 类别特征
# 特点：表示不同的类别，不能进行数学运算

# 类别特征示例
# import pandas as pd
# categorical_features = {
#     '性别': ['男', '女', '男', '女'],
#     '学历': ['本科', '硕士', '博士', '本科'],
#     '城市': ['北京', '上海', '广州', '深圳']
# }
# df_categorical = pd.DataFrame(categorical_features)
# print("类别特征示例：")
# print(df_categorical)

# 文本特征
# 特点：需要特殊处理才能被模型使用

# 文本特征示例
# import json
# text_features = {
#     '评论': [
#         '这个产品很好用，推荐购买！',
#         '质量一般，不太满意。',
#         '性价比高，值得入手。'
#     ]
# }
# print("文本特征示例：")
# print(json.dumps(text_features, indent=4, ensure_ascii=False))

# 特征工程示例
# 特征工程示例：从原始数据创建有用特征
# import pandas as pd
# import numpy as np
# 原始数据：房屋信息
# house_data = {
#     '面积': [80, 120, 60, 150, 90],
#     '卧室数': [2, 3, 1, 4, 2],
#     '建造年份': [2000, 2010, 1995, 2015, 2005],
#     '价格': [200, 350, 150, 500, 280]
# }
# df = pd.DataFrame(house_data)
# 创建新特征
# df['房龄'] = 2023 - df['建造年份']  # 房屋年龄
# df['每平米价格'] = df['价格'] / df['面积']  # 单价
# df['卧室面积比'] = df['卧室数'] / df['面积'] * 100  # 卧室占比
# print("原始数据 + 新特征：")
# print(df)
# 特征重要性分析
# correlation = df.corr()['价格'].sort_values(ascending=False)
# print("\n特征与价格的相关性：")
# print(correlation)
# print(df.corr())


# 标签（Label）
# 什么是标签？
# 标签是我们想要预测的"答案"，就像考试题的正确答案一样
# 在监督学习中，每个数据样本都有一个对应的标签

# 标签的作用：
# 指导模型学习方向
# 评估模型学习效果
# 定义问题的类型

# 标签的类型
# 分类标签
# 特点：离散的类别值

# 分类标签示例
# classification_labels = {
#     '邮件类型': ['垃圾邮件', '正常邮件', '垃圾邮件', '正常邮件'],
#     '情感倾向': ['正面', '负面', '中性', '正面'],
#     '疾病诊断': ['患病', '健康', '健康', '患病']
# }

# 回归标签
# 特点：连续的数值

# 回归标签示例
# regression_labels = {
#     '房价': [250000, 320000, 180000, 450000],
#     '温度': [25.5, 28.3, 22.1, 30.0],
#     '股票价格': [100.5, 105.2, 98.7, 110.3]
# }

# 标签质量的重要性
# 标签质量问题示例
# import numpy as np
# 模拟图像分类任务中的标签问题
# image_data = ['cat1.jpg', 'dog1.jpg', 'cat2.jpg', 'dog2.jpg']
# problematic_labels = ['猫', '犬', '猫咪', '狗']  # 标签不一致
# 标签标准化
# label_mapping = {
#     '猫': 'cat', '猫咪': 'cat',
#     '犬': 'dog', '狗': 'dog'
# }
# standardized_labels = [label_mapping[label] for label in problematic_labels]
# print("原始标签：", problematic_labels)
# print("标准化标签：", standardized_labels)

# 模型（Model）
# 什么是模型？
# 模型是机器学习算法 (Algorithm) 从数据中学到的"规律"或"模式"，就像学生从课本中学到的知识一样

# 模型的本质：
# 数学函数：输入特征，输出预测
# 参数集合：学到的规律的具体表示
# 决策规则：如何从输入得到输出

# 模型的表示

# 简单线性模型示例
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

# 模拟数据
# X = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 4, 6, 8, 10])
# 线性模型：y = w * x + b
# 学习到的参数：w = 2, b = 0
# w, b = 2, 0
# def linear_model(x):
#     """线性模型函数"""
#     return w * x + b
# 预测
# predictions = linear_model(X)
# 可视化
# plt.scatter(X, y, color='blue', label='真实数据')
# plt.plot(X, predictions, color='red', label='模型预测')
# plt.xlabel('输入 X')
# plt.ylabel('输出 y')
# plt.title('线性模型示例')
# plt.legend()
# plt.grid(True)
# plt.show()
# print(f"模型参数：w = {w}, b = {b}")
# print(f"预测结果：{predictions}")


# 模型的复杂度

# 模型复杂度对比
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# import numpy as np
# from matplotlib import pyplot as plt

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

# 生成非线性数据
# np.random.seed(42)
# X = np.random.rand(20, 1) * 10  # 生成 20 个 0-10 之间的随机数
# y = np.sin(X) + np.random.randn(20, 1) * 0.1

# 简单模型（线性）
# simple_model = LinearRegression()
# simple_model.fit(X, y)

# 复杂模型（高次多项式）
# poly_features = PolynomialFeatures(degree=6)
# X_poly = poly_features.fit_transform(X)
# complex_model = LinearRegression()
# complex_model.fit(X_poly, y)

# 可视化
# X_test = np.linspace(0, 10, 100).reshape(-1, 1)
# X_test_poly = poly_features.transform(X_test)
# plt.scatter(X, y, color='blue', label='训练数据')
# plt.plot(X_test, simple_model.predict(X_test), color='green', label='简单模型')
# plt.plot(X_test, complex_model.predict(X_test_poly), color='red', label='复杂模型')
# plt.xlabel('X 轴')
# plt.ylabel('y 轴')
# plt.title('模型复杂度对比')
# plt.legend()
# plt.grid(True)
# plt.show()


# 训练（Training）
# 什么是训练？
# 训练是模型学习的过程，就像学生上课学习知识一样
# 在训练过程中，模型不断调整参数，使预测结果越来越接近真实标签

# 训练过程示例

# 训练过程示例：简单线性回归
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

# 生成训练数据
# np.random.seed(42)
# X = np.random.rand(50, 1) * 10
# y = 3 * X + 2 + np.random.randn(50, 1) * 2

# 初始化模型参数
# w, b = 0.0, 0.0
# learning_rate = 0.01
# epochs = 100  # 训练轮数

# 记录训练过程
# loss_history = []

# 训练循环
# for epoch in range(epochs):
#     # 前向传播
#     y_pred = w * X + b

#     # 计算损失（均方误差）
#     loss = np.mean((y_pred - y) ** 2)
#     loss_history.append(loss)

#     # 计算梯度
#     dw = np.mean(2 * X * (y_pred - y))
#     db = np.mean(2 * (y_pred - y))

#     # 更新参数
#     w -= learning_rate * dw
#     b -= learning_rate * db

#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

# print(f"After training: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

# 可视化训练过程
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(loss_history)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('训练损失变化')
# plt.grid(True)
# plt.subplot(1, 2, 2)
# plt.scatter(X, y, color='blue', label='训练数据')
# plt.plot(X, w * X + b, color='red', label='训练后的模型')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('训练结果')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# print(f"最终模型参数：w = {w:.4f}, b = {b:.4f}")


# 推理（Inference）
# 什么是推理？
# 推理是使用训练好的模型进行预测的过程，就像学生用学到的知识解答考试题一样

# 推理过程示例
# import numpy as np
# 假设我们已经训练好了一个房价预测模型
# class HousePriceModel:
#     def __init__(self):
#         # 模拟训练好的参数
#         self.feature_weights: dict[str, float] = {
#             '面积': 2.5,
#             '卧室数': 10.0,
#             '房龄': -1.0,
#             '地段评分': 50.0
#         }
#         self.bias: float = 50.0

#     def predict(self, features: dict) -> float:
#         """
#         使用训练好的模型进行房价预测
#         """
#         price: float = self.bias
#         for feature_name, feature_value in features.items():
#             if feature_name in self.feature_weights:
#                 price += self.feature_weights[feature_name] * feature_value
#         return price

# 创建训练好的模型
# model = HousePriceModel()

# 推理：预测新房价
# new_houses = [
#     {'面积': 80, '卧室数': 2, '房龄': 5, '地段评分': 8},
#     {'面积': 120, '卧室数': 3, '房龄': 2, '地段评分': 9},
#     {'面积': 60, '卧室数': 1, '房龄': 10, '地段评分': 6}
# ]

# print("房价预测结果：")
# for i, house in enumerate(new_houses, 1):
#     predicted_price = model.predict(house)
#     print(f"房子{i}：预测价格 {predicted_price:.2f} 万元")

# 批量推理
# def batch_predict(model: HousePriceModel, house_list: list[dict]) -> list[float]:
#     """批量预测"""
#     return [model.predict(house) for house in house_list]

# batch_prices = batch_predict(model, new_houses)
# print(f"\n批量预测结果：{batch_prices}")

# 完整示例：从数据到推理

# 完整的机器学习流程示例
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# 1. 数据准备
# np.random.seed(42)
# n_samples = 200

# 生成特征数据
# area = np.random.normal(100, 30, n_samples)  # 面积
# bedrooms = np.random.randint(1, 5, n_samples)  # 卧室数
# age = np.random.randint(0, 20, n_samples)  # 房龄
# location_score = np.random.randint(1, 10, n_samples)  # 地段评分

# 生成标签（房价）- 基于特征的线性组合加噪声
# price = (area * 2.5 + bedrooms * 20 + age * (-2) + location_score * 15 + 
#          np.random.normal(0, 50, n_samples))

# 创建数据框
# data = pd.DataFrame({
#     '面积': area,
#     '卧室数': bedrooms,
#     '房龄': age,
#     '地段评分': location_score,
#     '价格': price
# })
# print("数据示例：")
# print(data.head())

# 2. 划分训练集和测试集
# features = ['面积', '卧室数', '房龄', '地段评分']
# X = data[features]
# y = data['价格']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(f"\n训练集大小：{X_train.shape[0]}")
# print(f"测试集大小：{X_test.shape[0]}")

# 3. 训练模型
# model = LinearRegression()
# model.fit(X_train, y_train)

# print(f"\n模型参数：")
# for feature, coef in zip(features, model.coef_):
#     print(f"{feature}: {coef:.2f}")

# print(f"截距: {model.intercept_:.2f}")

# 4. 评估模型
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)
# train_mse = mean_squared_error(y_train, y_train_pred)
# test_mse = mean_squared_error(y_test, y_test_pred)
# train_r2 = r2_score(y_train, y_train_pred)
# test_r2 = r2_score(y_test, y_test_pred)
# print(f"\n模型评估：")
# print(f"训练集 MSE: {train_mse:.2f}, R²: {train_r2:.2f}")
# print(f"测试集 MSE: {test_mse:.2f}, R²: {test_r2:.2f}")

# 5. 推理（预测新数据）
# new_houses = pd.DataFrame({
#     '面积': [85, 120, 65],
#     '卧室数': [2, 3, 1],
#     '房龄': [3, 1, 8],
#     '地段评分': [7, 9, 5]
# })
# predictions = model.predict(new_houses)
# print(f"\n新房价预测：")
# for i, price in enumerate(predictions, 1):
#     print(f"房子{i}: {price:.2f} 万元")



# 常见机器学习网络类型
# 传统机器学习
# 决策树: Decision Tree, DT
# 随机森林: Random Forest, RF
# 逻辑回归: Logistic Regression, LR
# 支持向量机: Support Vector Machine, SVM
# 朴素贝叶斯: Naive Bayes, NB
# 极端梯度提升: Extreme Gradient Boosting, XGBoost
# 轻量梯度提升机器: Light Gradient Boosting Machine, LightGBM

# 深度学习: Deep Learning, DL
# 人工神经网络: Artificial Neural Network, ANN
# 卷积神经网络: Convolutional Neural Network, CNN
# 循环神经网络: Recurrent Neural Network, RNN
# 长短期记忆网络: Long Short-Term Memory, LSTM
# 门控循环单元网络: Gated Recurrent Unit, GRU
# 生成对抗网络: Generative Adversarial Network, GAN
# 变换器: Transformer, Transformer
# 自编码器: Autoencoder, AE
# 图神经网络: Graph Neural Network, GNN










