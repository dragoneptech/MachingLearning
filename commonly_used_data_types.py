# 常用数据类型

# 机器学习中最常见的四种数据类型：数值型、文本型、图像型和类别型数据
# numrical data: 数值型数据: 连续型、离散型数据
# text data: 文本型数据: 结构化、非结构化数据
# image data: 图像型数据：灰度、彩色、视频序列
# categorical data: 类别型数据：名义型、有序型数据


# 数值型数据 numerical data: 连续型、离散型数据

# 连续型数值数据(continuous data)

# 连续型数值数据示例
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # -------------------------- 设置中文字体 start --------------------------
# # plt.rcParams 中的 rcParams 是 runtime configuration parameters 的缩写，中文可以理解为：
# # 运行时配置参数（运行时绘图参数配置表）
# # plt.rcParams['font.sans-serif'] = [
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
# # bug？必须设置字体才能显示中英文混排
# plt.rcParams['font.size'] = 15

# # 修复负号显示为方块的问题
# plt.rcParams['axes.unicode_minus'] = False
# # -------------------------- 设置中文字体 end --------------------------

# def continuous_data_example():
#     """连续型数值数据示例"""
    
#     print("=== 连续型数值数据示例 ===")
    
#     # 生成连续型数据
#     np.random.seed(42)
#     n_samples = 1000
    
#     # 身高数据（连续型）
#     heights = np.random.normal(170, 10, n_samples)  # 均值170，标准差10
#     weights = heights * 0.7 + np.random.normal(0, 5, n_samples)  # 体重与身高相关
#     temperatures = np.random.normal(36.5, 0.5, n_samples)  # 体温
    
#     # 创建数据框
#     continuous_data = pd.DataFrame({
#         '身高(cm)': heights,
#         '体重(kg)': weights,
#         '体温(°C)': temperatures,
#         '年龄': np.random.randint(18, 65, n_samples)
#     })
    
#     print("连续型数据示例：")
#     print(continuous_data.head())
#     print(f"\n数据统计信息：")
#     print(continuous_data.describe())
    
#     # 可视化连续型数据分布
#     plt.figure(figsize=(12, 8))
    
#     plt.subplot(2, 2, 1)
#     plt.hist(continuous_data['身高(cm)'], bins=30, alpha=0.7, color='skyblue')
#     plt.title('身高分布')
#     plt.xlabel('身高 (cm)')
#     plt.ylabel('频数')
    
#     plt.subplot(2, 2, 2)
#     plt.hist(continuous_data['体重(kg)'], bins=30, alpha=0.7, color='lightgreen')
#     plt.title('体重分布')
#     plt.xlabel('体重 (kg)')
#     plt.ylabel('频数')
    
#     plt.subplot(2, 2, 3)
#     plt.hist(continuous_data['体温(°C)'], bins=30, alpha=0.7, color='salmon')
#     plt.title('体温分布')
#     plt.xlabel('体温 (°C)')
#     plt.ylabel('频数')
    
#     plt.subplot(2, 2, 4)
#     plt.scatter(continuous_data['身高(cm)'], continuous_data['体重(kg)'], alpha=0.6)
#     plt.title('身高 vs 体重')
#     plt.xlabel('身高 (cm)')
#     plt.ylabel('体重 (kg)')
    
#     plt.tight_layout()
#     plt.show()
    
#     return continuous_data

# # 运行示例
# continuous_df = continuous_data_example()

# 离散型数值数据

# 离散型数值数据示例
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # -------------------------- 设置中文字体 start --------------------------
# # plt.rcParams 中的 rcParams 是 runtime configuration parameters 的缩写，中文可以理解为：
# # 运行时配置参数（运行时绘图参数配置表）
# # plt.rcParams['font.sans-serif'] = [
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
# # bug？必须设置字体才能显示中英文混排
# plt.rcParams['font.size'] = 15

# # 修复负号显示为方块的问题
# plt.rcParams['axes.unicode_minus'] = False
# # -------------------------- 设置中文字体 end --------------------------

# def discrete_data_example():
#     """离散型数值数据示例"""
    
#     print("\n=== 离散型数值数据示例 ===")
    
#     # 生成离散型数据
#     np.random.seed(42)
#     n_samples = 500
    
#     # 离散型数据
#     customer_count = np.random.poisson(10, n_samples)  # 泊松分布：顾客数量
#     product_rating = np.random.randint(1, 6, n_samples)  # 1-5星评分
#     defect_count = np.random.binomial(20, 0.1, n_samples)  # 二项分布：缺陷数量
#     call_duration = np.random.exponential(5, n_samples) * 60  # 指数分布：通话时长（秒）
    
#     # 创建数据框
#     discrete_data = pd.DataFrame({
#         '顾客数量': customer_count,
#         '产品评分': product_rating,
#         '缺陷数量': defect_count,
#         '通话时长(秒)': call_duration.astype(int)
#     })
    
#     print("离散型数据示例：")
#     print(discrete_data.head())
#     print(f"\n数据统计信息：")
#     print(discrete_data.describe())
    
#     # 可视化离散型数据
#     plt.figure(figsize=(12, 8))
    
#     plt.subplot(2, 2, 1)
#     plt.hist(discrete_data['顾客数量'], bins=range(0, max(discrete_data['顾客数量'])+2), 
#              alpha=0.7, color='orange')
#     plt.title('顾客数量分布')
#     plt.xlabel('顾客数量')
#     plt.ylabel('频数')
    
#     plt.subplot(2, 2, 2)
#     value_counts = discrete_data['产品评分'].value_counts().sort_index()
#     plt.bar(value_counts.index, value_counts.values, color='purple', alpha=0.7)
#     plt.title('产品评分分布')
#     plt.xlabel('评分')
#     plt.ylabel('频数')
    
#     plt.subplot(2, 2, 3)
#     plt.hist(discrete_data['缺陷数量'], bins=range(0, max(discrete_data['缺陷数量'])+2), 
#              alpha=0.7, color='red')
#     plt.title('缺陷数量分布')
#     plt.xlabel('缺陷数量')
#     plt.ylabel('频数')
    
#     plt.subplot(2, 2, 4)
#     plt.hist(discrete_data['通话时长(秒)'], bins=30, alpha=0.7, color='brown')
#     plt.title('通话时长分布')
#     plt.xlabel('通话时长 (秒)')
#     plt.ylabel('频数')
    
#     plt.tight_layout()
#     plt.show()
    
#     return discrete_data

# # 运行示例
# discrete_df = discrete_data_example()


# 数值型数据的处理方法

# 数值型数据处理方法
# class NumericDataProcessor:
#     def __init__(self):
#         self.scalers = {}
#         self.transformers = {}
    
#     def detect_outliers(self, data, method='iqr'):
#         """检测异常值"""
#         outliers_info = {}
        
#         for column in data.select_dtypes(include=[np.number]).columns:
#             if method == 'iqr':
#                 Q1 = data[column].quantile(0.25)
#                 Q3 = data[column].quantile(0.75)
#                 IQR = Q3 - Q1
#                 lower_bound = Q1 - 1.5 * IQR
#                 upper_bound = Q3 + 1.5 * IQR
                
#                 outliers = data[(data[column] < lower_bound) | 
#                                (data[column] > upper_bound)]
                
#             elif method == 'zscore':
#                 z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
#                 outliers = data[z_scores > 3]
            
#             outliers_info[column] = {
#                 'count': len(outliers),
#                 'indices': outliers.index.tolist(),
#                 'percentage': (len(outliers) / len(data)) * 100
#             }
        
#         return outliers_info
    
#     def handle_missing_values(self, data, strategy='mean'):
#         """处理缺失值"""
#         processed_data = data.copy()
        
#         for column in processed_data.select_dtypes(include=[np.number]).columns:
#             if processed_data[column].isnull().sum() > 0:
#                 if strategy == 'mean':
#                     processed_data[column].fillna(processed_data[column].mean(), inplace=True)
#                 elif strategy == 'median':
#                     processed_data[column].fillna(processed_data[column].median(), inplace=True)
#                 elif strategy == 'mode':
#                     processed_data[column].fillna(processed_data[column].mode()[0], inplace=True)
#                 elif strategy == 'forward':
#                     # processed_data[column].fillna(method='ffill', inplace=True)
#                     processed_data[column].ffill(inplace=True)
#                 elif strategy == 'backward':
#                     # processed_data[column].fillna(method='bfill', inplace=True)
#                     processed_data[column].bfill(inplace=True)
        
#         return processed_data
    
#     def normalize_data(self, data, method='minmax'):
#         """数据标准化"""
#         from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        
#         processed_data = data.copy()
#         numeric_columns = data.select_dtypes(include=[np.number]).columns
        
#         if method == 'minmax':
#             scaler = MinMaxScaler()
#         elif method == 'standard':
#             scaler = StandardScaler()
#         elif method == 'robust':
#             scaler = RobustScaler()
#         else:
#             raise ValueError("方法必须是 'minmax', 'standard', 或 'robust'")
        
#         processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])
#         self.scalers[method] = scaler
        
#         return processed_data
    
#     def create_features(self, data):
#         """特征工程"""
#         processed_data = data.copy()
#         numeric_columns = data.select_dtypes(include=[np.number]).columns
        
#         # 创建多项式特征
#         if len(numeric_columns) >= 2:
#             col1, col2 = numeric_columns[0], numeric_columns[1]
#             processed_data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
#             processed_data[f'{col1}_div_{col2}'] = data[col1] / (data[col2] + 1e-8)
        
#         # 创建统计特征
#         for column in numeric_columns:
#             processed_data[f'{column}_log'] = np.log1p(data[column])
#             processed_data[f'{column}_sqrt'] = np.sqrt(np.abs(data[column]))
#             processed_data[f'{column}_square'] = data[column] ** 2
        
#         return processed_data

# # 使用示例
# processor = NumericDataProcessor()

# # 检测异常值
# outliers = processor.detect_outliers(continuous_df)
# print("\n异常值检测结果：")
# for column, info in outliers.items():
#     if info['count'] > 0:
#         print(f"{column}: {info['count']} 个异常值 ({info['percentage']:.2f}%)")

# # 数据标准化
# normalized_data = processor.normalize_data(continuous_df, method='standard')
# print("\n标准化后的数据示例：")
# print(normalized_data.head())

# 文本型数据 text data: 结构化、非结构化数据

# 文本型数据的分类
# 1. 结构化文本数据

# 结构化文本数据示例
# import pandas as pd
# import re
# from collections import Counter

# def structured_text_example():
#     """结构化文本数据示例"""
    
#     print("\n=== 结构化文本数据示例 ===")
    
#     # 创建结构化文本数据
#     structured_data = pd.DataFrame({
#         '邮件ID': range(1, 11),
#         '发件人': [
#             'zhangsan@example.com', 'lisi@company.com', 'wangwu@service.com',
#             'zhaoliu@business.com', 'qianqi@personal.com', 'sunba@tech.com',
#             'zhoujiu@edu.com', 'wushi@org.com', 'zhengyi@gov.com',
#             'chener@health.com'
#         ],
#         '主题': [
#             '会议通知：明天下午3点开会',
#             '产品报价：最新价格表',
#             '客户反馈：服务满意度调查',
#             '项目进度：第一阶段完成',
#             '假期安排：国庆节放假通知',
#             '技术更新：系统升级公告',
#             '学术会议：论文征集通知',
#             '培训通知：新员工培训',
#             '政策文件：最新规定',
#             '健康提醒：体检通知'
#         ],
#         '内容长度': [156, 234, 189, 145, 98, 267, 198, 134, 312, 87]
#     })
    
#     print("结构化文本数据示例：")
#     print(structured_data)
    
#     # 文本特征提取
#     print("\n=== 文本特征分析 ===")
    
#     # 邮箱域名分析
#     domains = [email.split('@')[1] for email in structured_data['发件人']]
#     domain_counts = Counter(domains)
#     print(f"邮箱域名分布：{dict(domain_counts)}")
    
#     # 主题关键词分析
#     all_words = []
#     for subject in structured_data['主题']:
#         words = re.findall(r'[\u4e00-\u9fff]+', subject)  # 提取中文词汇
#         all_words.extend(words)
    
#     word_counts = Counter(all_words)
#     print(f"主题词频：{dict(word_counts)}")
    
#     # 内容长度统计
#     print(f"内容长度统计：")
#     print(structured_data['内容长度'].describe())
    
#     return structured_data

# # 运行示例
# structured_text_df = structured_text_example()

# 2. 非结构化文本数据

# 非结构化文本数据示例
# import pandas as pd

# def unstructured_text_example():
#     """非结构化文本数据示例"""
    
#     print("\n=== 非结构化文本数据示例 ===")
    
#     # 创建非结构化文本数据
#     unstructured_texts = [
#         """
#         人工智能技术正在快速发展，深度学习、机器学习、自然语言处理等领域取得了重大突破。
#         这些技术在医疗、金融、教育、交通等多个行业都有广泛应用，为社会发展带来了新的机遇。
#         未来，随着计算能力的提升和算法的改进，人工智能将在更多领域发挥重要作用。
#         """,
#         """
#         今天天气真好，阳光明媚，微风徐徐。我决定去公园散步，享受这美好的时光。
#         公园里有很多花，红的、黄的、紫的，五颜六色，非常美丽。小鸟在树上歌唱，
#         蝴蝶在花丛中飞舞，一切都显得那么和谐自然。
#         """,
#         """
#         股市今天表现强劲，上证指数上涨2.3%，深证成指上涨1.8%。
#         科技股领涨，多只股票涨停。分析师认为，这主要得益于近期出台的利好政策。
#         投资者信心得到提振，市场交投活跃，成交量明显放大。
#         """,
#         """
#         健康生活方式包括合理饮食、适量运动、充足睡眠和良好心态。
#         建议每天摄入蔬菜水果，减少油腻食物；每周至少运动3次，每次30分钟以上；
#         保证7-8小时睡眠；学会调节情绪，保持积极乐观的心态。
#         """
#     ]
    
#     text_categories = ['科技', '生活', '财经', '健康']
    
#     # 创建数据框
#     unstructured_df = pd.DataFrame({
#         '文本': unstructured_texts,
#         '类别': text_categories
#     })
    
#     print("非结构化文本数据示例：")
#     for i, row in unstructured_df.iterrows():
#         print(f"\n类别：{row['类别']}")
#         print(f"文本：{row['文本'][:100]}...")
    
#     return unstructured_df

# # 运行示例
# unstructured_text_df = unstructured_text_example()

# # 文本数据处理方法
# import jieba, re
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.preprocessing import LabelEncoder

# class TextDataProcessor:
#     def __init__(self):
#         self.vectorizers = {}
#         self.label_encoders = {}
    
#     def clean_text(self, text):
#         """文本清洗"""
#         # 移除特殊字符和数字
#         text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)
#         # 移除多余空格
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text
    
#     def tokenize_chinese(self, text):
#         """中文分词"""
#         words = jieba.lcut(text)
#         # 移除停用词（简化版）
#         stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
#         words = [word for word in words if word not in stop_words and len(word) > 1]
#         return words
    
#     def extract_features(self, texts, method='tfidf'):
#         """特征提取"""
#         # 文本预处理
#         cleaned_texts = [self.clean_text(text) for text in texts]

#         def chinese_tokenizer(text):
#             return self.tokenize_chinese(text)
        
#         if method == 'tfidf':         
#             vectorizer = TfidfVectorizer(max_features=1000, tokenizer=chinese_tokenizer, token_pattern=None)
#         elif method == 'count':
#             vectorizer = CountVectorizer(max_features=1000, tokenizer=chinese_tokenizer, token_pattern=None)
#         else:
#             raise ValueError("方法必须是 'tfidf' 或 'count'")
        
#         features = vectorizer.fit_transform(cleaned_texts)
#         self.vectorizers[method] = vectorizer
        
#         return features.toarray(), vectorizer.get_feature_names_out()
    
#     def analyze_text_statistics(self, texts):
#         """文本统计分析"""
#         stats = []
        
#         for text in texts:
#             cleaned_text = self.clean_text(text)
#             words = self.tokenize_chinese(cleaned_text)
            
#             stats.append({
#                 '字符数': len(text),
#                 '清洗后字符数': len(cleaned_text),
#                 '词数': len(words),
#                 '平均词长': np.mean([len(word) for word in words]) if words else 0,
#                 '唯一词数': len(set(words))
#             })
        
#         return pd.DataFrame(stats)
    
#     def encode_labels(self, labels):
#         """标签编码"""
#         encoder = LabelEncoder()
#         encoded_labels = encoder.fit_transform(labels)
#         self.label_encoders['default'] = encoder
#         return encoded_labels, encoder.classes_

# # 使用示例
# text_processor = TextDataProcessor()

# # 文本统计分析
# text_stats = text_processor.analyze_text_statistics(unstructured_text_df['文本'])
# print("\n文本统计分析：")
# print(text_stats)

# # 特征提取
# features, feature_names = text_processor.extract_features(
#     unstructured_text_df['文本'], method='tfidf'
# )
# print(f"\n特征矩阵形状：{features.shape}")
# print(f"前10个特征：{feature_names[:10]}")

# # 标签编码
# encoded_labels, label_classes = text_processor.encode_labels(
#     unstructured_text_df['类别']
# )
# print(f"\n编码后的标签：{encoded_labels}")
# print(f"标签类别：{label_classes}")


# 图像型数据

# 图像型数据的分类
# 1. 灰度图像


# 灰度图像示例
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def grayscale_image_example():
    """灰度图像示例"""
    
    print("\n=== 灰度图像示例 ===")
    
    # 创建简单的灰度图像
    # 创建一个 100x100 的灰度图像
    height, width = 100, 100
    
    # 创建渐变图像
    gradient = np.zeros((height, width))
    for i in range(height):
        gradient[i, :] = i  # 垂直渐变
    
    # 创建棋盘图案
    checkerboard = np.zeros((height, width))
    for i in range(0, height, 10):
        for j in range(0, width, 10):
            if (i // 10 + j // 10) % 2 == 0:
                checkerboard[i:i+10, j:j+10] = 255
    
    # 创建圆形图案
    circle = np.zeros((height, width))
    center_x, center_y = width // 2, height // 2
    radius = 30
    for i in range(height):
        for j in range(width):
            if (i - center_y) ** 2 + (j - center_x) ** 2 <= radius ** 2:
                circle[i, j] = 255
    
    # 显示图像
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(gradient, cmap='gray')
    plt.title('渐变图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(checkerboard, cmap='gray')
    plt.title('棋盘图案')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(circle, cmap='gray')
    plt.title('圆形图案')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 图像数据信息
    print(f"渐变图像形状：{gradient.shape}")
    print(f"数据类型：{gradient.dtype}")
    print(f"像素值范围：{gradient.min()} - {gradient.max()}")
    
    return gradient, checkerboard, circle

# 运行示例
gradient_img, checkerboard_img, circle_img = grayscale_image_example()


# 2. 彩色图像

# 彩色图像示例
def color_image_example():
    """彩色图像示例"""
    
    print("\n=== 彩色图像示例 ===")
    
    height, width = 100, 100
    
    # 创建 RGB 彩色图像
    # 红色渐变
    red_gradient = np.zeros((height, width, 3), dtype=np.uint8)
    red_gradient[:, :, 0] = np.linspace(0, 255, width)  # 红色通道渐变
    
    # 绿色渐变
    green_gradient = np.zeros((height, width, 3), dtype=np.uint8)
    green_gradient[:, :, 1] = np.linspace(0, 255, width)  # 绿色通道渐变
    
    # 蓝色渐变
    blue_gradient = np.zeros((height, width, 3), dtype=np.uint8)
    blue_gradient[:, :, 2] = np.linspace(0, 255, width)  # 蓝色通道渐变
    
    # 彩虹图案
    rainbow = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        hue = i / width
        # 简化的 HSV 到 RGB 转换
        if hue < 1/3:
            rainbow[:, i] = [255 * (1 - 3*hue), 255 * 3*hue, 0]
        elif hue < 2/3:
            rainbow[:, i] = [0, 255 * (2 - 3*hue), 255 * (3*hue - 1)]
        else:
            rainbow[:, i] = [255 * (3*hue - 2), 0, 255 * (3 - 3*hue)]
    
    # 显示图像
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(red_gradient)
    plt.title('红色渐变')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(green_gradient)
    plt.title('绿色渐变')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(blue_gradient)
    plt.title('蓝色渐变')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(rainbow)
    plt.title('彩虹图案')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 图像通道信息
    print(f"彩色图像形状：{rainbow.shape}")
    print(f"数据类型：{rainbow.dtype}")
    print(f"像素值范围：{rainbow.min()} - {rainbow.max()}")
    
    return red_gradient, green_gradient, blue_gradient, rainbow

# 运行示例
red_img, green_img, blue_img, rainbow_img = color_image_example()

















