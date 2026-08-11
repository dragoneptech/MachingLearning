# # 数据准备示例
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split

# class DataPreparer:
#     def __init__(self, data):
#         self.data = data.copy()  # 在 pandas 中，DataFrame.copy() 默认是深拷贝（即参数 deep=True）
#         self.processed_data = None
    
#     def clean_data(self):
#         """数据清洗"""
#         print("开始数据清洗...")
        
#         # 1. 处理缺失值
#         print(f"处理前缺失值数量：{self.data.isnull().sum().sum()}")
        
#         # 数值列用均值填充
#         numeric_columns = self.data.select_dtypes(include=[np.number]).columns
#         for col in numeric_columns:
#             if self.data[col].isnull().sum() > 0:
#                 # self.data[col].fillna(self.data[col].mean(), inplace=True)
#                 self.data[col] = self.data[col].fillna(self.data[col].mean())
        
#         # 类别列用众数填充
#         categorical_columns = self.data.select_dtypes(include=['str']).columns
#         for col in categorical_columns:
#             if self.data[col].isnull().sum() > 0:
#                 mode_val = self.data[col].mode()[0]
#                 # self.data[col].fillna(mode_val, inplace=True)
#                 self.data[col] = self.data[col].fillna(mode_val)
        
#         print(f"处理后缺失值数量：{self.data.isnull().sum().sum()}")
        
#         # 2. 处理重复值
#         duplicates_before = self.data.duplicated().sum()
#         self.data.drop_duplicates(inplace=True)
#         duplicates_after = self.data.duplicated().sum()
#         print(f"删除重复值：{duplicates_before - duplicates_after} 条")
        
#         # 3. 处理异常值（简单方法：使用 IQR）
#         for col in numeric_columns:
#             Q1 = self.data[col].quantile(0.25)
#             Q3 = self.data[col].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
            
#             outliers = ((self.data[col] < lower_bound) | 
#                        (self.data[col] > upper_bound)).sum()
#             if outliers > 0:
#                 # 用边界值替换异常值
#                 self.data[col] = self.data[col].clip(lower_bound, upper_bound)
#                 print(f"处理 {col} 列的 {outliers} 个异常值")
        
#         return self.data
    
#     def feature_engineering(self):
#         """特征工程"""
#         print("\n开始特征工程...")
        
#         # 1. 创建新特征（示例）
#         if 'price' in self.data.columns and 'rating' in self.data.columns:
#             # 创建性价比特征
#             self.data['price_per_rating'] = self.data['price'] / self.data['rating']
#             print("创建新特征：price_per_rating")
        
#         # 2. 特征选择（简单示例：移除低方差特征）
#         numeric_columns = self.data.select_dtypes(include=[np.number]).columns
#         low_variance_features = []
        
#         for col in numeric_columns:
#             if self.data[col].var() < 0.01:  # 方差阈值
#                 low_variance_features.append(col)
        
#         if low_variance_features:
#             self.data.drop(columns=low_variance_features, inplace=True)
#             print(f"移除低方差特征：{low_variance_features}")
        
#         return self.data
    
#     def transform_data(self):
#         """数据转换"""
#         print("\n开始数据转换...")
        
#         # 1. 编码类别变量
#         categorical_columns = self.data.select_dtypes(include=['str']).columns
#         label_encoders = {}
        
#         for col in categorical_columns:
#             le = LabelEncoder()
#             self.data[col] = le.fit_transform(self.data[col])
#             label_encoders[col] = le
#             print(f"编码类别变量：{col}")
        
#         # 2. 标准化数值变量
#         numeric_columns = self.data.select_dtypes(include=[np.number]).columns
#         scaler = StandardScaler()
        
#         if len(numeric_columns) > 0:
#             self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])
#             print(f"标准化数值变量：{list(numeric_columns)}")
        
#         return self.data, label_encoders, scaler
    
#     def split_data(self, target_column, test_size=0.2, val_size=0.2):
#         """数据划分"""
#         print(f"\n开始数据划分（测试集比例：{test_size}，验证集比例：{val_size}）...")
        
#         X = self.data.drop(columns=[target_column])
#         y = self.data[target_column]
        
#         # 首先分离出测试集
#         X_temp, X_test, y_temp, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=42
#         )
        
#         # 再从剩余数据中分离出验证集
#         val_size_adjusted = val_size / (1 - test_size)
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_temp, y_temp, test_size=val_size_adjusted, random_state=42
#         )
        
#         print(f"训练集大小：{X_train.shape[0]}")
#         print(f"验证集大小：{X_val.shape[0]}")
#         print(f"测试集大小：{X_test.shape[0]}")
        
#         return {
#             'X_train': X_train, 'y_train': y_train,
#             'X_val': X_val, 'y_val': y_val,
#             'X_test': X_test, 'y_test': y_test
#         }
    
#     def prepare_pipeline(self, target_column):
#         """完整的数据准备流水线"""
#         print("=" * 50)
#         print("数据准备流水线")
#         print("=" * 50)
        
#         # 1. 数据清洗
#         self.clean_data()
        
#         # 2. 特征工程
#         self.feature_engineering()
        
#         # 3. 数据转换
#         processed_data, encoders, scaler = self.transform_data()
        
#         # 4. 数据划分
#         splits = self.split_data(target_column)
        
#         self.processed_data = processed_data
#         return splits, encoders, scaler

# # 创建示例数据并演示数据准备
# np.random.seed(42)
# sample_data = pd.DataFrame({
#     'age': np.random.randint(18, 65, 1000),
#     'income': np.random.normal(50000, 15000, 1000),
#     'gender': np.random.choice(['男', '女'], 1000),
#     'city': np.random.choice(['北京', '上海', '广州'], 1000),
#     'target': np.random.choice([0, 1], 1000)
# })

# # 添加一些缺失值和异常值
# sample_data.loc[np.random.choice(1000, 50), 'income'] = np.nan
# sample_data.loc[np.random.choice(1000, 20), 'age'] = np.random.randint(100, 150)

# preparer = DataPreparer(sample_data)
# splits, encoders, scaler = preparer.prepare_pipeline('target')


# data_preparation.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreparer:
    def __init__(self, data):
        # 保留原始数据，后续操作副本
        self.data = data.copy()
        self.processed_data = None

        # 保存数据预处理过程中学到的参数
        self.numeric_columns = []
        self.categorical_columns = []

        self.numeric_fill_values = {}
        self.categorical_fill_values = {}
        self.iqr_bounds = {}
        self.low_variance_features = []

        self.label_encoders = {}
        self.scaler = None


    # ============================================================
    # 1. 基础数据清洗
    # ============================================================
    def clean_data(self, target_column):
        """基础数据清洗"""
        print("开始数据清洗...")

        # --------------------------------------------------------
        # 检查 target 是否存在
        # --------------------------------------------------------
        if target_column not in self.data.columns:
            raise ValueError(f"目标变量 {target_column} 不存在")

        print(f"处理前缺失值数量：" + f"{self.data.isnull().sum().sum()}")

        # --------------------------------------------------------
        # target 缺失不能用均值填充
        # 直接删除 target 缺失的样本
        # --------------------------------------------------------

        target_missing = (self.data[target_column].isnull().sum())
        if target_missing > 0:
            self.data = self.data.dropna(subset=[target_column])
            print(f"删除目标变量缺失样本：" + f"{target_missing} 条")

        # --------------------------------------------------------
        # 删除重复数据
        # --------------------------------------------------------
        duplicates_before = (self.data.duplicated().sum())
        self.data = (self.data.drop_duplicates().copy())
        duplicates_after = (self.data.duplicated().sum())
        print(f"删除重复值：" + f"{duplicates_before - duplicates_after} 条")

        return self.data


    # ============================================================
    # 2. 特征工程
    # ============================================================
    def feature_engineering(self, target_column):
        """特征工程"""
        print("\n开始特征工程...")

        # --------------------------------------------------------
        # 示例：创建 price_per_rating
        # --------------------------------------------------------
        if ('price' in self.data.columns and 'rating' in self.data.columns):
            self.data['price_per_rating'] = (self.data['price'] / self.data['rating'])
            print("创建新特征：price_per_rating")

        # 注意：
        # 这里只做不需要从整体数据学习参数的特征构造
        #
        # 低方差特征筛选放到数据划分之后，
        # 并且只利用训练集决定。

        return self.data


    # ============================================================
    # 3. 数据划分
    # ============================================================
    def split_data(self, target_column, test_size=0.2, val_size=0.2):
        """
        先划分数据，再进行均值填充、编码、
        标准化等需要 fit 的操作
        """
        print(f"\n开始数据划分" + f"（测试集比例：{test_size}，" + f"验证集比例：{val_size}）...")

        # --------------------------------------------------------
        # 特征 X
        # --------------------------------------------------------
        X = self.data.drop(columns=[target_column]).copy()

        # --------------------------------------------------------
        # 目标 y
        #
        # 非常重要：
        # target 保持原始数据，不做 StandardScaler
        # --------------------------------------------------------
        y = self.data[target_column].copy()

        # --------------------------------------------------------
        # 第一次划分：
        # 取出测试集
        # --------------------------------------------------------
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,             
            stratify=y  # 分类任务保持类别比例
        )

        # --------------------------------------------------------
        # 验证集比例调整
        #
        # test_size = 0.2
        # val_size  = 0.2
        #
        # 第一次已经留下80%
        #
        # 因此：
        #
        # 0.2 / 0.8 = 0.25
        #
        # 从剩余80%中取25%
        # 就等于原始数据20%
        # --------------------------------------------------------

        val_size_adjusted = (val_size /(1 - test_size))

        # --------------------------------------------------------
        # 第二次划分：
        # Train / Validation
        # --------------------------------------------------------
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_temp
        )

        print(f"训练集大小：" + f"{X_train.shape[0]}")
        print(f"验证集大小：" +f"{X_val.shape[0]}")
        print(f"测试集大小：" + f"{X_test.shape[0]}")
        print(f"训练集 target 类别：" + f"{sorted(y_train.unique())}")

        return (X_train, X_val, X_test, y_train, y_val, y_test)


    # ============================================================
    # 4. 数据转换
    # ============================================================
    def transform_data(self, X_train, X_val, X_test):
        """
        所有需要学习参数的预处理，
        都只能使用训练集进行 fit
        """
        print("\n开始数据转换...")

        # 创建副本，防止修改传入数据
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        # ========================================================
        # 4.1 找出数值列和类别列
        # ========================================================
        self.numeric_columns = list(X_train.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(X_train.select_dtypes(include=['object', 'string']).columns)
        print(f"数值特征：" + f"{self.numeric_columns}")
        print(f"类别特征：" + f"{self.categorical_columns}")

        # ========================================================
        # 4.2 数值列缺失值处理
        #
        # 只使用 X_train 计算均值
        # ========================================================
        for col in self.numeric_columns:
            # ------------------------------------
            # 用训练集计算均值
            # ------------------------------------
            mean_value = (X_train[col].mean())

            self.numeric_fill_values[col] = mean_value

            # ------------------------------------
            # Train / Val / Test
            # 使用同一个训练集均值
            # ------------------------------------
            X_train[col] = (X_train[col].fillna(mean_value))
            X_val[col] = (X_val[col].fillna(mean_value))
            X_test[col] = (X_test[col].fillna(mean_value))

        # ========================================================
        # 4.3 类别列缺失值处理
        #
        # 只使用训练集计算众数
        # ========================================================
        for col in self.categorical_columns:
            mode_series = X_train[col].mode()

            if len(mode_series) > 0:
                mode_value = (mode_series.iloc[0])
            else:
                mode_value = "未知"

            self.categorical_fill_values[col] = mode_value

            X_train[col] = (X_train[col].fillna(mode_value))
            X_val[col] = (X_val[col].fillna(mode_value))
            X_test[col] = (X_test[col].fillna(mode_value))

        # ========================================================
        # 4.4 IQR 异常值处理
        #
        # IQR边界只能由训练集计算
        # ========================================================
        print("\n处理数值异常值...")
        for col in self.numeric_columns:
            # ------------------------------------
            # 只在训练集计算 Q1 / Q3
            # ------------------------------------
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            self.iqr_bounds[col] = (lower_bound, upper_bound)

            # ------------------------------------
            # 统计训练集异常值
            # ------------------------------------
            outliers = ((X_train[col]<lower_bound)|(X_train[col]>upper_bound)).sum()

            # ------------------------------------
            # Train / Val / Test
            # 全部使用训练集得到的边界
            # ------------------------------------
            X_train[col] = (X_train[col].clip(lower_bound, upper_bound))
            X_val[col] = (X_val[col].clip(lower_bound, upper_bound))
            X_test[col] = (X_test[col].clip(lower_bound, upper_bound))

            if outliers > 0:
                print(f"处理 {col} 列的 " + f"{outliers} 个训练集异常值")

        # ========================================================
        # 4.5 类别变量编码
        #
        # LabelEncoder 只在训练集 fit
        # ========================================================
        print("\n开始类别变量编码...")
        self.label_encoders = {}

        for col in self.categorical_columns:
            le = LabelEncoder()
            # ------------------------------------
            # 训练集学习类别及编码方式
            # ------------------------------------
            X_train[col] = (le.fit_transform(X_train[col].astype(str)))
            self.label_encoders[col] = le

            # ------------------------------------
            # 建立类别 → 数字映射
            #
            # 用于验证集、测试集
            # ------------------------------------

            mapping = {category: index for index, category in enumerate(le.classes_)}

            print(f"类别映射 {col}：" + f"{mapping}")

            # ------------------------------------
            # 验证集
            #
            # 如果出现训练集没有的新类别，
            # 编码为 -1
            # ------------------------------------
            X_val[col] = (X_val[col].astype(str).map(mapping).fillna(-1).astype(int))

            # ------------------------------------
            # 测试集
            # ------------------------------------
            X_test[col] = (X_test[col].astype(str).map(mapping).fillna(-1).astype(int))

            print(f"编码类别变量：{col}")


        # ========================================================
        # 4.6 删除低方差特征
        #
        # 只能根据训练集判断
        # ========================================================
        self.low_variance_features = []
        for col in X_train.columns:
            # 编码以后所有列应该都是数值
            if pd.api.types.is_numeric_dtype(X_train[col]):
                variance = (X_train[col].var())
                if variance < 0.01:
                    self.low_variance_features.append(col)

        if self.low_variance_features:
            X_train = X_train.drop(columns=self.low_variance_features)
            X_val = X_val.drop(columns=self.low_variance_features)
            X_test = X_test.drop(columns=self.low_variance_features)
            print(f"移除低方差特征：" + f"{self.low_variance_features}")

        # ========================================================
        # 4.7 标准化数值变量
        #
        # 非常重要：
        #
        # scaler.fit() 只使用 X_train
        #
        # target 根本没有进入 X，
        # 因此绝对不会被标准化
        # ========================================================

        # 只标准化原始连续数值特征
        scale_columns = [col for col in self.numeric_columns if col in X_train.columns]
        self.scaler = StandardScaler()
        if len(scale_columns) > 0:
            # ------------------------------------
            # fit + transform 训练集
            # ------------------------------------
            X_train[scale_columns] = self.scaler.fit_transform(X_train[scale_columns])

            # ------------------------------------
            # 验证集只 transform
            # ------------------------------------
            X_val[scale_columns] = self.scaler.transform(X_val[scale_columns])

            # ------------------------------------
            # 测试集只 transform
            # ------------------------------------
            X_test[scale_columns] = self.scaler.transform(X_test[scale_columns])

            print(f"标准化数值变量：" + f"{scale_columns}")
           
        return (X_train, X_val,X_test, self.label_encoders, self.scaler)


    # ============================================================
    # 5. 完整数据准备流水线
    # ============================================================
    def prepare_pipeline(self, target_column, test_size=0.2, val_size=0.2):
        """完整的数据准备流水线"""
        print("=" * 50)
        print("数据准备流水线")
        print("=" * 50)

        # --------------------------------------------------------
        # 1. 基础清洗
        # --------------------------------------------------------
        self.clean_data(target_column)

        # --------------------------------------------------------
        # 2. 特征工程
        # --------------------------------------------------------
        self.feature_engineering(target_column)

        # --------------------------------------------------------
        # 3. 先划分数据
        # --------------------------------------------------------
        (X_train, X_val, X_test, y_train, y_val, y_test) = self.split_data(target_column, test_size, val_size)

        # --------------------------------------------------------
        # 4. 再进行预处理
        # --------------------------------------------------------
        (X_train, X_val, X_test, encoders, scaler) = self.transform_data(X_train, X_val, X_test)

        # --------------------------------------------------------
        # 5. 保存数据划分结果
        # --------------------------------------------------------
        splits = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }

        # --------------------------------------------------------
        # 保存一份处理后的数据
        # --------------------------------------------------------
        train_processed = X_train.copy()
        train_processed[target_column] = y_train

        val_processed = X_val.copy()
        val_processed[target_column] = y_val

        test_processed = X_test.copy()
        test_processed[target_column] = y_test

        self.processed_data = pd.concat([train_processed, val_processed, test_processed], axis=0)

        # --------------------------------------------------------
        # 最后检查 target
        # --------------------------------------------------------
        print("\n数据准备完成")
        print(f"训练集 target 唯一值：" + f"{sorted(y_train.unique())}")
        print(f"验证集 target 唯一值：" + f"{sorted(y_val.unique())}")

        print(f"测试集 target 唯一值：" + f"{sorted(y_test.unique())}")

        print(f"训练特征：" + f"{list(X_train.columns)}")

        return (splits, encoders, scaler)


# ================================================================
# 创建示例数据
# ================================================================
np.random.seed(42)
sample_data = pd.DataFrame({
    'age': np.random.randint(18, 65, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'gender': np.random.choice(['男', '女'], 1000),
    'city': np.random.choice(['北京', '上海', '广州'], 1000),
    # ------------------------------------------------------------
    # target 为分类标签
    #
    # 从始至终保持 0 / 1
    # ------------------------------------------------------------
    'target': np.random.choice([0, 1], 1000)
})


# ================================================================
# 人为增加缺失值
# ================================================================
# 随机选择50个不同的样本
missing_indices = np.random.choice(sample_data.index, size=50, replace=False)
sample_data.loc[missing_indices, 'income'] = np.nan

# ================================================================
# 人为增加年龄异常值
# ================================================================
# 随机选择20个不同的样本
outlier_indices = np.random.choice(sample_data.index, size=20, replace=False)
# 每个异常样本产生不同的异常年龄
sample_data.loc[outlier_indices, 'age'] = np.random.randint(100, 150, size=20)

# ================================================================
# 执行数据准备
# ================================================================
preparer = DataPreparer(sample_data)
splits, encoders, scaler = preparer.prepare_pipeline('target')

# ================================================================
# 最终检查
# ================================================================
print("\n" + "=" * 50)
print("最终数据检查")
print("=" * 50)

print("X_train.shape：",splits['X_train'].shape)
print("y_train.shape：", splits['y_train'].shape)
print("X_test.shape：", splits['X_test'].shape)
print("y_test.shape：", splits['y_test'].shape)
print("y_train类别：", sorted(splits['y_train'].unique()))
print("y_train数据类型：", splits['y_train'].dtype)