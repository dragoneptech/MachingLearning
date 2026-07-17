# 导入需要的库
import numpy as np  # NumPy: Numerical Python，用于科学计算和数组操作
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns  # seaborn 是基于 matplotlib 的高级绘图库

# 设置图表风格，让图表更好看
sns.set_style("whitegrid")

# -------------------------- 设置中文字体 start --------------------------
# plt.rcParams 中的 rcParams 是 runtime configuration parameters 的缩写，中文可以理解为：
# 运行时配置参数（运行时绘图参数配置表）
plt.rcParams['font.sans-serif'] = [
    # Windows 优先
    'SimHei', 'Microsoft YaHei',
    # macOS 优先
    'PingFang SC', 'Heiti TC',
    # Linux 优先
    'WenQuanYi Micro Hei', 'DejaVu Sans'
]
# 修复负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# -------------------------- 设置中文字体 end --------------------------

# 1. 准备数据
# 假设我们有房屋面积和对应的价格数据
# 房屋面积（平方米）
# sklearn 要求输入特征 X 必须是二维数组，因此我们使用 reshape(-1, 1) 将一维数组转换为二维数组 (此处 1 个特征)
# 样本数量 × 特征数量
house_sizes = np.array([50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1, 1)
# 房屋价格（万元）
# 创建标签数据
house_prices = np.array([150, 180, 210, 240, 270, 300, 330, 360])

# 2. 创建并训练模型
# 创建线性回归模型
model = LinearRegression()
# 用数据训练模型（学习面积和价格之间的关系）
model.fit(house_sizes, house_prices)

# 3. 使用模型进行预测
# 预测 85 平方米的房屋价格
# 注意：predict 方法的输入必须是二维数组，因此我们传入 [[85]] 而不是 [85]
# 样本数量 × 特征数量，此处预测样本数量为 1，特征数量为 1
predicted_price = model.predict([[85]])
print(f"85 平方米的房屋预测价格：{predicted_price[0]:.2f} 万元")

# 4. 可视化结果
plt.scatter(house_sizes, house_prices, color='blue', label='实际数据')
plt.plot(house_sizes, model.predict(house_sizes), color='red', label='预测线')
plt.scatter([85], predicted_price, color='green', s=100, label='预测点')
plt.xlabel('房屋面积（平方米）')
plt.ylabel('房屋价格（万元）')
plt.title('机器学习测试 -- 房屋面积与价格关系')
plt.legend()
plt.grid(True)

print("线性回归模型训练完成，预测结果可视化已显示。")
print(f"模型系数（斜率）：{model.coef_[0]:.2f}")
print(f"模型截距：{model.intercept_:.2f}")

plt.show()
