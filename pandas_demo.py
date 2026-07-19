# 示例：使用 Pandas 和 Matplotlib 进行基础数据分析
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------- 设置中文字体 start --------------------------
# plt.rcParams['font.sans-serif'] = [  #设置无衬线字体
#     # Windows 优先
#     'SimHei', 'Microsoft YaHei',
#     # macOS 优先
#     'PingFang SC', 'Heiti TC',
#     # Linux 优先
#     'WenQuanYi Micro Hei', 'DejaVu Sans'
# ]

# 英文字体为新罗马，中文字体为宋体
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
# 衬线字体
plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]
# 无衬线字体，与Latex相关
plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun"]
# for mathtext.fontset
# supported values are ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
# 设置数学公式字体为 custom，设置LaTeX字体为用户自定义
plt.rcParams['mathtext.fontset'] = 'custom'  # 设置数学公式字体为 custom，设置LaTeX字体为用户自定义

plt.rcParams['font.size'] = 15
# bug？必须设置字体才能显示中英文混排

# 修复负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# -------------------------- 设置中文字体 end --------------------------

# 1. 读取数据
data = pd.read_csv('house_prices.csv', encoding='gbk')  # 假设数据文件为 house_prices.csv，编码为 gbk
print("数据前5行：")
print(data.head())

# 2. 查看数据基本信息
print("\n数据信息：")
print(data.info())

# 3. 绘制房屋面积与价格的散点图
plt.figure(figsize=(10, 6))
plt.scatter(data['面积'], data['价格'], alpha=0.5)  # alpha 设置透明度。数据点很多时，半透明更容易观察密集区域
plt.title('房屋面积 vs 价格')
plt.xlabel('面积 (平方米)')
plt.ylabel('价格 (万元)')
plt.grid(True)
plt.show()