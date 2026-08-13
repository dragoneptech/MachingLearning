import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun", "Arial", "SimHei"]

plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 13  # If font size <= 12, the Chinese characters will not show.

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

plt.plot(x, y)
plt.xlabel('X-axis / X 轴')
plt.ylabel('Y-axis / Y 轴')
plt.title('Font Style Example / 字体样式示例')
plt.show()

# -------------------------- 设置中文字体 start --------------------------
# plt.rcParams 中的 rcParams 是 runtime configuration parameters 的缩写，中文可以理解为：
# 运行时配置参数（运行时绘图参数配置表）
# plt.rcParams['font.sans-serif'] = [
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
# bug？必须设置字体才能显示中英文混排
plt.rcParams['font.size'] = 15

# 修复负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# -------------------------- 设置中文字体 end --------------------------