#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'  # 使用 "Times New Roman"
plt.rcParams['font.size'] = 40

# 数据参数
num_samples = 100000  # 生成的样本组数
sample_size = 4  # 每组样本大小

# 生成数据并排序
data = np.sort(np.random.normal(0, 0.25, (num_samples, sample_size)), axis=1)

plt.figure(figsize=(22, 4))  # 修改图的大小，以容纳三个直方图

# 计算每组的最大值和次大值的和，设置不同的颜色和透明度，并绘制直方图
diffmax1 = data[:, -1] + data[:, -2] - data[:, -3]
plt.hist(diffmax1, bins=50, density=True, color='orange', alpha=0.5, label='Solution 1')

# 计算每组的次大值和第三大值的和，设置不同的颜色和透明度，并绘制直方图
diffmax2 = data[:, -2] + data[:, -3] - data[:, -4]
plt.hist(diffmax2, bins=50, density=True, color='blue', alpha=0.5, label='Solution 2')

# 计算每组的最大值和最小值之间的差值，设置不同的颜色和透明度，并绘制直方图
diffmax3 = data[:, -1] - data[:, -4]
plt.hist(diffmax3, bins=50, density=True, color='green', alpha=0.5, label='Solution 2')

# 添加x轴标签、y轴标签和图例
plt.xlabel('Maximum of Difference')
plt.ylabel('Probability Density')
plt.legend(loc='upper right')  # 添加图例

# 保存图形
plt.savefig('monte carlo.svg', format='svg')

# 显示图形
plt.show()
