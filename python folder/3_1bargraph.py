# 3_1 How to construct a bar graph with overlay in python
import matplotlib.pyplot as plt
import numpy as np

categories = ['Category 1', 'Category 2', 'Category 3']
values1 = [10, 15, 20]
values2 = [5, 12, 18]

bar_width = 0.35
bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width

plt.bar(bar_positions1, values1, width=bar_width, label='Set 1', color='blue')
plt.bar(bar_positions2, values2, width=bar_width, label='Set 2', color='orange')

plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Grouped Bar Chart')

plt.xticks(bar_positions1 + bar_width / 2, categories)
plt.legend()

plt.show()
