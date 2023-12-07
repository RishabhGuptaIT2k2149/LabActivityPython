# 3_3 How to construct histograms with overlay using python
import matplotlib.pyplot as plt
import numpy as np

data1 = np.random.randn(1000)
data2 = np.random.randn(1000) + 2

plt.hist(data1, bins=30, alpha=0.5, label='Dataset 1')
plt.hist(data2, bins=30, alpha=0.5, label='Dataset 2')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram with Overlay')

plt.legend()
plt.show()
