# 12_2 How to apply the confidence difference criterion using python.
import numpy as np
import scipy.stats as stats

data1 = np.random.normal(0, 1, 150)
data2 = np.random.normal(1, 1.2, 150)

confidence_interval = stats.t.interval(0.95, len(data1) - 1, loc=np.mean(data1) - np.mean(data2),
                                       scale=stats.sem(data1 - data2))

if confidence_interval[0] <= 0 <= confidence_interval[1]:
    print("The difference is not statistically significant.")
else:
    print("The difference is statistically significant.")
