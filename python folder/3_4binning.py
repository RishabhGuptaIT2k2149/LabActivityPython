# 3_4 How to perform binning based on predictive value using python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = pd.DataFrame({
    'feature': np.random.randn(1000),
    'target': np.random.choice([0, 1], size=1000, replace=True)
})
data['quantiles'] = pd.qcut(data['feature'], q=[0, 0.25, 0.5, 0.75, 1.0])
quantile_means = data.groupby('quantiles')['target'].mean().reset_index()

plt.figure(figsize=(8, 6))
plt.bar(quantile_means['quantiles'].astype(str), quantile_means['target'], color='salmon')
plt.title('Binning Based on Predictive Value (Quantiles)')
plt.xlabel('Quantiles')
plt.ylabel('Mean Target Value')
plt.show()
