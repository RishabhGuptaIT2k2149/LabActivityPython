# 2_5 How to identify outliers in python.
import pandas as pd
from scipy import stats

data = {'Values': [10, 20, 30, 40, 50, 200]}
df = pd.DataFrame(data)
z_scores = stats.zscore(df['Values'])
threshold = 3
outliers = abs(z_scores) > threshold

print("Outliers:")
print(df[outliers])
