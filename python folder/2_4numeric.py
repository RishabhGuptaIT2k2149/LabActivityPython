# 2_4 How to standardize numeric fields using python.
import pandas as pd

data = {'Feature1': [456, 322, 300, 60, 67],
        'Feature2': [0.5, 2.0, 1.5, 3.0, 2.5]}
df = pd.DataFrame(data)
print("Original dataset:")
print(df)

df_scaled = (df - df.mean()) / df.std()
print("\nStandardized dataset:")
print(df_scaled)
