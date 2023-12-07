# 3_2 How to construct contingency tables using python
import pandas as pd

data = {'Category1': ['A', 'A', 'B', 'C', 'B', 'A'],
        'Category2': ['X', 'Y', 'X', 'Z', 'X', 'Y']}
df = pd.DataFrame(data)
unique_values_category1 = df['Category1'].unique()
unique_values_category2 = df['Category2'].unique()

contingency_table = pd.crosstab(df['Category1'], df['Category2'])

print("Unique values in Category1:", unique_values_category1)
print("Unique values in Category2:", unique_values_category2)
print("\nContingency Table:")
print(contingency_table)
