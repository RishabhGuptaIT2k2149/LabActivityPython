# 2_2 How to change misleading field values using python
import pandas as pd
data = {'Name': ['Rishabh', 'Abc', 'Akash', 'Hrushika', 'Aparna'],
        'Age': [25, -1, 30, -40, 20],
        'Gender': ['M', 'M', 'M', 'F', 'F']}

df = pd.DataFrame(data)
df_original = df.copy()
df['Age'] = df['Age'].apply(lambda x: x if x > 0 else pd.NA)
print("Original DataFrame:")
print(df_original)
print("\nModified DataFrame:")
print(df)
