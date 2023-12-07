# 2_3 How to re express categorical field values using python
import pandas as pd

data = {'Item': ['Apple', 'Banana', 'Orange', 'Bread', 'Milk', 'Eggs']}
df = pd.DataFrame(data)

print("Original dataset:")
print(df)
item_mapping = {
    'Apple': 'Fruit',
    'Banana': 'Fruit',
    'Orange': 'Fruit',
    'Bread': 'Bakery',
    'Milk': 'Dairy',
    'Eggs': 'Dairy'
}

df['Item'] = df['Item'].map(item_mapping)
print("\nRe-expressed dataset:")
print(df)
