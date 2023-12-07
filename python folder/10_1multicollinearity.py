# 10_1 Demonstrate how you will identify multicollinearity in python.
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

new_data = {'var_x': [4, 6, 2, 5, 3, 2, 7, 1, 8, 3],
            'var_y': [2, 4, 1, 3, 5, 2, 6, 1, 7, 2],
            'var_z': [1, 3, 2, 4, 2, 1, 3, 4, 5, 6]}

df_new = pd.DataFrame(new_data)

X_new = df_new[['var_x', 'var_y', 'var_z']]

vif_data_new = pd.DataFrame()
vif_data_new["Variable"] = X_new.columns
vif_data_new["VIF"] = [variance_inflation_factor(X_new.values, i) for i in range(X_new.shape[1])]

print(vif_data_new)
