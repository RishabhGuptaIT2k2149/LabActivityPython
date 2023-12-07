# 11_2 How to perform Poisson regression using python.
import pandas as pd
import statsmodels.api as sm

data = {'y': [2, 0, 3, 4, 1, 6, 5, 2, 3, 4],
        'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'X2': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}

df = pd.DataFrame(data)

X = df[['X1', 'X2']]
y = df['y']

X = sm.add_constant(X)

poisson_model = sm.Poisson(y, X)
poisson_results = poisson_model.fit()

print(poisson_results.summary())
