# 10_2 Demonstrate how you will apply principal components analysis using python.
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {'feature1': [2, 4, 1, 3, 5, 2, 6, 1, 7, 2],
        'feature2': [1, 3, 2, 4, 2, 1, 3, 4, 5, 6],
        'feature3': [3, 2, 5, 1, 4, 6, 2, 4, 1, 3]}

df = pd.DataFrame(data)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])

plt.scatter(df_pca['Principal Component 1'], df_pca['Principal Component 2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Result')
plt.show()
