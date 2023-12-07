# 4_2 How to balance the training data set
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import load_iris
from collections import Counter

data = load_iris()
X, y = data.data, data.target

print("Class distribution before undersampling:", Counter(y))

under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled_under, y_resampled_under = under_sampler.fit_resample(X, y)

print("Class distribution after undersampling:", Counter(y_resampled_under))

over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled_over, y_resampled_over = over_sampler.fit_resample(X, y)

print("Class distribution after oversampling:", Counter(y_resampled_over))
