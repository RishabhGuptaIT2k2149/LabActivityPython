# Logistic Regression modelling
# 11_1 How to perform logistic regression modelling.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

new_data = {'feature_a': [4, 5, 6, 7, 5, 4, 6, 7, 8, 9],
            'feature_b': [2, 3, 4, 5, 3, 2, 4, 5, 6, 7],
            'target_label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]}

df_new = pd.DataFrame(new_data)

X_new = df_new[['feature_a', 'feature_b']]
y_new = df_new['target_label']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

logreg_model_new = LogisticRegression()
logreg_model_new.fit(X_train_new, y_train_new)

y_pred_new = logreg_model_new.predict(X_test_new)

accuracy_new = accuracy_score(y_test_new, y_pred_new)
conf_matrix_new = confusion_matrix(y_test_new, y_pred_new)
class_report_new = classification_report(y_test_new, y_pred_new)

print(f'Accuracy: {accuracy_new}')
print(f'Confusion Matrix:\n{conf_matrix_new}')
print(f'Classification Report:\n{class_report_new}')
