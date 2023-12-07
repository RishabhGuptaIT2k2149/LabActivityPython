# 4_5 How to build random forests using python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

accuracy = accuracy_score(y_test, clf.predict(X_test))
print(f"Accuracy: {accuracy * 100:.2f}%")
