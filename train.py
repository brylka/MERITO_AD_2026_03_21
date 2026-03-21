import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

data = load_digits()
X, y = data.data, data.target

model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
model.fit(X, y)

joblib.dump(model, 'model.joblib')
