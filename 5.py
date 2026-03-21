import lightgbm as lgb
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42, verbose=-1)
model.fit(X_train, y_train)

model_train_acc = model.score(X_train, y_train)
model_test_acc = model.score(X_test, y_test)
model_cv_scores = cross_val_score(model, X, y, cv=10)

print("LightGBM:")
print("-" * 40)
print(f"Dokładność na zbiorze treningowym: {model_train_acc:.4f}")
print(f"Dokładność na zbiorze testowym:    {model_test_acc:.4f}")


print(f"Walidacja cv:")
print(f"Średnia dokładność: {model_cv_scores.mean():.4f}")
print(f"Odchylenie std:     {model_cv_scores.std():.4f}")