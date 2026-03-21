import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Wczytanie danych
data = load_digits()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modele do porównania
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                                 random_state=42, n_jobs=-1, verbosity=0),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1,
                                    random_state=42, verbose=-1)
}

text = "PORÓWNANIE MODELI\n"
text += "=" * 70 + "\n"
text += f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12} {'CV Std':<10} {'Czas':<8}\n"
text += "-" * 70 + "\n"

for name, model in models.items():
    # Pomiar czasu
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Metryki
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X, y, cv=10)

    text += f"{name:<20} {train_acc:<12.4f} {test_acc:<12.4f} {cv_scores.mean():<12.4f} {cv_scores.std():<10.4f} {train_time:<8.4f}\n"

print(text)