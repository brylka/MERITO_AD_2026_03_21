from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

data = load_digits()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

n_trees_range = [10, 25, 50, 100, 150, 200, 300, 500, 750, 1000]
train_scores = []
test_scores = []
oob_scores = []

# wymień RandomForest na XGBoost
# Dodatkowe: Porównanie RandomForest oraz XGBoost zależne od ilości drzew

for n_trees in n_trees_range:
    random_forest = RandomForestClassifier(
        n_estimators=n_trees,
        oob_score=True,
        random_state=42)
    random_forest.fit(X_train, y_train)

    train_scores.append(random_forest.score(X_train, y_train))
    test_scores.append(random_forest.score(X_test, y_test))
    oob_scores.append(random_forest.oob_score_)

plt.plot(n_trees_range, train_scores, 'b-o', label="treningowa")
plt.plot(n_trees_range, test_scores, 'r-s', label="testowa")
plt.plot(n_trees_range, oob_scores, 'g-^', label="OOB")
plt.xlabel('Ilość drzew')
plt.ylabel('Dokładność')
plt.title('Dokładność a Liczba drzew')
plt.grid(True)
plt.legend()
plt.show()

# dodać cv + wykres


# print(n_trees_range)
# print(train_scores)
# print(test_scores)
# print(oob_scores)