from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

# Ewaluacja
train_acc = single_tree.score(X_train, y_train)
test_acc = single_tree.score(X_test, y_test)

print("POJEDYNCZE DRZEWO DECYZYJNE")
print("-" * 40)
print(f"Dokładność na zbiorze treningowym: {train_acc:.4f}")
print(f"Dokładność na zbiorze testowym:    {test_acc:.4f}")