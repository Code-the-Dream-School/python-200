### CTD Python 200  
**Decision Trees and Ensemble Learning (Random Forests)**

## Why trees?

Imagine being asked to identify a flower. You might think:

> “If the petals are tiny, it’s probably setosa.  
> If not, check petal width… big petals? Probably virginica.”

That thought process — breaking a decision into a sequence of **yes/no questions** — is exactly how a **Decision Tree** works.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/3cd4e0d6-8da7-4dc3-b05b-f1379fae0f4c" />

**Image credit: [decision-tree-geeks documentation](https://www.geeksforgeeks.org/machine-learning/decision-tree/)**

Deep learning models are often described as “black boxes.”  
Decision Trees are the opposite: **transparent and interpretable**.  
You can literally trace any prediction step-by-step.

However, a single decision tree can become too confident in the training data.  
This leads to **overfitting**: great performance on seen examples, worse on new data.

Random Forests solve this problem by combining many trees.

---

## How Decision Trees Work

1. Start with a root node using one feature to split data
2. Ask a “yes/no” question that reduces label mixing
3. Continue splitting until no more useful splits exist
4. Reach a leaf node → make a prediction

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/646615fd-2ced-489c-8aa6-790df6802540" />

---

## What you’ll learn today

- Train and interpret a **Decision Tree Classifier**
- Measure node uncertainty using **Gini Impurity**
- Why trees tend to **overfit**
- Use a **Random Forest** to improve accuracy
- Evaluate all models using metrics from KNN lesson
- Compare performance on **Iris** and **Digits**

---

## Setup

```python
!pip install seaborn -q

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams["figure.dpi"] = 120
````

---

## Load datasets

```python
iris = load_iris(as_frame=True)
X_iris, y_iris = iris.data, iris.target

digits = load_digits()
X_digits, y_digits = digits.data, digits.target

print("Iris shape:", X_iris.shape)
print("Digits shape:", X_digits.shape)
```

---

## Split data

```python
def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_i, X_test_i, y_train_i, y_test_i = split(X_iris, y_iris)
X_train_d, X_test_d, y_train_d, y_test_d = split(X_digits, y_digits)
```

---

## Part A — Decision Tree Classifier

### Train the tree

```python
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_i, y_train_i)

preds_tree = tree_clf.predict(X_test_i)
print("Decision Tree Accuracy (Iris):", accuracy_score(y_test_i, preds_tree))
print(classification_report(y_test_i, preds_tree))
```

### Visualizing decisions

```python
plt.figure(figsize=(18, 12))
plot_tree(tree_clf, filled=True,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          rounded=True, fontsize=9)
plt.title("Decision Tree - Iris Dataset")
plt.show()
```

<img width="1415" height="966" alt="awe" src="https://github.com/user-attachments/assets/a3faa07d-73ae-488d-ac82-99b6040e6baf" />

**Decision Trees measure impurity** → less mixed = more confident.

---

## The Overfitting Problem

```python
preds_digits_tree = tree_clf.fit(X_train_d, y_train_d).predict(X_test_d)
print("Decision Tree Accuracy (Digits):", accuracy_score(y_test_d, preds_digits_tree))
```

Decision Tree performance typically drops on complex datasets like Digits.

---

## Part B — Random Forest (Ensemble Method)

Random Forest = Many trees voting → **reduced overfitting**

<img width="712" height="376" alt="Screenshot 2025-11-20 at 2 08 19 PM" src="https://github.com/user-attachments/assets/7311ca18-2d66-48b1-8d11-4b073b09a975" />

```python
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_i, y_train_i)
preds_rf = rf_clf.predict(X_test_i)

print("Random Forest Accuracy (Iris):", accuracy_score(y_test_i, preds_rf))
print(classification_report(y_test_i, preds_rf))
```

Test on Digits:

```python
preds_rf_digits = rf_clf.fit(X_train_d, y_train_d).predict(X_test_d)
print("Random Forest Accuracy (Digits):", accuracy_score(y_test_d, preds_rf_digits))
```

---

## Confusion Matrices (Iris + Digits)

```python
def plot_confusions(model, model_name):
    # Iris
    model.fit(X_train_i, y_train_i)
    preds_i = model.predict(X_test_i)
    cm_i = confusion_matrix(y_test_i, preds_i)
    disp_i = ConfusionMatrixDisplay(cm_i, display_labels=iris.target_names)

    # Digits
    model.fit(X_train_d, y_train_d)
    preds_d = model.predict(X_test_d)
    cm_d = confusion_matrix(y_test_d, preds_d)
    disp_d = ConfusionMatrixDisplay(cm_d)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    disp_i.plot(ax=axes[0], colorbar=False)
    axes[0].set_title(f"{model_name} - Iris")

    disp_d.plot(ax=axes[1], colorbar=False)
    axes[1].set_title(f"{model_name} - Digits")

    plt.tight_layout()
    plt.show()

plot_confusions(DecisionTreeClassifier(random_state=42), "Decision Tree")
plot_confusions(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
```

---

## Accuracy Comparison (KNN vs Tree vs Forest)

```python
models = [
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
]

def evaluate(models, X_train, X_test, y_train, y_test):
    scores = []
    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores.append({"model": name, "accuracy": accuracy_score(y_test, preds)})
    return pd.DataFrame(scores)

df_iris = evaluate(models, X_train_i, X_test_i, y_train_i, y_test_i)
df_digits = evaluate(models, X_train_d, X_test_d, y_train_d, y_test_d)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

sns.barplot(data=df_iris, x="model", y="accuracy", ax=axes[0])
axes[0].set_title("Accuracy Comparison - Iris")

sns.barplot(data=df_digits, x="model", y="accuracy", ax=axes[1])
axes[1].set_title("Accuracy Comparison - Digits")

plt.tight_layout()
plt.show()
```

---

## Feature Importance (Random Forest)

```python
rf_clf.fit(X_train_i, y_train_i)
feat_df = pd.DataFrame({
    "feature": X_iris.columns,
    "importance": rf_clf.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df, x="importance", y="feature")
plt.title("Feature Importance — Random Forest (Iris)")
plt.tight_layout()
plt.show()
```

Petal length & width are most informative in Iris.

---

## Key takeaways

* **Decision Trees** are interpretable but can **overfit**
* **Random Forests** combine many trees → **better generalization**
* Model evaluation should include:

  * Accuracy
  * Classification reports
  * Confusion matrices
* Random Forest especially improves performance on complex data like Digits

---

## Explore More

Optional resources:

* [https://www.ibm.com/think/topics/decision-trees](https://www.ibm.com/think/topics/decision-trees)
* [https://www.youtube.com/watch?v=JcI5E2Ng6r4](https://www.youtube.com/watch?v=JcI5E2Ng6r4)
* [https://www.youtube.com/watch?v=gkXX4h3qYm4](https://www.youtube.com/watch?v=gkXX4h3qYm4)
* [https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html)

---

## Next steps

* Hyperparameter tuning (`max_depth`, `min_samples_split`, `n_estimators`)
* Preventing overfitting with **cross-validation**
* Confusion matrices: error trade-offs
