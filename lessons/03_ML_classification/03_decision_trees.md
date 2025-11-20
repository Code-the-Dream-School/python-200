# Lesson 3  
### CTD Python 200  
**Decision Trees and Ensemble Learning (Random Forests)**

## Why trees?

Imagine being asked to identify a flower. You might think:

> “If the petals are tiny, it’s probably a setosa.  
> If not, check petal width… big petals? Probably virginica.”

That thought process — breaking a decision into simple **yes/no** questions — is exactly what a **Decision Tree** does.

Deep learning models are often described as “black boxes.”  
Decision Trees are the opposite: **transparent and interpretable.**  
You can literally trace any prediction step-by-step.

But there’s a catch:  
A single tree can become *too* confident in the training data.  
It memorizes noise → **overfitting** → looks great in training, worse in real life.

Random Forests fix this.  
But first, let’s build (and visualize!) a single tree.

---

## What you’ll learn today

By the end of this lesson, you will:

- Train a **Decision Tree Classifier** and understand its structure  
- Interpret decisions directly from the tree visualization  
- Learn why trees tend to **overfit**  
- Use a **Random Forest** (an ensemble of trees) to improve performance  
- Evaluate with **accuracy + classification report** (from KNN lesson)  
- Compare performance on **Iris and Digits datasets**

---

## Step 1 — Setup

```python
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
````

Load datasets:

```python
iris = load_iris(as_frame=True)
X_iris, y_iris = iris.data, iris.target

digits = load_digits()
X_digits, y_digits = digits.data, digits.target
```

Split the data:

```python
def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_i, X_test_i, y_train_i, y_test_i = split(X_iris, y_iris)
X_train_d, X_test_d, y_train_d, y_test_d = split(X_digits, y_digits)
```

We now have two datasets:

| Dataset    | Type            | Classes | Why use it?              |
| ---------- | --------------- | ------- | ------------------------ |
| **Iris**   | Tabular numeric | 3       | Simple + visualizable    |
| **Digits** | Image-like grid | 10      | More realistic challenge |

---

## Part A — Decision Tree Classifier

Train the tree:

```python
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_i, y_train_i)

preds_tree = tree_clf.predict(X_test_i)
print("Decision Tree Accuracy (Iris):", accuracy_score(y_test_i, preds_tree))
print(classification_report(y_test_i, preds_tree))
```

### Visualize the decision-making

```python
plt.figure(figsize=(16, 10))
plot_tree(tree_clf, filled=True,
          feature_names=iris.feature_names,
          class_names=iris.target_names)
plt.title("Iris Decision Tree")
plt.show()
```

Look how the model makes decisions:

> “Is petal width ≤ 0.8?
> → Yes → Setosa.”

Each **split** reduces uncertainty in the data.
Technically, trees measure uncertainty using **Gini Impurity** —
but you can think of it as: *“How mixed is this node?”*
Pure nodes = confident predictions.

---

## The Overfitting Problem

Decision trees love **deep**, very specific rules.

Example:
“If petal width = 1.75 and sepal width > 3.2 and…”
…that rule might only apply to *one* weird training sample.

Accuracy in training: 🚀
Accuracy on new data: 😬

Let’s test the same tree on **Digits** — a harder task:

```python
preds_digits_tree = tree_clf.fit(X_train_d, y_train_d).predict(X_test_d)
print("Decision Tree Accuracy (Digits):", accuracy_score(y_test_d, preds_digits_tree))
```

You’ll likely see poorer generalization — proof of **overfitting**.

---

## Part B — Enter the Random Forest 🌲🌲🌲

> A forest is a group of trees that never all overfit in the same way.

How it works (intuition only — no rocket science):

1. It samples slightly different training subsets
2. Builds a tree for each subset
3. They **vote** on predictions

This reduces **variance**, producing robust results.

```python
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_i, y_train_i)
preds_rf = rf_clf.predict(X_test_i)

print("Random Forest Accuracy (Iris):", accuracy_score(y_test_i, preds_rf))
print(classification_report(y_test_i, preds_rf))
```

Now compare on Digits:

```python
preds_rf_digits = rf_clf.fit(X_train_d, y_train_d).predict(X_test_d)
print("Random Forest Accuracy (Digits):", accuracy_score(y_test_d, preds_rf_digits))
```

You should see **noticeable improvement** — especially for digits.

---

## Comparing performance

| Model                | Interpretability         | Overfitting Risk | Iris Accuracy | Digits Accuracy |
| -------------------- | ------------------------ | ---------------- | ------------- | --------------- |
| Single Decision Tree | ⭐⭐⭐⭐⭐ (human readable)   | High             | Good          | Weaker          |
| Random Forest        | ⭐⭐ (harder to visualize) | Much lower       | Great         | Much better     |

We’ve gained reliability — at the small cost of transparency.

---

## Understanding which features matter

Random Forests can show what features they rely on:

```python
feat_df = pd.DataFrame({
    "feature": X_iris.columns,
    "importance": rf_clf.feature_importances_
})
sns.barplot(data=feat_df, x="importance", y="feature")
plt.title("Iris Feature Importance")
plt.show()
```

This is a powerful middle ground:
**better performance + partial explainability.**

---

## Key ideas to take with you

* **Decision Trees** mimic human reasoning but **overfit** easily
* **Gini Impurity** measures how “mixed” a node is
* **Random Forest = many trees voting**
  → more stable, less overfitting
* Same metrics from KNN lesson help us compare fairly
* On complex data (Digits), forests shine even more

---

## Explore More

Optional but recommended for intuition:

* Decision Trees
  [https://www.ibm.com/think/topics/decision-trees](https://www.ibm.com/think/topics/decision-trees)
  [https://www.youtube.com/watch?v=JcI5E2Ng6r4](https://www.youtube.com/watch?v=JcI5E2Ng6r4)
  [https://www.youtube.com/watch?v=u4IxOk2ijSs](https://www.youtube.com/watch?v=u4IxOk2ijSs)
  [https://www.youtube.com/watch?v=zs6yHVtxyv8](https://www.youtube.com/watch?v=zs6yHVtxyv8)
* Random Forests
  [https://www.youtube.com/watch?v=gkXX4h3qYm4](https://www.youtube.com/watch?v=gkXX4h3qYm4)
* scikit-learn demo
  [https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html)

---

## Next up

Preventing overfitting with **hyperparameter tuning**:

* `max_depth`
* `min_samples_split`
* cross-validation
* confusion matrices

```

---
