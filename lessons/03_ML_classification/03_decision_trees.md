# Lesson 3  
### CTD Python 200  
**Decision Trees and Ensemble Learning (Random Forests)**

## Why trees?

Imagine being asked to identify a flower. You might think:

> “If the petals are tiny, it’s probably setosa.  
> If not, check petal width… big petals? Probably virginica.”

That thought process — breaking a decision into a sequence of **yes/no questions** — is exactly how a **Decision Tree** works.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/3cd4e0d6-8da7-4dc3-b05b-f1379fae0f4c" />

**Image credit: [scikit-learn.org documentation](https://www.geeksforgeeks.org/machine-learning/decision-tree/)**

Deep learning models are often described as “black boxes.”  
Decision Trees are the opposite: **transparent and interpretable**.  
You can literally trace any prediction step-by-step by following the tree branches.

However, a single decision tree can become too confident in the training data.  
This leads to overfitting: excellent performance on seen examples, weaker performance on new ones.

Random Forests solve this problem by combining many trees.


## How Decision Trees Work?
1. Start with the Root Node: It begins with a main question at the root node which is derived from the dataset’s features.

2. Ask Yes/No Questions: From the root, the tree asks a series of yes/no questions to split the data into subsets based on specific attributes.

3. Branching Based on Answers: Each question leads to different branches:

If the answer is yes, the tree follows one path.
If the answer is no, the tree follows another path.
4. Continue Splitting: This branching continues through further decisions helps in reducing the data down step-by-step.

5. Reach the Leaf Node: The process ends when there are no more useful questions to ask leading to the leaf node where the final decision or prediction is made.

Let’s look at a simple example to understand how it works. Imagine we need to decide whether to drink coffee based on the time of day and how tired we feel. The tree first checks the time:

1. In the morning: It asks “Tired?”

If yes, the tree suggests drinking coffee.
If no, it says no coffee is needed.
2. In the afternoon: It asks again “Tired?”

If yes, it suggests drinking coffee.
If no, no coffee is needed.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/646615fd-2ced-489c-8aa6-790df6802540" />

**Image credit: [scikit-learn.org documentation](https://www.geeksforgeeks.org/machine-learning/decision-tree/)**


## What you’ll learn today

By the end of this lesson, you will:

- Train and interpret a **Decision Tree Classifier**
- Visualize how splitting decisions are made
- Understand **Gini Impurity** as a measure of node “mixedness”
- Learn why individual trees can **overfit**
- Use a **Random Forest** to improve stability and accuracy
- Compare both models with evaluation metrics from the KNN lesson
- Test both **Iris** (simple) and **Digits** (more complex) datasets

## Setup

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

| Dataset    | Type            | Classes | Why use it?                         |
| ---------- | --------------- | ------- | ----------------------------------- |
| **Iris**   | Tabular numeric | 3       | Simple, easy to visualize decisions |
| **Digits** | Image-like grid | 10      | More realistic, harder to classify  |

## Part A — Decision Tree Classifier

Train the tree:

```python
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_i, y_train_i)

preds_tree = tree_clf.predict(X_test_i)
print("Decision Tree Accuracy (Iris):", accuracy_score(y_test_i, preds_tree))
print(classification_report(y_test_i, preds_tree))
```

### Visualizing decisions

```python
plt.figure(figsize=(16, 10))
plot_tree(tree_clf, filled=True, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names)
plt.title("Iris Decision Tree")
plt.show()
```

Decision Trees measure uncertainty in each split using **Gini Impurity**:
low impurity = more confident prediction.

They continue splitting until the data is as “pure” as possible.

## The Overfitting Problem

Trees can become overly specific to the training data:

```python
preds_digits_tree = tree_clf.fit(X_train_d, y_train_d).predict(X_test_d)
print("Decision Tree Accuracy (Digits):", accuracy_score(y_test_d, preds_digits_tree))
```

Notice that accuracy generally drops on the more complex Digits dataset — a sign of overfitting.

## Part B — Random Forest (Ensemble Method)

A **Random Forest** builds many trees on slightly different samples of the training data.
Each tree votes, and the most common answer wins.

This reduces overfitting because the trees do not make the same mistakes.

```python
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_i, y_train_i)
preds_rf = rf_clf.predict(X_test_i)

print("Random Forest Accuracy (Iris):", accuracy_score(y_test_i, preds_rf))
print(classification_report(y_test_i, preds_rf))
```

Test on Digits as well:

```python
preds_rf_digits = rf_clf.fit(X_train_d, y_train_d).predict(X_test_d)
print("Random Forest Accuracy (Digits):", accuracy_score(y_test_d, preds_rf_digits))
```

You should see significantly improved generalization.

## Model comparison

| Model         | Interpretability            | Overfitting Risk | Iris Accuracy | Digits Accuracy |
| ------------- | --------------------------- | ---------------- | ------------- | --------------- |
| Decision Tree | Very high                   | High             | Good          | Moderate        |
| Random Forest | More difficult to interpret | Much lower       | Very good     | Strong          |

Interpretability decreases slightly, but robustness increases substantially.

## Feature importance (Random Forest)

Which features matter most?

```python
feat_df = pd.DataFrame({
    "feature": X_iris.columns,
    "importance": rf_clf.feature_importances_
})
sns.barplot(data=feat_df, x="importance", y="feature")
plt.title("Iris Feature Importance")
plt.show()
```

This provides insight into which measurements are driving decisions.

## Key ideas to take with you

* **Decision Trees** are interpretable and mimic human decision-making
* They tend to **overfit** without constraints
* **Random Forests** combine many trees to reduce variance and improve accuracy
* Same evaluation tools from the KNN lesson let us compare fairly
* Forests shine on complex, higher-dimensional data like Digits

## Explore More

Optional resources for deeper intuition:

* Decision Trees
  [https://www.ibm.com/think/topics/decision-trees](https://www.ibm.com/think/topics/decision-trees)
  [https://www.youtube.com/watch?v=JcI5E2Ng6r4](https://www.youtube.com/watch?v=JcI5E2Ng6r4)
  [https://www.youtube.com/watch?v=u4IxOk2ijSs](https://www.youtube.com/watch?v=u4IxOk2ijSs)
  [https://www.youtube.com/watch?v=zs6yHVtxyv8](https://www.youtube.com/watch?v=zs6yHVtxyv8)

* Random Forests
  [https://www.youtube.com/watch?v=gkXX4h3qYm4](https://www.youtube.com/watch?v=gkXX4h3qYm4)

* scikit-learn example
  [https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html)

## Next steps

In the next lesson, we’ll build on this foundation and explore:

* Controlling model complexity (`max_depth`, `min_samples_split`)
* Reducing overfitting with **cross-validation**
* Confusion matrices and evaluation tradeoffs
---

