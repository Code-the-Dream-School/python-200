# Lesson 3  
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

**Image credit: [decision-tree-geeks documentation](https://www.geeksforgeeks.org/machine-learning/decision-tree/)**

---

## What you’ll learn today

By the end of this lesson, you will be able to:

- Train and interpret a **Decision Tree Classifier**  
- Measure node uncertainty using **Gini Impurity**  
- Explain why trees tend to **overfit**  
- Use a **Random Forest** to improve accuracy and robustness  
- Evaluate models using the metrics from the KNN lesson  
- Compare performance on the **Iris** and **Digits** datasets  

---

## Setup

This part is your roadmap for the lesson.
We tell you which models you’ll use (Decision Tree and Random Forest) and which datasets you’ll work with.
You’ll also see how to measure performance and compare models fairly using the same metrics.

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

The Iris dataset is a small table of 150 flowers.
For each flower, we have 4 numbers: sepal length, sepal width, petal length, and petal width (all in cm).
Each row is labeled as one of 3 species: setosa, versicolor, or virginica.
It’s simple, clean, and perfect for learning classification.

The Digits dataset is like tiny black-and-white pictures of handwritten numbers.
Each digit image is 8×8 pixels, flattened into 64 numbers that say how dark each pixel is.
The label tells us which digit (0–9) the image represents.
It’s more complex than Iris and closer to a “real” machine learning problem.

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

Here we split each dataset into a training set and a test set.
The model sees the training data and learns from it, but never sees the test data during training.
Later, we use the test set to check how well the model generalizes to new, unseen examples.

```python
def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_i, X_test_i, y_train_i, y_test_i = split(X_iris, y_iris)
X_train_d, X_test_d, y_train_d, y_test_d = split(X_digits, y_digits)
```

We now have two datasets:

| Dataset | Type            | Classes | Why use it?                         |
| ------- | --------------- | ------- | ----------------------------------- |
| Iris    | Tabular numeric | 3       | Simple, easy to visualize decisions |
| Digits  | Image-like grid | 10      | More realistic, harder to classify  |

---

## Part A — Decision Tree Classifier

### Train the tree (Iris)

In this section, we train a single decision tree on the Iris data.
We then use it to make predictions on the test set and measure how often it’s correct.
This gives us a first baseline: how good is one tree by itself?

```python
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_i, y_train_i)

preds_tree = tree_clf.predict(X_test_i)
print("Decision Tree Accuracy (Iris):", accuracy_score(y_test_i, preds_tree))
print(classification_report(y_test_i, preds_tree))
```

<img width="537" height="243" alt="Screenshot 2025-11-20 at 2 41 28 PM" src="https://github.com/user-attachments/assets/5da09bd0-2b4e-4f01-b15b-72a581b48d9d" />

**Image Credit: Google Colab**

---

### Visualizing decisions

```python
plt.figure(figsize=(18, 12))
plot_tree(
    tree_clf,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree - Iris Dataset")
plt.show()
```

<img width="1698" height="1157" alt="Visualize" src="https://github.com/user-attachments/assets/e83a4922-2cdd-4fa4-97f2-d8dccb64acb6" />

**Image Credit: Google Colab**

Decision Trees measure impurity at each split.
Lower impurity = clearer separation between classes → more confident prediction.

---

## The Overfitting Problem

A tree can become too specialized to the training data.
Now we test the same tree on the more complex Digits dataset.
We’ll see that accuracy is worse, because the tree tends to memorize specific patterns from the training set.
This introduces the idea of overfitting: doing great on training data but worse on new data.

```python
preds_digits_tree = tree_clf.fit(X_train_d, y_train_d).predict(X_test_d)
print("Decision Tree Accuracy (Digits):", accuracy_score(y_test_d, preds_digits_tree))
```

Result- Decision Tree Accuracy (Digits): 0.825

### What do we learn from this result?

When we trained the Decision Tree on the simple Iris dataset, it performed very well — almost perfect accuracy.

But when we tried the **same model** on the more complex Digits dataset, accuracy dropped to around **82–83%**.
This shows that:

* The tree **memorized** many tiny details in the training data
* But those details **did not apply** to the new digit images
* So the model **struggles** when the task is harder and more varied

This behavior is called **overfitting**:

> The model is smart on training data…
> but not very smart on new data.

In real machine learning work, we want models that **generalize**, not just memorize —
and this is exactly why we introduce **Random Forests** next.

On the more complex **Digits** dataset, accuracy typically drops — a sign of overfitting and poorer generalization.

---

## Part B — Random Forest (Ensemble Method)

A **Random Forest** builds many trees on slightly different samples of the training data.
Each tree votes, and the most common answer wins.

Imagine you have a complex problem to solve, and you gather a group of experts from different fields to provide their input. Each expert provides their opinion based on their expertise and experience. Then, the experts would vote to arrive at a final decision.

**Example**
In the diagram below, we have a random forest with n decision trees, and we’ve shown the first 5, along with their predictions (either “Dog” or “Cat”). Each tree is exposed to a different number of features and a different sample of the original dataset, and as such, every tree can be different. Each tree makes a prediction.

Looking at the first 5 trees, we can see that 4/5 predicted the sample was a Cat. The green circles indicate a hypothetical path the tree took to reach its decision. The random forest would count the number of predictions from decision trees for Cat and for Dog, and choose the most popular prediction.

<img width="712" height="376" alt="Screenshot 2025-11-20 at 2 08 19 PM" src="https://github.com/user-attachments/assets/7311ca18-2d66-48b1-8d11-4b073b09a975" />

**Image Credits:[https://www.datacamp.com/tutorial/random-forests-classifier-python]**

```python
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_i, y_train_i)
preds_rf = rf_clf.predict(X_test_i)

print("Random Forest Accuracy (Iris):", accuracy_score(y_test_i, preds_rf))
print(classification_report(y_test_i, preds_rf))
```

<img width="537" height="242" alt="Screenshot 2025-11-20 at 3 27 38 PM" src="https://github.com/user-attachments/assets/484072b2-ae60-4930-bd27-1daa66147194" />

**Image Credits:Google Colab**

Test on Digits as well:

```python
preds_rf_digits = rf_clf.fit(X_train_d, y_train_d).predict(X_test_d)
print("Random Forest Accuracy (Digits):", accuracy_score(y_test_d, preds_rf_digits))
```

<img width="530" height="362" alt="Screenshot 2025-11-20 at 3 28 53 PM" src="https://github.com/user-attachments/assets/743efffe-ab2e-48b9-b4e8-e2abe82c9be4" />

**Image Credits:Google Colab**

You should see improved generalization compared to a single tree, especially on Digits.

---

## Confusion Matrices (Iris + Digits)

Confusion matrices help you see **which classes** are being confused with which.

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

Now we compare three models side by side:

* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest

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
axes[0].set_ylim(0, 1.05)

sns.barplot(data=df_digits, x="model", y="accuracy", ax=axes[1])
axes[1].set_title("Accuracy Comparison - Digits")
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.show()
```

This gives a visual summary of how each model performs on both datasets.

---

## Feature Importance (Random Forest)

Random Forests can also tell us which features matter most.

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

In the Iris dataset, petal length and petal width are usually the most important features.

---

## Key takeaways

* **Decision Trees** are highly interpretable and mimic human decision-making
* They tend to **overfit** if we let them grow unchecked
* **Random Forests** combine many trees to reduce variance and improve accuracy
* Evaluation should include:

  * Accuracy
  * Classification reports
  * Confusion matrices
* Random Forests often shine on more complex, higher-dimensional data like **Digits**

---

## Explore More
* Decision Trees

  * [https://www.ibm.com/think/topics/decision-trees](https://www.ibm.com/think/topics/decision-trees)
  * [https://www.youtube.com/watch?v=JcI5E2Ng6r4](https://www.youtube.com/watch?v=JcI5E2Ng6r4)

* Random Forests

  * [https://www.youtube.com/watch?v=gkXX4h3qYm4](https://www.youtube.com/watch?v=gkXX4h3qYm4)

* scikit-learn demo

  * [https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html)

---

## Next steps

In upcoming lessons, we will:

* Tune hyperparameters such as `max_depth`, `min_samples_split`, and `n_estimators`
* Use **cross-validation** to better estimate model performance
* Explore confusion matrices in more depth and discuss error trade-offs

