# Lesson 3  
### CTD Python 200  
**Decision Trees and Ensemble Learning (Random Forests)**

---

## Why Trees?

In the previous lesson, we learned about **K-Nearest Neighbors (KNN)**.
KNN makes predictions by comparing distances between data points.

That works well in some settings — but not all.

Now imagine a different way of thinking:

> “If an email has lots of dollar signs and exclamation points, it might be spam.  
> If it also contains words like *free* or *remove*, that makes spam even more likely.”

That style of reasoning — asking a **sequence of yes/no questions** — is exactly how a **Decision Tree** works.

<img width="800" height="400" alt="decision tree" src="https://github.com/user-attachments/assets/3cd4e0d6-8da7-4dc3-b05b-f1379fae0f4c" />

**Image credit:** GeeksForGeeks

Unlike many machine learning models that behave like black boxes, decision trees are:

- **Interpretable** — you can read every decision
- **Human-like** — they resemble flowcharts
- **Powerful on tabular data**

But they also have a weakness…

---

## The Big Idea

A **single decision tree** can become too confident.
If allowed to grow without constraints, it may memorize the training data.

This problem is called **overfitting**.

To solve it, we use **Random Forests**, which combine many trees into one stronger model.

---

## What You’ll Learn Today

By the end of this lesson, you will be able to:

- Compare **KNN vs Decision Trees vs Random Forests**
- Explain why trees outperform KNN on tabular data
- Understand **overfitting** in decision trees
- Use **Random Forests** to improve generalization
- Interpret results using **precision, recall, and F1 score**
- Connect model behavior to real-world intuition (spam detection)

---

## Dataset: Spambase (Real-World Tabular Data)

In this lesson we use the **Spambase dataset** from the UCI Machine Learning Repository.

Each row represents an **email**.
The features are numeric signals such as:

- How often words like `"free"`, `"remove"`, `"your"` appear
- Frequency of characters like `"!"` and `"$"`
- Statistics about capital letter usage

The label tells us whether the email is:

- `1` → spam  
- `0` → not spam (ham)

This dataset is ideal because:
- It’s realistic
- It has many features
- It clearly shows differences between models

---

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
````

---

## Load the Dataset

```python
from io import BytesIO
import requests

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
```

The dataset contains **4601 emails** and **58 columns**
(57 features + 1 label).

```python
print(df.shape)
df.head()
```

<img width="930" height="251" alt="Screenshot 2026-01-05 at 4 17 33 PM" src="https://github.com/user-attachments/assets/b49624ed-cc21-4d6a-8a05-4efe4f2ce950" />

---

## Train / Test Split

We separate features (`X`) from labels (`y`) and use a **stratified split**.
This keeps the spam ratio similar in both sets.

```python
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0,
    stratify=y
)
```

---

## Model 1 — KNN (Scaled Baseline)

We start with **KNN**, using proper feature scaling.
This is our **baseline** model.

```python
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

knn_scaled.fit(X_train, y_train)
pred_knn = knn_scaled.predict(X_test)

print(classification_report(y_test, pred_knn, digits=3))
```

<img width="537" height="156" alt="Screenshot 2026-01-05 at 4 18 28 PM" src="https://github.com/user-attachments/assets/d35ea1c0-fdf9-4b80-94a1-53a008ad89e7" />

### What to Notice

* KNN works reasonably well
* But performance is limited on high-dimensional tabular data
* Distance alone is not enough to capture complex patterns

This sets the stage for trees.

---

## Model 2 — Decision Tree

Decision Trees do **not use distance**.
Instead, they learn **rules** like:

> “Is the frequency of `$` greater than X?”

This makes them very effective for tabular datasets like spam detection.

```python
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)

print(classification_report(y_test, pred_tree, digits=3))
```

<img width="522" height="154" alt="Screenshot 2026-01-05 at 4 19 28 PM" src="https://github.com/user-attachments/assets/7f87bfcf-ab0e-4567-a17f-1e40689f2d66" />


### Why Trees Often Beat KNN Here

* Each feature is evaluated independently
* Trees naturally model non-linear relationships
* No scaling required
* Well-suited for mixed and sparse signals

But there’s a problem…

---

## Overfitting Warning ⚠️

A decision tree can keep splitting until it perfectly classifies the training data.

That means:

* Very low training error
* Worse performance on new data

This is **high variance** behavior.

To fix this, we use ensembles.

---

## Model 3 — Random Forest 🌲🌲🌲

A **Random Forest** is a collection of decision trees.

Each tree:

* Sees a random sample of the data
* Uses a random subset of features
* Makes its own prediction

The forest **votes**, and the majority wins.

```python
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=0
)

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print(classification_report(y_test, pred_rf, digits=3))
```

<img width="517" height="163" alt="Screenshot 2026-01-05 at 4 20 15 PM" src="https://github.com/user-attachments/assets/70956555-6be3-430e-a4f8-dca201c3c261" />


---

## Why Random Forests Work Better

* Individual trees make different mistakes
* Voting cancels out errors
* Variance is reduced
* Generalization improves

This is called **ensemble learning**:

> Many weak learners → one strong learner

---

## Comparing F1 Scores

Accuracy alone can be misleading for spam detection.
We care about both:

* Catching spam (recall)
* Not blocking real emails (precision)

The **F1 score** balances both.

```python
models = {
    "KNN (scaled)": pred_knn,
    "Decision Tree": pred_tree,
    "Random Forest": pred_rf
}

for name, preds in models.items():
    score = f1_score(y_test, preds)
    print(f"{name:15s} F1 = {score:.3f}")
```

<img width="297" height="65" alt="Screenshot 2026-01-05 at 4 21 13 PM" src="https://github.com/user-attachments/assets/81688ef4-279c-4f0d-a38c-f08c81944f15" />


### Typical Pattern You’ll See

```
KNN (scaled)     < Decision Tree < Random Forest
```

---

## Final Takeaways

### Decision Trees

* Easy to understand and visualize
* Excellent for tabular data
* High variance → prone to overfitting

### Random Forests

* Reduce overfitting
* Improve reliability
* Often the best default choice for tabular ML

| Concept  | Lesson                        |
| -------- | ----------------------------- |
| KNN      | Distance-based, needs scaling |
| Trees    | Rule-based, interpretable     |
| Forests  | Ensembles reduce variance     |
| F1 Score | Balances precision & recall   |

> **If you want strong real-world performance on tabular data, Random Forests are a safer choice than a single tree.**

---

## Next Steps

In upcoming lessons, we will:

* Use **cross-validation** for more reliable evaluation
* Tune hyperparameters
* Explore **feature importance** and model interpretation
* Discuss trade-offs between different error types
---
