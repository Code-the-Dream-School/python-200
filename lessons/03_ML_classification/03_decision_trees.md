# Lesson 3  
## CTD Python 200  
### Decision Trees and Ensemble Learning (Random Forests)

---

## Before We Begin: Optional Learning Resources

If you’d like extra intuition before or after this lesson, these resources are very helpful:

**Text (Displayr – very intuitive):**  
https://www.displayr.com/what-is-a-decision-tree/

**Video (StatQuest, ~8 minutes):**  
https://www.youtube.com/watch?v=7VeUPuFGJHk  

Seeing trees explained visually makes everything that follows much easier to understand.

---

## From Distance to Decisions

In the previous lesson, we learned **K-Nearest Neighbors (KNN)**.  
KNN makes predictions by measuring *distance* between data points.

That works well for small, clean datasets like Iris.

But many real-world problems don’t behave like points in space.

Now imagine how *you* decide whether an email is spam:

> “Does the email contain lots of dollar signs?”  
> “Does it use words like *free* or *winner*?”  
> “Are there long blocks of capital letters?”

This style of reasoning — asking a **sequence of yes/no questions** — is exactly how a **Decision Tree** works.

<img width="800" alt="decision tree example" src="https://github.com/user-attachments/assets/4fdb3b63-6e0d-4b2b-a1c0-7f65e84a50f4" />

**Image credit:** Displayr

Decision trees are powerful because they resemble **human decision-making**.  
They feel like flowcharts rather than equations.

---

## What Is a Decision Tree?

A decision tree repeatedly asks questions like:

> “Is this feature greater than some value?”

Each question splits the data into smaller and more focused groups.  
Eventually, the tree reaches a **leaf**, where it makes a prediction.

Unlike KNN:
- Trees do **not** use distance
- Trees do **not** need feature scaling
- Trees work very well with real-world tabular data

To see why, we’ll use a realistic dataset.

---

## Dataset: Spambase (Real Email Data)

In this lesson we use the **Spambase dataset** from the UCI Machine Learning Repository.

Each row represents an **email**.  
Each column represents a measurable signal from that email.

Some examples of what the features capture:
- How often words like `"free"`, `"remove"`, `"your"` appear
- Frequency of symbols like `"!"` and `"$"`
- Statistics about capital letter usage

The label tells us whether the email is:
- `1` → spam  
- `0` → not spam (ham)

This dataset is ideal because it is:
- Messier than Iris
- High-dimensional
- Much closer to real-world ML problems

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

## Loading the Dataset

```python
from io import BytesIO
import requests

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
```

At first glance, this dataset looks intimidating — just numbers.
That’s normal. Our job as data scientists is to **give meaning to numbers**.

```python
print(df.shape)
df.head()
```

The dataset contains **4,601 emails** and **58 columns**
(57 features + 1 target label).

---

## Quick EDA: Understanding the Data

Before modeling, we pause and explore.

### 1. Is the dataset balanced?

```python
df.iloc[:, -1].value_counts()
```

We see both spam and non-spam emails are well represented.
This makes evaluation more reliable.

---

### 2. Capital letters as a signal

One feature measures how much of an email is written in capital letters.

```python
plt.hist(df.iloc[:, 54], bins=30)
plt.title("Capital Letter Run Length Average")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

Very large values often correspond to **aggressive spam formatting**.

---

### 3. Why this matters

Unlike Iris, these features are:

* Not spatial
* Not symmetric
* Not naturally distance-based

This is where decision trees shine.

---

## Train / Test Split

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

Stratification ensures that both sets contain similar proportions of spam and non-spam emails.

---

## Baseline Model: KNN (With Scaling)

We begin with KNN again, this time **properly scaled**.
This serves as our baseline.

```python
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

knn_scaled.fit(X_train, y_train)
pred_knn = knn_scaled.predict(X_test)

print(classification_report(y_test, pred_knn, digits=3))
```

KNN performs reasonably well, but struggles with:

* High dimensionality
* Sparse, rule-like patterns

Now we introduce trees.

---

## How Decision Trees Decide (Intuition)

At each split, a decision tree asks:

> “Which question best separates spam from non-spam?”

To answer this, the tree uses a measure called **Gini impurity**.

You don’t need the formula. The intuition is enough:

* High impurity → mixed classes
* Low impurity → mostly one class

Each split tries to **reduce impurity** as much as possible.

---

## Model 2 — Decision Tree

```python
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)

print(classification_report(y_test, pred_tree, digits=3))
```

You should see a noticeable improvement over KNN.

This happens because:

* Trees evaluate features independently
* They capture non-linear rules
* They align well with how spam is structured

---

## A Cautionary Note: Overfitting

A decision tree can keep splitting until it memorizes the training data.

That leads to:

* Excellent training performance
* Worse performance on new emails

This is called **overfitting**.

Rather than fixing a single tree, we take a smarter approach.

---

## Model 3 — Random Forests 🌲🌲🌲

A **Random Forest** builds many trees and lets them vote.

Each tree:

* Sees a random subset of emails
* Uses a random subset of features
* Makes its own prediction

Together, they form a much more reliable model.

```python
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=0
)

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print(classification_report(y_test, pred_rf, digits=3))
```

---

## Comparing Models with F1 Score

Spam detection needs balance:

* Catch spam (recall)
* Avoid blocking real emails (precision)

The **F1 score** captures both.

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

You’ll typically observe:

```
KNN < Decision Tree < Random Forest
```

---

## What We’ve Learned

In this lesson, you:

* Learned how decision trees make decisions
* Saw why trees outperform KNN on tabular data
* Understood overfitting intuitively
* Used random forests to reduce variance
* Evaluated models using F1 score

---

## Looking Ahead

Next, we will:

* Tune tree depth and forest size
* Use cross-validation
* Interpret feature importance
* Discuss real-world trade-offs

---

### 🚀 You’ve just taken a major step toward real-world machine learning.

```

---
