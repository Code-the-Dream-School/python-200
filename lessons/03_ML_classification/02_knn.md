# Lesson 2  
## CTD Python 200  
### k-Nearest Neighbor (KNN) Classifiers

---

## Why Do We Need Classifiers?

So far in this course, we’ve focused on *describing* data: loading it, exploring it, and summarizing patterns.
Now we reach an important turning point.

A **classifier** is a model that assigns a category (or label) to a data point.
Instead of asking questions like:

> “What is the average value?”

we now ask questions like:

> “Which category does this data point belong to?”

Examples of classification problems include deciding whether an email is spam, identifying the species of a flower, or determining whether a transaction is fraudulent.

In this lesson, you will build and evaluate your **first hands-on classifier**:
the **k-Nearest Neighbor (KNN)** algorithm.

KNN is intentionally simple.
That simplicity allows us to focus on the core ideas behind classification before moving on to more complex models.

---

## The Intuition Behind KNN

<img width="790" height="461" alt="KNN intuition diagram" src="https://github.com/user-attachments/assets/515472de-f80d-4149-8a38-dda46229305d" />

**Image credit:** GeeksforGeeks

Imagine you discover a new flower in a garden.
You don’t know its species, but you *do* know the species of many nearby flowers.

A natural strategy might be:

> “Look at the flowers that are most similar to this one.
> If most of them belong to the same species, this one probably does too.”

This is exactly how **k-Nearest Neighbor** works.

When KNN classifies a new data point, it:
- finds the **k closest points** in the training data,
- looks at their labels,
- and lets them **vote** on the final prediction.

If most neighbors belong to the same class, that class wins.

There is no complex training phase here.
KNN simply stores the data and compares new points to what it has already seen.

---

## A Tiny Example (Pause and Think)

Suppose we choose **k = 3**.

A new flower’s three nearest neighbors include:
- 2 flowers from the *Setosa* species
- 1 flower from the *Versicolor* species

**Question:**  
What will KNN predict?

**Answer:**  
KNN predicts *Setosa*, because it receives the majority of votes.

This simple voting idea is the foundation of the entire algorithm.

---

## A First Dataset: Iris

To learn classification, we want a dataset that is:
small, clean, easy to understand, and well-studied.

For this lesson, we use the classic **Iris dataset**.

Each row represents a flower.
For each flower, we measure four physical properties:
sepal length, sepal width, petal length, and petal width.

The label tells us which species the flower belongs to.
There are three possible species in total.

The Iris dataset is simple enough for learning,
but rich enough to demonstrate real classification behavior.

---

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
````

---

## Loading the Iris Dataset

```python
iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

print(X.shape)
X.head()
```

<img width="735" height="225" alt="Iris dataset preview" src="https://github.com/user-attachments/assets/73cb1811-3b89-43b6-82bf-14c2b2b48fdc" />

The dataset contains **150 flowers** and **4 numeric features**.
The target labels are encoded as numbers, but they correspond to real species names.

---

## Train / Test Split

Before building any model, we split the data into two parts.

The **training set** is used to make decisions.
The **test set** is kept separate and used only at the end.

This separation is critical.
Our goal is not to memorize the data,
but to perform well on **new, unseen examples**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## Our First KNN Model (Without Scaling)

Let’s start with a basic KNN classifier using **k = 5**.
This means each prediction is based on the five closest neighbors.

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
```

<img width="549" height="208" alt="KNN classification report" src="https://github.com/user-attachments/assets/3b861154-c8ca-432f-b256-70bab7e08baa" />

At first glance, the accuracy looks quite good.
However, this result hides an important issue.

---

## Why Distance Can Be Tricky

KNN determines similarity by computing **distances** between data points.
Distance calculations are strongly affected by the *scale* of features.

If one feature ranges from 0 to 100 and another ranges from 0 to 1,
the larger-scale feature will dominate the distance —
even if it is not more informative.

As a result, KNN can make poor decisions simply because features are measured in different units.

To fix this problem, we use **feature scaling**.

---

## KNN with Feature Scaling (The Right Way)

Feature scaling transforms each feature so that it:

* has a mean of 0,
* and a standard deviation of 1.

After scaling, all features contribute fairly to distance calculations.

```python
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

knn_scaled.fit(X_train, y_train)
preds_scaled = knn_scaled.predict(X_test)

print("Accuracy (scaled):", accuracy_score(y_test, preds_scaled))
print(classification_report(y_test, preds_scaled))
```

<img width="524" height="205" alt="Scaled KNN results" src="https://github.com/user-attachments/assets/4dc08460-6198-421f-a558-ce6035cc65dc" />

In most cases, you will see improved performance.

> **Key takeaway:**
> KNN almost always requires feature scaling.

---

## Understanding Evaluation Metrics

So far, we’ve focused on **accuracy** —
the fraction of predictions the model gets correct.

Accuracy is useful, but it does not tell the whole story.

To understand classifiers more deeply, we also consider:

* **Precision**, which measures how reliable positive predictions are,
* **Recall**, which measures how many real cases the model successfully identifies,
* **F1 score**, which balances precision and recall.

These metrics are especially important in real-world problems,
where different mistakes have very different consequences.

---

## Confusion Matrix: Seeing Errors Clearly

A confusion matrix helps us visualize *where* the model makes mistakes.

```python
cm = confusion_matrix(y_test, preds_scaled)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
plt.show()
```

<img width="641" height="475" alt="Confusion matrix for KNN" src="https://github.com/user-attachments/assets/1d045263-93db-4df1-92fb-4cabe796e26e" />

Even when accuracy is high, confusion matrices can reveal
which classes are being confused with one another.

---

## What We’ve Learned

In this lesson, you:

* built and evaluated your **first classifier**,
* learned how KNN uses distance and voting,
* saw why feature scaling is essential,
* and explored evaluation metrics beyond accuracy.

These ideas will appear again and again as we move to more advanced models.

---

## External References (Recommended)

If you would like additional explanations from different perspectives:

**IBM (text):**
[https://www.ibm.com/think/topics/knn](https://www.ibm.com/think/topics/knn)

**IBM Technology (video, ~10 minutes):**
[https://www.youtube.com/watch?v=b6uHw7QW_n4](https://www.youtube.com/watch?v=b6uHw7QW_n4)

Seeing the same idea explained in multiple ways helps build strong intuition.

---

## Looking Ahead

In the next lesson, we introduce **Decision Trees**.

Decision Trees take a very different approach from KNN.
Instead of measuring distance, they learn a sequence of rules.

Understanding KNN deeply will make it much easier to understand
why decision trees — and later, random forests — are so powerful.
