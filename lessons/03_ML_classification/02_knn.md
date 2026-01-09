# Lesson 2  
### CTD Python 200  
**k-Nearest Neighbor (KNN) Classifiers**

---

## Why Do We Need Classifiers?

So far in this course, we’ve focused on working with data: loading it, exploring it, and summarizing it.
Now we reach an important turning point.

A **classifier** is a model that assigns labels to data.
Instead of answering questions like “what is the average?” we now ask questions like:

> “Which category does this data point belong to?”

This lesson introduces your **first hands-on classifier**:
the **k-Nearest Neighbor (KNN)** algorithm.

KNN is simple, intuitive, and powerful enough to teach us many core ideas that apply to *all* classification models.

---

## The Intuition Behind KNN


<img width="790" height="461" alt="Screenshot 2026-01-09 at 1 31 57 PM" src="https://github.com/user-attachments/assets/515472de-f80d-4149-8a38-dda46229305d" />

**Image Credits- GeeksforGeeks**

Imagine you discover a new flower.
You don’t know its species, but you *do* know the species of many nearby flowers.

A natural strategy might be:

> “Look at the flowers most similar to this one.
> If most of them are the same species, that’s probably what this one is too.”

This is exactly how **k-Nearest Neighbor** works.

When we want to classify a new data point, KNN:

1. Looks at the **k closest points** in the training data  
2. Checks their labels  
3. Assigns the most common label among them  

There is no training phase in the usual sense.
KNN simply **stores the data** and compares new points to it.

If this feels very human and intuitive, that’s a good sign.

---

## A First Dataset: Iris

To learn classification, we want a dataset that is:

- Small
- Clean
- Easy to understand
- Well-studied

For this lesson, we’ll use the classic **Iris dataset**.

Each row represents a flower.
For each flower, we measure four things:

- Sepal length
- Sepal width
- Petal length
- Petal width

The label tells us which species the flower belongs to.
There are three possible species.

This dataset is simple enough to learn on, but rich enough to reveal important ideas.

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
```

## Loading the Iris Dataset

```python
iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

print(X.shape)
X.head()
```

<img width="735" height="225" alt="Screenshot 2026-01-09 at 1 25 08 PM" src="https://github.com/user-attachments/assets/73cb1811-3b89-43b6-82bf-14c2b2b48fdc" />

We have 150 flowers and 4 numeric features.
The target labels are encoded as numbers, but they correspond to real species names.

## Train / Test Split

Before building any model, we split our data into two parts.

The training set is what the model learns from.
The test set is used only at the end, to see how well the model generalizes to new data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The key idea here is fairness:
the model should be evaluated on data it has never seen before.

## Our First KNN Model (Without Scaling)

Let’s start with a very simple KNN classifier.
We’ll use k = 5, meaning the model will look at the 5 nearest neighbors.


```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

```
<img width="549" height="208" alt="Screenshot 2026-01-09 at 1 27 42 PM" src="https://github.com/user-attachments/assets/3b861154-c8ca-432f-b256-70bab7e08baa" />

You will likely see fairly good accuracy — but this result hides an important problem.

## Why Distance Can Be Tricky

KNN decides what is “near” by computing distances between points.
But distance depends heavily on scale.

If one feature has values between 0 and 100, and another ranges between 0 and 1,
the larger feature will dominate the distance calculation — even if it isn’t more important.

In other words, KNN can be misled simply because features are measured in different units.

To fix this, we need feature scaling.

## KNN with Feature Scaling (The Right Way)

We will now standardize all features so they:

Have mean 0

Have standard deviation 1

This puts every feature on equal footing.

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

<img width="524" height="205" alt="Screenshot 2026-01-09 at 1 28 58 PM" src="https://github.com/user-attachments/assets/4dc08460-6198-421f-a558-ce6035cc65dc" />

In most cases, you should see an improvement.
This is a crucial lesson:

# KNN almost always requires feature scaling.

## Understanding Evaluation Metrics

So far, we’ve looked at accuracy — the fraction of predictions that were correct.

Accuracy is useful, but it does not tell the whole story.
To understand classifiers more deeply, we also look at:

Precision: When the model predicts a class, how often is it correct?

Recall: When a class is present, how often does the model find it?

F1 score: A balance between precision and recall

These metrics become especially important in real-world problems like spam detection or medical diagnosis, where different types of errors have different costs.

The classification report shows all of these values together.

## Confusion Matrix: Seeing Errors Clearly

A confusion matrix helps us visualize where the model is making mistakes.

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

<img width="641" height="475" alt="Screenshot 2026-01-09 at 1 30 20 PM" src="https://github.com/user-attachments/assets/1d045263-93db-4df1-92fb-4cabe796e26e" />

This plot shows which species are being confused with one another.
Even when accuracy is high, confusion matrices can reveal patterns of error.

What We’ve Learned

In this lesson, we saw that:

KNN classifies by comparing distances between points

The choice of k matters

Feature scaling is essential for distance-based models

Accuracy alone is not enough to evaluate classifiers

Precision, recall, and confusion matrices give deeper insight

Most importantly, you’ve now trained and evaluated your first classifier.


## External References (Recommended)

If you’d like another explanation of KNN from different perspectives, these are excellent resources:

Text explanation (IBM):
https://www.ibm.com/think/topics/knn

Video explanation (IBM Technology, 10 minutes):
https://www.youtube.com/watch?v=b6uHw7QW_n4

Reading or watching the same idea explained in multiple ways is one of the best ways to build intuition.

## Looking Ahead

In the next lesson, we’ll introduce Decision Trees.

Decision Trees take a very different approach from KNN:
instead of measuring distance, they learn a sequence of rules.

Understanding KNN deeply will make it much easier to understand why trees — and later, random forests — are so powerful.

