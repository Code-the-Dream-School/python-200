# k-Nearest Neighbor (KNN) Classifiers

If you’d like an additional explanation of KNN before or after this lesson, these are excellent references you can return to at any time.

**Text (IBM):**
[https://www.ibm.com/think/topics/knn](https://www.ibm.com/think/topics/knn)

**Video (IBM Technology, ~10 minutes):**
[https://www.youtube.com/watch?v=b6uHw7QW_n4](https://www.youtube.com/watch?v=b6uHw7QW_n4)



## Why Do We Need Classifiers?

Earlier in the course, and especially in last week’s overview of machine learning, we talked about how many systems you use every day rely on ML models behind the scenes. Spam filters, recommendation systems, fraud detection, face recognition, and search engines all rely heavily on *classification algorithms*.

So far in this course, we’ve focused on *describing* data: loading it, exploring it, and summarizing patterns. Now we reach an important turning point.

A *classifier* is a model that assigns a category (or label) to a data point. Instead of asking:

> "What is the average value?"

We now ask:

> "Which category does this data point belong to?"

Examples of classification problems include deciding whether an email is spam, identifying the species of a flower, or determining whether a transaction is fraudulent.

In this lesson, you will build and evaluate your first hands-on classifier: the *k-Nearest Neighbor (KNN)* algorithm.

KNN is intentionally simple. That simplicity allows us to focus on the *core ideas behind classification* before moving on to more complex models later in the course.

## The Intuition Behind KNN

<img width=”790” height=”461” alt=”KNN intuition diagram” src=”https://github.com/user-attachments/assets/515472de-f80d-4149-8a38-dda46229305d” />

**Image credit:** GeeksforGeeks

Imagine you discover a new flower in a garden. You don’t know its species, but you *do* know the species of many nearby flowers.

A natural strategy might be:

> “Look at the flowers that are most similar to this one.
> If most of them belong to the same species, this one probably does too.”

This is exactly how *k-Nearest Neighbor* works. When KNN classifies a new data point, it finds the `k` closest points in the training data, looks at their labels, and lets them vote on the final prediction. There is no complex training phase -- KNN simply stores the data and compares new points to what it has already seen.

## A Tiny Example (Pause and Think)

Suppose we choose `k = 3`.

A new flower’s three nearest neighbors include:

* two flowers from the *Setosa* species,
* one flower from the *Versicolor* species.

KNN predicts *Setosa*, because that label receives the majority of votes.

This simple voting idea is the foundation of the entire algorithm.

<img width=”717” height=”422” alt=”KNN voting example” src=”https://github.com/user-attachments/assets/962bbc4f-2998-4995-b531-d33df1bbdc4e” />

**Image Credit:** IBM

## The Iris Dataset: The “Hello World” of Classification

<img width=”1000” height=”447” alt=”Iris flowers” src=”https://github.com/user-attachments/assets/01bd1abe-b4c6-463a-9991-b002773bba2c” />

**Image Credit:** CodeSignal

Before building any model, we need to understand our data. The *Iris dataset* is often called the *hello world of classification* -- small, clean, balanced, and extremely well-studied, making it perfect for learning.

Each row represents a flower. For each flower, we measure:

* sepal length
* sepal width
* petal length
* petal width

All measurements are in centimeters. The label tells us which species the flower belongs to: *Setosa*, *Versicolor*, or *Virginica*. In real-world projects, data exploration always comes *before* modeling, so we begin with a small amount of exploratory data analysis (EDA) to build intuition.

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
```

## Loading the Dataset

```python
iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

print(X.shape)
X.head()
```

<img width="704" height="237" alt="Screenshot 2026-01-30 at 1 39 58 PM" src="https://github.com/user-attachments/assets/d6b9714b-e9ab-4768-affd-50dc7952ca8b" />

The dataset contains 150 flowers and 4 numeric features. There are no missing values, and all features are measured in the same units.

## Quick EDA: Building Intuition

First, we check whether the dataset is balanced.

```python
sns.countplot(x=y.map(dict(enumerate(iris.target_names))))
plt.title("Number of Flowers per Species")
plt.show()
```

<img width="651" height="484" alt="Screenshot 2026-01-30 at 1 42 03 PM" src="https://github.com/user-attachments/assets/f5ac73a9-4f6d-41ad-9800-aea75ea04f5d" />

Each species appears the same number of times, which makes model evaluation more reliable. Next, we look at how petal measurements separate species:

```python
sns.scatterplot(
    x=X["petal length (cm)"],
    y=X["petal width (cm)"],
    hue=y.map(dict(enumerate(iris.target_names)))
)
plt.title("Petal Length vs Petal Width")
plt.show()
```

<img width="647" height="469" alt="Screenshot 2026-01-30 at 1 42 31 PM" src="https://github.com/user-attachments/assets/c4dcf161-9c70-4e96-9282-7bc3dd789dc6" />

Petal measurements separate species extremely well, especially *Setosa*. Sepal measurements overlap more, but the pairplot below gives a fuller picture of all feature relationships together:

```python
sns.pairplot(
    pd.concat([X, y.rename("species")], axis=1),
    hue="species"
)
plt.show()
```

<img width="1058" height="986" alt="download" src="https://github.com/user-attachments/assets/4f5b1a22-b533-4e02-8235-414b8441b8b8" />


From just a few plots, we already learn that some features are much more informative than others and that a simple classifier should work well.

## Train / Test Split

Before modeling, we split the data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The `stratify=y` argument ensures each species appears in similar proportions in both sets, making our evaluation fair.

## Our First KNN Model

In the previous scikit-learn lesson, you learned the *standard model-building API* used throughout this course:

1. Create the model
2. Fit the model
3. Make predictions
4. Evaluate results

We follow that exact pattern here.

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
```

<img width="531" height="207" alt="Screenshot 2026-01-12 at 11 06 50 AM" src="https://github.com/user-attachments/assets/5f0f8fa1-a17a-410f-9ba1-822cc5526e71" />

You should see strong performance -- even a simple, intuitive method like KNN can work surprisingly well on clean data. Note: very high scores like these are *unusual* in real-world datasets. The Iris dataset is intentionally simple and well-separated; in realistic problems like spam detection or fraud detection, perfect scores are extremely rare.

## Understanding Evaluation Metrics

You do *not* need to master these metrics on your own right now -- we covered them in detail in the classifier evaluation lesson, which you should refer back to as needed. Briefly: accuracy tells us how often the model is correct, precision and recall describe different types of errors, and the F1-score balances the two.

## Confusion Matrix: Seeing Errors Clearly

The confusion matrix shows where the model is getting confused. Each row represents the true species and each column the predicted species, so numbers along the diagonal are correct predictions and off-diagonal numbers are mistakes.

```python
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
plt.show()
```

<img width="648" height="470" alt="Screenshot 2026-01-12 at 11 09 56 AM" src="https://github.com/user-attachments/assets/7946f208-9d04-4a67-9ab1-7e6ce9e0ecd6" />

Even when accuracy is high, confusion matrices help us see *which* classes are confused with each other.




