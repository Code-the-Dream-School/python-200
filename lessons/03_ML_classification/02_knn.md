# Lesson 2  
## CTD Python 200  
### k-Nearest Neighbor (KNN) Classifiers

---

## Before We Begin: Optional Learning Resources

If you’d like an additional explanation of KNN before or after this lesson, these are excellent references:

**Text (IBM):**  
https://www.ibm.com/think/topics/knn  

**Video (IBM Technology, ~10 minutes):**  
https://www.youtube.com/watch?v=b6uHw7QW_n4  

These resources explain the same ideas using different examples and visuals, which can really help build intuition.

---

## Why Do We Need Classifiers?

So far in this course, we’ve focused on *describing* data: loading it, exploring it, and summarizing patterns.  
Now we reach an important turning point.

A **classifier** is a model that assigns a category (or label) to a data point.  
Instead of asking:

> “What is the average value?”

We now ask:

> “Which category does this data point belong to?”

Examples of classification problems include deciding whether an email is spam, identifying the species of a flower, or determining whether a transaction is fraudulent.

In this lesson, you will build and evaluate your **first hands-on classifier**:  
the **k-Nearest Neighbor (KNN)** algorithm. KNN is intentionally simple. That simplicity lets us focus on the *core ideas behind classification* before moving on to more complex models.

---

## The Intuition Behind KNN

<img width="790" height="461" alt="KNN intuition diagram" src="https://github.com/user-attachments/assets/515472de-f80d-4149-8a38-dda46229305d" />

**Image credit:** GeeksforGeeks

Imagine you discover a new flower in a garden. You don’t know its species, but you *do* know the species of many nearby flowers.

A natural strategy might be:

> “Look at the flowers that are most similar to this one.  
> If most of them belong to the same species, this one probably does too.”

This is exactly how **k-Nearest Neighbor** works.

When KNN classifies a new data point, it:
- finds the **k closest points** in the training data,
- looks at their labels,
- and lets them **vote** on the final prediction.

There is no complex training phase. KNN simply stores the data and compares new points to what it has already seen.

---

## A Tiny Example (Pause and Think)

Suppose we choose **k = 3**.

A new flower’s three nearest neighbors include:
- two flowers from the *Setosa* species,
- one flower from the *Versicolor* species.

**What will KNN predict?**

It predicts *Setosa*, because that label receives the majority of votes.

This simple voting idea is the foundation of the entire algorithm.

<img width="717" height="422" alt="Screenshot 2026-01-12 at 11 26 58 AM" src="https://github.com/user-attachments/assets/962bbc4f-2998-4995-b531-d33df1bbdc4e" />

**Image Credit:** IBM

---

## The Iris Dataset: The “Hello World” of Classification

<img width="1000" height="447" alt="image" src="https://github.com/user-attachments/assets/01bd1abe-b4c6-463a-9991-b002773bba2c" />

**Image Credit:** Code Signal

Before we build any model, we need to understand our data.

The **Iris dataset** is often called the *hello world of classification*. It is small, clean, balanced, and extremely well-studied — perfect for learning.

Each row represents a flower, and for each flower we measure:
- **Sepal length**
- **Sepal width**
- **Petal length**
- **Petal width**

All measurements are in centimeters. The label tells us which species the flower belongs to:
*Setosa*, *Versicolor*, or *Virginica*.
Before jumping into machine learning, let’s do a small amount of **exploratory data analysis (EDA)** — just enough to build intuition.

---

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

## Loading the Dataset-

```python
iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

print(X.shape)
X.head()
```

<img width="710" height="221" alt="Screenshot 2026-01-12 at 10 46 12 AM" src="https://github.com/user-attachments/assets/56f4fb4b-5fb2-4525-a430-b9695e4b82e4" />

The dataset contains 150 flowers and 4 numeric features.
There are no missing values, and all features are measured in the same units.

---

## Quick EDA: Building Intuition-
1- How many flowers of each species?

```python
sns.countplot(x=y.map(dict(enumerate(iris.target_names))))
plt.title("Number of Flowers per Species")
plt.show()
```

<img width="709" height="480" alt="Screenshot 2026-01-12 at 10 48 26 AM" src="https://github.com/user-attachments/assets/23459eeb-d3b0-4b32-a3bf-0750c5951320" />

We see that the dataset is perfectly balanced. Each species has the same number of examples.

2- Petal Length vs Petal Width

```python
sns.scatterplot(
    x=X["petal length (cm)"],
    y=X["petal width (cm)"],
    hue=y.map(dict(enumerate(iris.target_names)))
)
plt.title("Petal Length vs Petal Width")
plt.show()
```

<img width="721" height="478" alt="Screenshot 2026-01-12 at 10 49 33 AM" src="https://github.com/user-attachments/assets/41341f42-5f41-47af-ae3b-695f7a7a6aca" />

This plot shows something very important which is petal measurements separate species extremely well, especially Setosa.

3- Sepal Length vs Sepal Width

```python
sns.scatterplot(
    x=X["sepal length (cm)"],
    y=X["sepal width (cm)"],
    hue=y.map(dict(enumerate(iris.target_names)))
)
plt.title("Sepal Length vs Sepal Width")
plt.show()
```

<img width="695" height="469" alt="Screenshot 2026-01-12 at 10 50 59 AM" src="https://github.com/user-attachments/assets/7b474535-e052-4bb1-8e66-607e7b65cdc9" />

Sepal measurements overlap more, making classification harder using only these features.

4- Pairwise Feature Relationships

```python
sns.pairplot(
    pd.concat([X, y.rename("species")], axis=1),
    hue="species"
)
plt.show()
```

<img width="1058" height="986" alt="image" src="https://github.com/user-attachments/assets/1bf0a8d7-547f-46b2-8e2f-64836bdd0d3c" />

This gives us a high-level overview of how features relate to each other and to the labels.

## What We Learn from EDA

From just a few plots, we already learn:

Some features separate classes much better than others
The data is clean and well-structured
Classification should be feasible with a simple model
Now we’re ready to build our first classifier.

---

## Train / Test Split-

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The stratify=y argument ensures that each species appears in roughly the same proportion in both the training and test sets. This makes our evaluation fair and reliable.

## Our First KNN Model

This is the moment where we actually build and use our first machine learning model. First, we create a KNN classifier with n_neighbors=5.
This means that whenever the model makes a prediction, it will look at the five most similar flowers in the training data and let them vote on the label.

Next, we call .fit(X_train, y_train).
For KNN, this step does not learn equations or rules. Instead, it simply stores the training data so it can compare new flowers to it later.

Then we use .predict(X_test) to ask the model to classify flowers it has never seen before. Each prediction is based on which training flowers are closest to that test flower. Finally, we evaluate how well the model did. Seeing strong performance here is encouraging, it shows that even a very simple, intuitive method like KNN can work surprisingly well on real data.

**At this point, you have officially trained and evaluated your first classifier!!!** 

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
```

<img width="531" height="207" alt="Screenshot 2026-01-12 at 11 06 50 AM" src="https://github.com/user-attachments/assets/5f0f8fa1-a17a-410f-9ba1-822cc5526e71" />

You should see very strong performance, even with this simple model.

## A Note on Feature Scaling (And Why We Skip It Here)

You may have heard that KNN usually requires feature scaling — and that is generally true. However, in this specific dataset:
All features are measured in centimeters
All features are on similar scales
Values are in the same order of magnitude

To keep this first example as simple and intuitive as possible, we intentionally skip scaling here. Interestingly, scaling can sometimes make performance slightly worse on this dataset — a subtle effect that we will explore later in assignments. For now, our goal is understanding how KNN works, not optimizing performance.

## Understanding Evaluation Metrics

This classification report shows perfect performance on the test set.

Accuracy = 1.0 means the model predicted every flower correctly.

Precision = 1.0 means whenever the model predicted a species, it was correct.

Recall = 1.0 means the model successfully found all flowers of each species.

F1-score = 1.0 confirms a perfect balance between precision and recall.

Each class has a support of 10, meaning the test set is balanced.

## Confusion Matrix: Seeing Errors Clearly

The confusion matrix shows where the model is getting confused. Each row represents the true species of a flower. Each column represents the species predicted by the model. Numbers along the diagonal are correct predictions. Numbers off the diagonal are mistakes.

This visualization helps us see which species the model mixes up, not just how often it is correct. Even when accuracy is high, the confusion matrix reveals patterns in the errors that accuracy alone cannot show.

In this case, most predictions are correct, and the few mistakes tend to happen between species that are naturally similar.

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

Even when accuracy is high, confusion matrices reveal which classes are being confused.

## What We’ve Learned

In this lesson, you:
- Built and evaluated your first classifier
- Learned how KNN uses distance and voting
- Explored the Iris dataset through EDA
- Saw why evaluation metrics matter

These ideas form the foundation for everything that comes next.

## Looking Ahead

In the next lesson, we introduce Decision Trees. Instead of measuring distance, decision trees learn a sequence of rules.

Understanding KNN deeply will make it much easier to understand why trees and later random forests are so powerful.

#### You’ve just taken your first real step into machine learning!!!
