# Lesson 1  
### CTD Python 200  
**Introduction to scikit-learn and the Machine Learning Ecosystem**

## Why learn scikit-learn?

<img width="550" height="333" alt="Scikit-learn overview diagram" src="https://github.com/user-attachments/assets/056d6b1a-4184-4bfd-9473-216739e4dfb7" />

*Image credit: scikit-learn.org documentation*

Think about the “smart” systems you use every day — Netflix recommending your next show, Spotify building a playlist that just fits, or your bank detecting fraud.

All rely on **machine learning** — detecting patterns in data to predict or automate decisions.

And in Python, the most popular toolkit for this is **scikit-learn** (`sklearn`).

- Free and open source  
- Reliable and well-documented  
- Best-in-class for **classical, structured-data ML**  

<img width="529" height="262" alt="scikit-learn workflow diagram" src="https://github.com/user-attachments/assets/fde97815-318b-4912-8ab5-78f50434c7f2" />

Before exploring deep learning tools like TensorFlow or PyTorch, it's important to start here — because scikit-learn teaches the foundations used everywhere else.

## What you’ll learn today

By the end of this lesson, you’ll be able to:

1. Explain why scikit-learn matters in ML  
2. Recognize the core API: **Create → Fit → Predict**  
3. Try a quick demo of **K-Means clustering**, an unsupervised ML technique  

## A quick tour of scikit-learn

`scikit-learn` sits on top of NumPy, SciPy, and Matplotlib and supports:

- **Supervised learning** — predict known labels  
- **Unsupervised learning** — find hidden structure  
- Tools for data preprocessing, evaluation, and workflows  

And nearly every model follows the same pattern:

```python
model = ModelClass()                 # 1. Create
model.fit(X_data, y_data)            # 2. Learn
y_predictions = model.predict(X_test) # 3. Predict
````

That consistency makes it fast to try new ideas.

Documentation:
[https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

## The ecosystem in context

| Library            | Role                  |
| ------------------ | --------------------- |
| NumPy              | Fast math + arrays    |
| Pandas             | Data loading + tables |
| Matplotlib/Seaborn | Visualization         |
| scikit-learn       | Machine learning      |

> Pandas prepares → scikit-learn learns → Matplotlib shows the results

## Installation and setup

```bash
pip install scikit-learn
```

Imports we’ll use today:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
```

## First steps — the core API in action

We’ll predict cupcake sales based on temperature with **Linear Regression**.
(*We’ll go deeper into Linear Regression in the next lesson.*)

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# temperature (°C) vs cupcakes sold
X = np.array([[15],[18],[21],[24],[27]])
y = np.array([150,200,240,310,400])

model = LinearRegression()   # Create
model.fit(X, y)              # Fit (learn)
print(model.predict([[30]])) # Predict
```

**Output**

```
[460.5]
```

Predicting the future — not bad for one line of math!

## Demo — K-Means clustering (unsupervised learning)

Unsupervised learning = finding structure without labels.

Example: Grouping customers by spending + visit frequency.

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create synthetic data with 3 clear clusters
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42)  # Create
kmeans.fit(X)                                   # Fit
labels = kmeans.predict(X)                      # Predict

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=60)
plt.title("Customer Segments Found by K-Means")
plt.xlabel("Feature 1 – Spending Score")
plt.ylabel("Feature 2 – Visit Frequency")
plt.show()
```

<img width="656" height="478" alt="K-Means clustering output" src="https://github.com/user-attachments/assets/a107db74-8ea1-49da-a8d2-7e1c0afe1953" />

K-Means groups nearby points into clusters — no instructions needed.
It’s a powerful way to explore unknown datasets.

## Key takeaways

* scikit-learn powers most *everyday* ML in Python
* Its **Create → Fit → Predict** API works across all models
* Two major ML branches:

  * **Supervised** = predict labels
  * **Unsupervised** = discover structure
* K-Means is a simple yet useful unsupervised algorithm

## Next steps

Upcoming lessons will cover:

* Evaluating model performance
* Train/Test splits
* Pipelines for clean workflows
* A deeper look at **Linear Regression**

Until then, recommended beginner-friendly resources:

* [https://courses.dataschool.io/introduction-to-machine-learning-with-scikit-learn/](https://courses.dataschool.io/introduction-to-machine-learning-with-scikit-learn/)
* [https://www.youtube.com/watch?v=SW0YGA9d8y8](https://www.youtube.com/watch?v=SW0YGA9d8y8)
* [https://www.youtube.com/watch?v=SIEaLBXr0rk](https://www.youtube.com/watch?v=SIEaLBXr0rk)

```

---

Awesome — here is the **cleaned + streamlined** version of **Lesson 1.5** that aligns with the formatting of Lesson 1:

* Removed excess emojis & horizontal rules
* Removed repeated screenshots of identical plots
* Trimmed exploratory suggestions to fit CTD style
* Kept focus on **Core API: Create → Fit → Predict**
* Aligned tone + structure to Lesson 1
* Added a short recap + preview of what’s next

---

# Lesson 1.5  
### CTD Python 200  
**Hands-On Lab: The Core Fit → Predict Workflow + Tiny Clustering**

## What you’ll do

In this short lab, you’ll take your **first hands-on step** into machine learning with scikit-learn.

You’ll practice the key API pattern used in nearly every ML workflow:

> **Create → Fit → Predict**

We’ll complete two mini projects:

1. **Classification** using the classic *Iris* dataset  
2. **Clustering** using K-Means  

## Step 1 — Setup

Use Jupyter Notebook or Google Colab.

```bash
pip install scikit-learn pandas matplotlib seaborn
````

Imports:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
```

---

## Part A — Iris Classification (Supervised Learning)

We’ll teach a model to classify flowers based on their measurements.

### Load the dataset

```python
iris = load_iris(as_frame=True)
df = iris.frame
df.head()
```

### Visualize the features

```python
sns.pairplot(df, hue='target', diag_kind='hist')
plt.show()
```

You can see that the species separate well in certain feature combinations — great for training a model.

### Split into training and testing data

```python
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Train + predict

```python
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train) 
preds = clf.predict(X_test)
```

Look at a sample:

```python
pd.DataFrame({'actual': y_test.values, 'predicted': preds}).head()
```

Congrats — you just trained your first supervised ML model!

---

## Part B — Mini K-Means Clustering (Unsupervised Learning)

No labels here — the model will discover structure on its own.

### Generate some synthetic data

```python
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=42)
```

### Run K-Means

```python
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X)
```

### Visualize results

```python
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=60)
plt.title("K-Means: 3 Discovered Clusters")
plt.xlabel("Feature 1 – Spending Score")
plt.ylabel("Feature 2 – Visit Frequency")
plt.show()
```

K-Means grouped similar customers — without ever being told what groups should exist.

---

## Key takeaways

* You practiced the **Create → Fit → Predict** workflow
* You built:

  * A **classification** model (with labels)
  * A **clustering** model (without labels)
* Visualization helps reveal what your model learned

These steps are exactly what data scientists do every day.

---

## Try on your own

A few ways to explore further:

* Change the train/test split (e.g., `test_size=0.3`)

* Try a different supervised model:

  ```python
  from sklearn.tree import DecisionTreeClassifier
  clf = DecisionTreeClassifier()
  ```

* Try a different number of clusters:

  ```python
  kmeans = KMeans(n_clusters=4)
  ```

Small experiments = big learning.

---

## Moving forward

Next lesson:
**Evaluating models + Train/Test splits + Intro to Pipelines**

```
---
