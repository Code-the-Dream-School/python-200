# Lesson 1
### CTD Python 200  
**Introduction to scikit-learn and the Machine Learning Ecosystem**

---

## Contributor Note
Focus this lesson on three things only:
- Why scikit-learn matters for non-deep-learning ML  
- The core API: create → fit → predict  
- A tiny K-Means clustering demo  

Save model evaluation, pipelines, cross-validation, etc. for Week 2.

---

## Motivation — Why scikit-learn?

<img width="550" height="333" alt="Screenshot 2025-10-24 at 3 09 00 PM" src="https://github.com/user-attachments/assets/056d6b1a-4184-4bfd-9473-216739e4dfb7" />


⭐ Think: Netflix recommendations, Spotify mixes, fraud alerts.  
These systems use machine learning to learn patterns from data.

In Python, the go-to toolkit for traditional (non-deep-learning) ML is **scikit-learn (sklearn)**.

✅ Reliable and well-maintained  
✅ Free and open-source  
✅ Consistent APIs across many algorithms  

<img width="529" height="262" alt="Screenshot 2025-10-24 at 3 09 06 PM" src="https://github.com/user-attachments/assets/fde97815-318b-4912-8ab5-78f50434c7f2" />


Before deep learning, learn these foundations — most projects build on them.

---

## Learning Goals

By the end of this lesson, you can:

- Explain why scikit-learn is used for day-to-day ML.
- Use the core API pattern: **create → fit → predict.**
- Run a tiny unsupervised example with **K-Means.**

(Everything else—evaluation, pipelines—comes in Week 2.)

---

## The Ecosystem (30-second tour)

- **NumPy**: fast numeric arrays  
- **Pandas**: data loading/cleaning (tables/CSVs)  
- **Matplotlib / Seaborn**: quick plots  
- **scikit-learn**: the ML algorithms + helpers  

> Mental model: *Pandas gets the data ready → scikit-learn learns from it → Matplotlib shows results.*

---

## Install & Import

```bash
pip install scikit-learn
```

```python
import sklearn
```
You’ll usually import specific parts as needed.

---

## The Core API Pattern — “Fit → Predict”

Every estimator in scikit-learn uses the same shape:

1. Create a model instance  
2. `fit(X, y)` → learn from data  
3. `predict(new_X)` → make predictions on new data  

---

### Mini Example (Regression flavor)

(Just to see the pattern once; we’ll demo clustering next.)

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# temperature (°C) vs cupcakes sold
X = np.array([[15],[18],[21],[24],[27]])
y = np.array([150,200,240,310,400])

model = LinearRegression()   # 1) create
model.fit(X, y)              # 2) fit
print(model.predict([[30]])) # 3) predict → e.g. [460.5]
```

---

## Demo — K-Means Clustering (Unsupervised)

We’ll cluster customers by two simple features (e.g., spending score, visit frequency).  
No labels — just structure discovery.

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Fake 2D customer data: [spending_score, visit_frequency]
X = np.random.rand(50, 2)

# 1) create
kmeans = KMeans(n_clusters=3, random_state=42)

# 2) fit (learn cluster centers)
kmeans.fit(X)

# 3) predict labels for each point
labels = kmeans.predict(X)

# quick plot
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.title("Customer Segments (K-Means)")
plt.xlabel("Spending Score")
plt.ylabel("Visit Frequency")
plt.show()
```

### Takeaways
- K-Means groups similar points — no labels required  
- Same **create → fit → predict** API  
- Useful for quick segmentation (e.g., frequent vs casual customers)

---

## Assessment / Practice

**Q1.** What are the two main types of machine learning scikit-learn supports?  
**A1.** Supervised and Unsupervised learning.  

**Q2.** What’s the basic pattern every model follows?  
**A2.** Create → Fit → Predict.  

**Q3.** Which Python libraries are commonly used alongside scikit-learn?  
**A3.** NumPy, Pandas, Matplotlib, Joblib.  

**Q4 (Bonus).** What tool helps you combine multiple steps like scaling and modeling into one workflow?  
**A4.** The Pipeline system.

---

## What’s Next (Week 2 teaser)

- Train/test splits, metrics, cross-validation  
- Preprocessing and Pipelines so you don’t leak information  
- Comparing models the right way  

---

# Lesson 1.5  
### CTD Python 200  
**Hands-On Lab: The Core API (Fit → Predict) + Tiny Clustering**

---

## Contributor Note
Keep this lab minimal: one classification toy task (Iris) to practice the API, then a very short K-Means run to mirror the demo. No metrics deep-dive yet — just show predictions and a basic plot.

---

## Objectives

- Load a small dataset  
- Practice **create → fit → predict**  
- See a simple clustering plot  

---

## Setup

```bash
pip install scikit-learn pandas matplotlib seaborn
```

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

## Part A — Iris (Classification in 8 lines)

```python
iris = load_iris(as_frame=True)
df = iris.frame

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=200)  # create
clf.fit(X_train, y_train)               # fit
preds = clf.predict(X_test)             # predict

pd.DataFrame({'actual': y_test.values, 'pred': preds}).head()
```

(Optional quick viz—no deep evaluation yet):  

```python
sns.pairplot(df, hue='target', diag_kind='hist')
plt.show()
```

---

## Part B — Mini K-Means (Unsupervised)

```python
X = np.random.rand(60, 2)  # pretend: [spend, visits]

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X)

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.title("K-Means: 3 quick segments")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

---

## Key Takeaways – Lesson 1.5

1️⃣ The scikit-learn workflow is simple: **Create → Fit → Predict.**  
   You practiced it with both classification and clustering.  

2️⃣ **Classification example (Iris):**  
   - Used LogisticRegression to predict flower species.  
   - Learned from labeled data → *supervised learning.*  

3️⃣ **Clustering example (K-Means):**  
   - Grouped unlabeled points into segments.  
   - Found patterns without knowing answers → *unsupervised learning.*  

4️⃣ **Same API across tasks:**  
   Whether it’s regression, classification, or clustering, the process (fit, then predict) stays consistent.  

5️⃣ **Visualization helps you see what the model learned.**  
   Simple scatter or pair plots make patterns and clusters easy to understand.

---

✅ **Bottom line:**  
You now know how to take raw data, fit a model, make predictions, and visualize results — the essential first step in any machine-learning workflow.
