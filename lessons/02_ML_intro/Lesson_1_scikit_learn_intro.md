# Lesson 1
### CTD Python 200  
**Introduction to scikit-learn and the Machine Learning Ecosystem**

## Why learn scikit-learn?

<img width="550" height="333" alt="Screenshot 2025-10-24 at 3 09 00 PM" src="https://github.com/user-attachments/assets/056d6b1a-4184-4bfd-9473-216739e4dfb7" />

**Image credit: scikit-learn.org documentation**

Think about all the “smart” systems you use every day — Netflix recommending what to watch next, Spotify building a playlist that just *gets* you, or your bank flagging a suspicious transaction.  
All of these rely on **machine learning** — computers learning patterns from data so they can make predictions or decisions automatically.

When you start doing machine learning in Python, the first tool most professionals reach for is **scikit-learn** (often imported as `sklearn`).  
It’s the go-to toolkit for what we call **classical ML** — the kind that works beautifully on tables, CSVs, and structured data.

1. Free and open source  
2. Beautifully designed and consistent  
3. One of the best-maintained projects in the Python data ecosystem  

<img width="529" height="262" alt="Screenshot 2025-10-24 at 3 09 06 PM" src="https://github.com/user-attachments/assets/fde97815-318b-4912-8ab5-78f50434c7f2" />

**Image credit: [https://www.geeksforgeeks.org/machine-learning]**

Before moving on to deep-learning frameworks like TensorFlow or PyTorch, it’s important to understand how things work in scikit-learn — because nearly every modern ML project builds on these same ideas.

## What you’ll learn today

By the end of this lesson, you’ll be able to:

1. Explain why scikit-learn is such a core part of the ML toolkit.  
2. Recognize the **core API pattern** every scikit-learn model follows: `create → fit → predict`.  
3. Try a quick demo of **K-Means clustering**, an unsupervised algorithm that finds patterns without being told what’s right or wrong.

## A quick tour of scikit-learn

`scikit-learn` is built on top of **NumPy**, **SciPy**, and **Matplotlib**, and gives you tools for nearly every classical ML task.

- **Supervised learning** — predicting labels from examples  
  *(e.g., spam vs not-spam, house-price prediction)*
- **Unsupervised learning** — finding structure without labels  
  *(e.g., clustering similar customers, reducing data dimensions)*
- **Utilities** — preprocessing data, splitting into train/test sets, evaluating models, and building pipelines that keep your workflow tidy

And here’s the best part: every model, from a simple linear regression to a fancy random forest, uses the same rhythm.

```python
model = ModelClass()      # 1. Create
model.fit(X_data, y_data) # 2. Learn from data
y_predictions = model.predict(X_test) # 3. Predict on new inputs
````

That **create → fit → predict** pattern is what makes scikit-learn so pleasant to use — once you learn it, you can apply it to almost any algorithm.

If you’d like to explore the official documentation (highly recommended!), visit:
[https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

## The scikit-learn ecosystem in context

Machine learning in Python usually means using several libraries together.
Here’s the “cast of characters” you’ll see in nearly every project:

| Library                  | What it does                              |
| :----------------------- | :---------------------------------------- |
| **NumPy**                | Handles numbers, arrays, and fast math    |
| **Pandas**               | Loads and manipulates tables or CSV files |
| **Matplotlib / Seaborn** | Visualizes your results                   |
| **scikit-learn**         | Learns from the data                      |

You can think of them working together like this:

> **Pandas** gets the data ready → **scikit-learn** learns from it → **Matplotlib** shows the results.

## Installation and setup

Install scikit-learn from the command line:

```bash
pip install scikit-learn
```

And import it (plus a few essentials):

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
```

## First steps — seeing the API in action

Let’s warm up with a tiny example.
Imagine you own a bakery and want to predict cupcake sales based on the temperature outside.
We’ll use **Linear Regression** for that, and we’ll take a deeper dive into Linear Regression in the next lesson.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# temperature (°C) vs cupcakes sold
X = np.array([[15],[18],[21],[24],[27]])
y = np.array([150,200,240,310,400])

model = LinearRegression()   # 1. create
model.fit(X, y)              # 2. fit (learn)
print(model.predict([[30]])) # 3. predict
```

**Output:**

```
[460.5]
```

So on a 30 °C day, our model predicts we’ll sell about 460 cupcakes!
This pattern — `create → fit → predict` — will come up again and again in your ML journey.

## Demo — K-Means clustering (unsupervised learning)

Now let’s look at an example where the model finds patterns *without* being told the answers.
This is called **unsupervised learning**.

Imagine you run a coffee-shop chain. You’ve collected data on how often customers visit and how much they spend, and you want to find natural customer groups — maybe “regulars,” “occasional visitors,” and “rare guests.”

We’ll use **K-Means clustering** to discover those groups automatically.

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create synthetic data with 3 clear clusters
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=42)

# 1. Create the model
kmeans = KMeans(n_clusters=3, random_state=42)

# 2. Fit to the data (find cluster centers)
kmeans.fit(X)

# 3. Predict cluster labels
labels = kmeans.predict(X)

# 4. Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=60)
plt.title("Customer Segments Found by K-Means")
plt.xlabel("Feature 1 – Spending Score")
plt.ylabel("Feature 2 – Visit Frequency")
plt.show()
```

<img width="656" height="478" alt="Screenshot 2025-11-06 at 9 18 30 PM" src="https://github.com/user-attachments/assets/a107db74-8ea1-49da-a8d2-7e1c0afe1953" />

**Image credit: Google Colab**

What you’ll see: three colorful clusters.
Each color represents one of the groups the algorithm discovered.
K-Means figured out which points are close together and assigned them the same label — without you ever telling it what the groups should be!

This is a great way to explore data when you don’t yet know what patterns might exist.

## Key takeaways

* **scikit-learn** is the foundation of most “everyday” ML projects in Python.
* Its consistent **create → fit → predict** API makes it easy to experiment with different algorithms.
* **Supervised learning** uses labeled data (e.g., predicting house prices).
* **Unsupervised learning** finds structure in unlabeled data (e.g., clustering customers).
* **K-Means** is one example of unsupervised ML — it can automatically reveal groups in your dataset.

## Next steps

In the next lessons, we’ll build on this foundation and explore:

* How to evaluate model performance
* How to split data into training and testing sets
* How to use **Pipelines** to combine preprocessing and modeling

Until then, if you’d like more practice, try these free beginner resources:

* [Data School’s Intro to Machine Learning with scikit-learn](https://courses.dataschool.io/introduction-to-machine-learning-with-scikit-learn/)
* [https://www.youtube.com/watch?v=SW0YGA9d8y8](https://www.youtube.com/watch?v=SW0YGA9d8y8)
* [https://www.youtube.com/watch?v=SIEaLBXr0rk](https://www.youtube.com/watch?v=SIEaLBXr0rk)

