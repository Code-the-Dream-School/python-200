# Assignment: k-Nearest Neighbor (KNN) Classifiers

In this assignment, you’ll build your **first distance-based classifier** using the k-Nearest Neighbor (KNN) algorithm.

The goal is not just to run code — it’s to:
- Practice exploratory data analysis (EDA)
- Understand how KNN makes predictions
- Learn how evaluation metrics help interpret model performance

This assignment has two parts:

**Warmup exercises (short, focused practice)**
**Mini-project (real classification workflow)**

---

## How to Submit

Submit either:

- A **GitHub notebook/script** link, OR  
- A **Kaggle/Colab notebook** with outputs saved.

At the top of your notebook include:

1. What dataset you used  
2. What you built  
3. One insight you learned  

---

# Part 1: Warmup Exercises (Do Without AI)

These should be quick. Focus on understanding, not speed.

---

## Task 1 — Load and Explore the Iris Dataset

Load the Iris dataset using scikit-learn.

### Requirements:
- Print dataset shape
- Display first 5 rows
- Print class counts

### Reflection (2–3 sentences):
- What does one row represent?
- Why is Iris often called the “hello world” dataset?

---

## Task 2 — Visual Exploration

Create two visualizations:

1. Scatter plot:
   - petal length vs petal width
   - color by species

2. One additional visualization of your choice:
   - histogram, boxplot, or pairplot

### Reflection:
Which features seem most useful for classification? Why?

---

## Task 3 — Train/Test Split Practice

Split the data using:

- `test_size=0.2`
- `random_state=42`
- `stratify=y`

### Short answer:
Why do we use `stratify=y`?

---

## Task 4 — First KNN Model

Train a KNN classifier:

```python
KNeighborsClassifier(n_neighbors=5)
````

Then:

* Fit on training data
* Predict on test data
* Print accuracy

### Reflection:

In your own words, how does KNN make a prediction?

---

## Task 5 — Evaluation Metrics

Print:

* Classification report
* Confusion matrix

### Reflection:

* Which species is easiest to predict?
* Which is hardest?
* Why might that be?

---

# Part 2: Mini Project — Choosing the Best K for KNN

Now you’ll do a slightly more realistic ML workflow.

## Goal:

Find the best value of **k** for a KNN classifier.

---

## Step 1 — Test Multiple k Values

Train KNN models with:

```
k = 1, 3, 5, 7, 9, 11
```

For each k:

* Train model
* Compute accuracy on test set
* Store results

---

## Step 2 — Plot Results

Create a line plot:

* X-axis: k value
* Y-axis: accuracy

This helps visualize the effect of k.

---

## Step 3 — Interpret Results

Write ~5 sentences answering:

* Which k worked best?
* What happens when k is very small?
* What happens when k is large?
* How does this relate to overfitting vs generalization?

(No need for formal definitions — intuition is enough.)

---

## Step 4 — Final Model

Using your best k:

* Retrain model
* Print classification report
* Display confusion matrix

---

# Optional Stretch (Highly Recommended)

Try ONE of these:

### Option A — Feature Scaling Experiment

Standardize features using:

```python
StandardScaler()
```

Compare performance with and without scaling.

Write 2–3 sentences:
Did scaling help? Why might that happen?

---

### Option B — New Dataset Challenge

Try KNN on another dataset:

* Wine dataset (sklearn)
* Breast cancer dataset (sklearn)

Repeat the workflow:

EDA → Train/Test Split → KNN → Evaluation

Write 5–7 sentences about differences from Iris.

---

# What We’re Looking For

A strong submission includes:

Clean, readable notebook
Code runs without errors
Visualizations included
Written interpretation (not just numbers)
Clear explanation of KNN intuition

---

# Tips If You Get Stuck

* Revisit the KNN lesson notebook
* Check scikit-learn documentation
* Ask questions in Slack — that’s expected

Remember:

**Confusion is part of learning ML.**