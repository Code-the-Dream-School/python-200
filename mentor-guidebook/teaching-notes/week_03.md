# Week 3: Classification

## Overview

Students moved from predicting numbers to predicting categories. The week covered two foundational steps before touching any algorithm: preprocessing (getting data into the right shape) and evaluation (knowing what "good" actually means for a classifier). They then applied two classification algorithms — K-Nearest Neighbors and Logistic Regression — to real datasets.

## Key Concepts

**Preprocessing** — Raw data almost never goes straight into a model. The key steps are: scaling numeric features (so large-valued features don't dominate), encoding categorical features (converting text labels to numbers), and handling missing values. These steps must be fitted on training data and applied to test data — not the other way around.

**The confusion matrix** — A table showing true positives, true negatives, false positives, and false negatives. Everything else (precision, recall, F1) is derived from it. The lesson uses a Covid test as an analogy: a false negative (testing negative when positive) is very different from a false positive, and the costs of each error type depend on the problem.

**Precision vs. recall tradeoff** — Precision: of the things I flagged, how many were actually positive? Recall: of all the actual positives, how many did I catch? Improving one usually hurts the other. The right balance depends on the cost of each type of error.

**K-Nearest Neighbors (KNN)** — Classify a new point by finding the k training points closest to it and taking a majority vote. Intuitive, but sensitive to the choice of k and computationally expensive on large datasets.

**Logistic Regression** — Despite the name, it's a classifier. It models the *probability* of belonging to a class using a sigmoid function, then applies a threshold (usually 0.5) to make a binary decision. Interpretable and fast.

## Common Questions

- **"Why do we scale features?"** — Distance-based algorithms like KNN treat a 1-unit difference in income the same as a 1-unit difference in age. Scaling puts everything on the same footing. Tree-based models and logistic regression are less sensitive, but scaling rarely hurts.
- **"Why is accuracy a bad metric for imbalanced data?"** — If 99% of emails are not spam, a model that predicts "not spam" every time has 99% accuracy but is completely useless. Precision and recall tell you what's actually happening.
- **"How do I choose k in KNN?"** — Cross-validation: try several values of k and pick the one that performs best on held-out data. The lesson walks through this with the Iris dataset.

## Watch Out For

- **Fitting scalers on the full dataset** — This is the most common preprocessing mistake. The scaler should be fitted on training data only, then used to transform both training and test data. Scikit-learn pipelines are the clean way to enforce this.
- **Using accuracy on imbalanced classes** — The spam detection assignment has a class imbalance. If a student reports "95% accuracy" without checking precision/recall, ask them to look at the confusion matrix.
- **PCA confusion** — PCA (dimensionality reduction) is introduced in the preprocessing lesson. Students sometimes think it's a required step. It's a tool for specific problems (too many features, visualization), not something you always apply.

## Suggested Activities

1. **Confusion matrix game:** Describe a scenario — "You're building a model to detect fraudulent credit card transactions. 0.1% of transactions are fraud." Ask students: which is worse, a false positive (blocking a real transaction) or a false negative (missing fraud)? How does the answer change if the model is for cancer screening?

2. **K selection exercise:** Ask students to explain what happens when k=1 and k=n (where n is the size of the training set). What are the failure modes at each extreme? This reveals the bias-variance tradeoff without needing to introduce those terms formally.

3. **Classifier comparison:** The assignment includes a "classifier shootout." Ask students to share their results — which model won on their dataset? Did anyone get a different result? Why might different classifiers perform differently on the same data?
