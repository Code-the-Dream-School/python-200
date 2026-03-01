# Week 3 Assignments

This week's assignments cover the week 3 material, including:

- Data preprocessing: scaling and train/test splitting
- k-Nearest Neighbors (KNN) and classifier evaluation
- Cross-validation and hyperparameter tuning
- Logistic Regression and regularization

As with previous weeks, the warmup exercises are meant to build muscle memory for the core mechanics -- try to work through them without AI assistance. The mini-project will ask you to apply these tools in a more open-ended context.

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_03/`. Inside that folder, create two files and an outputs directory:

1. `warmup_03.py`  : for the warmup exercises
2. `project_03.py` : for the mini-project
3. `outputs/`      : for any plots or data files your code generates

When finished, commit and open a PR as described in the [assignments README](README.md).

# Part 1: Warmup Exercises

Put all warmup exercises in a single file: `warmup_03.py`. Use comments to mark each section and question (e.g. `# --- Preprocessing ---` and `# Q1`). Use `print()` to display all outputs.

All questions in this warmup use the Iris dataset. Run this setup block once at the top of your file and reuse the variables throughout:

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
```

## Preprocessing

### Preprocessing Question 1

Split `X` and `y` into training and test sets using an 80/20 split with `stratify=y` and `random_state=42`. Print the shapes of all four arrays.

### Preprocessing Question 2

Fit a `StandardScaler` on `X_train` and use it to transform both `X_train` and `X_test`. Print the mean of each column in `X_train_scaled` -- they should all be very close to 0. Add a comment explaining in one sentence why you fit the scaler on `X_train` only.

## KNN

### KNN Question 1

Build a `KNeighborsClassifier` with `n_neighbors=5`, fit it on the *unscaled* training data (`X_train`), and predict on the test set. Print the accuracy score and the full classification report.

### KNN Question 2

Repeat KNN Question 1 using the *scaled* data (`X_train_scaled`, `X_test_scaled`). Print the accuracy score. Add a comment: does scaling improve performance, hurt it, or make no difference? Why might that be for this particular dataset?

### KNN Question 3

Using `cross_val_score` with `cv=5`, evaluate the k=5 KNN model on the unscaled training data. Print each fold score, the mean, and the standard deviation. Add a comment: is this result more or less trustworthy than a single train/test split, and why?

### KNN Question 4

Loop over k values `[1, 3, 5, 7, 9, 11, 13, 15]`. For each, compute 5-fold cross-validation accuracy on the unscaled training data and print k and the mean CV score. Add a comment identifying which k you would choose and why.

## Classifier Evaluation

### Classifier Evaluation Question 1

Using your predictions from KNN Question 1, create a confusion matrix and display it with `ConfusionMatrixDisplay`, passing `display_labels=iris.target_names`. Save the figure to `outputs/knn_confusion_matrix.png`. Add a comment: which pair of species does the model most often confuse?

## The sklearn API: Decision Trees

### Decision Trees Question 1

Create a `DecisionTreeClassifier(max_depth=3, random_state=42)`, fit it on the unscaled training data, and predict on the test set. Print the accuracy score and classification report. Add a comment comparing the Decision Tree accuracy to KNN. Then add a second comment: given that Decision Trees don't rely on distance calculations, would scaled vs. unscaled data affect the result?

## Logistic Regression and Regularization

### Logistic Regression Question 1

Train three logistic regression models on the scaled Iris data, identical in every way except for the `C` parameter: `C=0.01`, `C=1.0`, and `C=100`. Use `max_iter=1000` and `solver='liblinear'` for all three. For each model, print the `C` value and the total size of all coefficients using `np.abs(model.coef_).sum()`. Add a comment: what happens to the total coefficient magnitude as `C` increases? What does this tell you about what regularization is doing?
