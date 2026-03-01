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

Using your predictions from KNN Question 1, create a confusion matrix and display it with `ConfusionMatrixDisplay`, passing `display_labels=iris.target_names`. Save the figure to `outputs/knn_confusion_matrix.png`. Add a comment: which pair of species does the model most often confuse (if any)?

## The sklearn API: Decision Trees

### Decision Trees Question 1

Create a `DecisionTreeClassifier(max_depth=3, random_state=42)`, fit it on the unscaled training data, and predict on the test set. Print the accuracy score and classification report. Add a comment comparing the Decision Tree accuracy to KNN. Then add a second comment: given that Decision Trees don't rely on distance calculations, would scaled vs. unscaled data affect the result?

## Logistic Regression and Regularization

### Logistic Regression Question 1

Train three logistic regression models on the scaled Iris data, identical in every way except for the `C` parameter: `C=0.01`, `C=1.0`, and `C=100`. Use `max_iter=1000` and `solver='liblinear'` for all three. For each model, print the `C` value and the total size of all coefficients using `np.abs(model.coef_).sum()`. Add a comment: what happens to the total coefficient magnitude as `C` increases? What does this tell you about what regularization is doing?

# Part 2: Mini-Project -- Spam or Ham? A Classifier Shootout

This project uses the [Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) from the UCI Machine Learning Repository -- the same dataset introduced in the logistic regression lesson. Each row represents an email. The 57 numeric features describe measurable properties: how often certain words appear, how often certain characters appear, or statistics about runs of capital letters. The target variable is `spam_label` (1 = spam, 0 = not spam).

Your goal is to build the best spam classifier you can, understand why different models behave differently on this dataset, and package your best model into a reusable prediction pipeline.

Put all code in `project_03.py` and save any figures to `outputs/`.

## Task 1: Load and Explore

The logistic regression lesson shows exactly how to load this dataset. Adapt that code for your script.

Once it is loaded, take some time to understand what you are working with. How many emails are in the dataset? How balanced are the two classes? What does that balance (or imbalance) mean for how you should interpret a raw accuracy score?

Now explore how a few key features differ between spam and ham. For each of `word_freq_free`, `char_freq_!`, and `capital_run_length_total`, create a boxplot showing the distribution of that feature for spam emails versus ham emails. Save them to `outputs/`. What do you notice? Are the differences between classes dramatic or subtle?

Then look at the raw scale of the features more broadly. Notice that many emails have a value of zero for most word-frequency features -- most emails do not contain the word "free" at all. What does this heavy skew toward zero tell you about the data? Why does the numeric scale vary so dramatically across features (some are tiny fractions, others reach into the thousands)? Why might that matter for some of the models you are about to build?

## Task 2: Prepare Your Data

Before building any models, prepare your data for the experiments in Task 3. You will need a train/test split and will need to think about how to handle the feature scales you noticed in Task 1. Document your choices in comments.

## Task 3: A Classifier Comparison

Build and evaluate the following five classifiers. For each, print the accuracy and the full classification report.

- `KNeighborsClassifier(n_neighbors=5)` trained on the *unscaled* data
- `KNeighborsClassifier(n_neighbors=5)` trained on the *scaled* data
- `DecisionTreeClassifier(random_state=42)` -- before settling on a final depth, try `max_depth` values of `3`, `5`, `10`, and `None` (unlimited). For each, print both the training accuracy and the test accuracy. What do you notice as depth increases? What does that tell you about overfitting? Pick the depth you would use in production and add a comment explaining your reasoning. Then, using your chosen depth, print the accuracy and full classification report as you did for the other classifiers.
- `RandomForestClassifier` (introduced below)
- `LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')`

After you have results for all five, write a comment summarizing what you see. Which model performs best? For a spam filter specifically, is accuracy the right metric to optimize -- or would you rather minimize false positives (legitimate email marked as spam) or false negatives (spam that gets through)? Take a position and defend it.

For your best-performing classifier, create a confusion matrix using `ConfusionMatrixDisplay` and save it to `outputs/best_model_confusion_matrix.png`. Given the costs described above, which type of error does your best model make more often?

### A note on Decision Trees and Random Forests

For a visual introduction, this [StatQuest video (~8 min)](https://www.youtube.com/watch?v=7VeUPuFGJHk) is worth watching before or after you work through this section.

Think about how you personally decide whether an email is spam:

> "Does it use words like *free* or *winner*?"
> "Are there long blocks of capital letters?"
> "Does it contain lots of dollar signs?"

This is exactly how a decision tree works -- a sequence of yes/no questions, where each answer leads to the next question, until the tree reaches a prediction. Unlike KNN, which measures distances between data points, a tree examines one feature at a time and asks whether it crosses a threshold. This means trees do not need feature scaling and they produce rules that are straightforward to inspect.

The tree learns these questions from data. At each step, it tries many possible splits and picks the one that creates the most predictive split -- where most examples belong to the same class. The measure of purity is called *Gini impurity*: high when a group is evenly mixed (the tree is uncertain), low when one class dominates (the tree is confident). Before a good split, a group might be evenly mixed -- Gini impurity is high:

```text
Before split (high Gini impurity -- very uncertain):
Spam:     ██████████
Not spam: ██████████
```

After a well-chosen split (say, on whether `char_freq_$` is high), the two groups become much more certain -- Gini impurity drops:

```text
After splitting on char_freq_$ (low Gini impurity -- much more certain):
Group A ($ is rare):
Spam: ██  Not spam: ████████████████
Group B ($ is frequent):
Spam: ████████████████  Not spam: ██
```

The tree keeps splitting until the groups are pure enough -- or until it reaches its *depth limit*, meaning the maximum number of questions it is allowed to ask before it must make a prediction.

A tree with no limit will keep splitting until every training example has its own leaf, memorizing the training data rather than learning general patterns. You will see this overfitting directly when you compare train and test accuracy across different `max_depth` values above.

Even with a depth limit, a single tree is still fragile -- small changes in the training data can produce a very different tree. A *Random Forest* addresses this in a way that echoes cross-validation: instead of one tree trained on the full dataset, it trains hundreds of trees, each on a different random sample of the training examples and a random subset of features. Just as cross-validation averages results across multiple data splits to get a more reliable estimate, a Random Forest averages predictions across many trees to get a more reliable answer -- that diversity is baked into the model itself. No single tree is authoritative, but the crowd as a whole tends to get it right.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

Random forests are one of the most practically useful classifiers in the scikit-learn toolkit -- they have won countless Kaggle competitions and remain a go-to choice for tabular data in production. Training 100 trees takes noticeably longer than training one, so expect a longer wait than you saw with the Decision Tree. They also tend to generalize well without needing extensive cross-validation -- the internal averaging across many trees already provides a form of stability.

Both the Decision Tree and the Random Forest expose a `.feature_importances_` attribute. After building both, print the top 10 most important features for each and save a bar chart of the Random Forest importances to `outputs/feature_importances.png`. Do the two models agree on which features matter most? Do the results match your intuition about what makes an email spam?

In Task 4 you will cross-validate all your models -- the variance across folds for the Random Forest should be noticeably lower than for the Decision Tree.

## Task 4: Cross-Validation

A single train/test split can give you a misleading picture -- you might have gotten lucky (or unlucky) with how the data was divided. Cross-validation gives a more reliable estimate of how well a model generalizes to unseen data.

Using `cross_val_score` with `cv=5`, run cross-validation on the training data for each of your classifiers from Task 3. For each, print the mean and standard deviation of the fold scores. Which model is the most accurate? Which is the most stable (lowest variance across folds)? Does the ranking match what you saw with the single train/test split?

## Task 5: Building a Prediction Pipeline

### Intro to scikit-learn pipelines

So far you have been managing preprocessing manually: fit the scaler on training data, transform training data, transform test data, then pass the results to a classifier. This works, but it requires careful bookkeeping -- it is easy to forget a step, apply transformations in the wrong order, or accidentally leak information from the test set into the scaler.

In the [Week 1 pipelines lesson](../../lessons/01_analysis_intro/07_pipelines.md) you learned that a pipeline is a series of connected steps where the output of one becomes the input of the next. scikit-learn's `Pipeline` class brings this idea directly into model building for ML.

A pipeline is defined as a list of *named steps*, where each step is a `("name", object)` pair. The name is a label you choose; the object is any sklearn transformer or model. Steps run in the order they are listed.

For example:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

knn5_pipeline = Pipeline([
    ("scaler",     StandardScaler()), # name, object pattern
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])
```

Once built, a pipeline behaves like any other sklearn model. When you call `knn5_pipeline.fit(X_train, y_train)`, it fits the scaler on the training data and passes the scaled result to the classifier. When you call `knn5_pipeline.predict(X_test)`, it applies the same scaling -- learned from training data only -- to the test data before predicting. No extra steps required.

```python
knn5_pipeline.fit(X_train, y_train)
y_pred = knn5_pipeline.predict(X_test)
```

In scikit-learn, evaluation is typically handled outside the pipeline. The pipeline's job is preprocessing and prediction; what you do with those predictions -- `classification_report`, `accuracy_score`, `ConfusionMatrixDisplay` -- is up to you and happens after `predict()`.

One exception is that a pipeline does have a built-in method for calculating accuracy: `pipeline.score(X_test, y_test)`. `score()` does quite a bit in one call: it runs `X_test` through the pipeline, generates predictions, and compares them against `y_test`, returning accuracy.

### Build your pipelines

Build two pipelines: one for your best tree-based classifier and one for your best non-tree-based classifier. For each, fit on the training data and print the full classification report on the test set. Confirm the results match your earlier manual approach.

Comment on your results: do the two pipelines have the same structure? Why or why not? What is the practical value of packaging a model this way, especially when handing it off to someone else or deploying it?

Good luck on this mini-project!
