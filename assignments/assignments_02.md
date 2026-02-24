# Week 2 Assignments

This week's assignments cover the week 2 material, including:

- The scikit-learn API
- Linear regression: fitting, evaluating, and interpreting models
- Train/test splits and generalization
- Multiple regression with numeric and binary features

As with week 1, the warmup exercises are meant to build muscle memory for the core mechanics -- try to work through them without AI assistance. The mini-project will ask you to apply these tools in a more open-ended context.

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_02/`. Inside that folder, create two files and an outputs directory:

1. `warmup_02.py`  : for the warmup exercises
2. `project_02.py` : for the mini-project
3. `outputs/`      : for any plots or data files your code generates

When finished, commit and open a PR as described in the [assignments README](README.md).

# Part 1: Warmup Exercises

Put all warmup exercises in a single file: `warmup_02.py`. Use comments to mark each section and question (e.g. `# --- scikit-learn API ---` and `# Q1`). Use `print()` to display all outputs.

## The scikit-learn API

### scikit-learn Question 1

The core pattern in scikit-learn is `create → fit → predict`. Practice it here with a simple dataset: years of work experience versus annual salary.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
```

Create a `LinearRegression` model, fit it to this data, and then predict the salary for someone with 4 years of experience and someone with 8 years. Print the slope (`model.coef_[0]`), the intercept (`model.intercept_`), and the two predictions. Label each printed value.

### scikit-learn Question 2

scikit-learn requires the feature array `X` to be 2D even when you only have one feature. Start with this 1D array:

```python
x = np.array([10, 20, 30, 40, 50])
```

Print its shape. Use `.reshape()` to convert it to a 2D array and print the new shape. Add a comment explaining, in your own words, why scikit-learn needs `X` to be 2D.

### scikit-learn Question 3

K-Means is an unsupervised algorithm that follows the same `create → fit → predict` pattern as everything else in scikit-learn. Use the code below to generate a synthetic dataset with three natural clusters:

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
```

Create a `KMeans` model with `n_clusters=3` and `random_state=42`, fit it to `X_clusters`, and predict a cluster label for each point. Print the cluster centers (`kmeans.cluster_centers_`) and how many points fell into each cluster using `np.bincount(labels)`.

Then create a scatter plot coloring each point by its cluster label, plot the cluster centers as black X's, add a title and axis labels. Save the figure to `outputs/kmeans_clusters.png`.

## Linear Regression

The questions below all use the same synthetic medical costs dataset: 100 patients, each with an age (20 to 65), a smoker flag (0 = non-smoker, 1 = smoker), and an annual medical cost as the target. Generate it once and reuse the variables throughout.

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)
```

### Linear Regression Question 1

Before fitting anything, look at the data. Create a scatter plot of `age` on the x-axis and `cost` on the y-axis. Color the points by smoker status by passing `c=smoker` and `cmap="coolwarm"` to `plt.scatter()`. Add a title `"Medical Cost vs Age"`, label both axes, and save to `outputs/cost_vs_age.png`.

Add a comment describing what you see. Are there two distinct groups visible? What does that suggest about the `smoker` variable?

### Linear Regression Question 2

Split the data into training and test sets using `age` as the only feature, an 80/20 split, and `random_state=42`. Reshape `age` to a 2D array before using it as `X`. Print the shapes of all four arrays.

### Linear Regression Question 3

Fit a `LinearRegression` model to your training data from Question 2. Print the slope and intercept. Then predict on the test set and print:

- RMSE: `np.sqrt(np.mean((y_pred - y_test) ** 2))`
- R² on the test set: `model.score(X_test, y_test)`

Add a comment interpreting the slope in plain English -- what does it mean for medical costs?

### Linear Regression Question 4

Now add `smoker` as a second feature and fit a new model.

```python
X_full = np.column_stack([age, smoker])
```

Split, fit, and print the test R². Compare it to the R² from Question 3 -- does adding the smoker flag help? Print both coefficients:

```python
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
```

Add a comment interpreting the `smoker` coefficient: what does it represent in practical terms?

### Linear Regression Question 5

A *predicted vs actual plot* is a standard tool for evaluating regression models. Each test observation becomes a dot: the model's prediction goes on the x-axis, the true value goes on the y-axis. A perfect model would place every point on the diagonal line where predicted equals actual.

Using the two-feature model from Linear Regression Question 4, create this plot for the test set. Add a diagonal reference line, a title `"Predicted vs Actual"`, labeled axes, and save to `outputs/predicted_vs_actual.png`.

Add a comment: what does it mean when a point falls above the diagonal? What about below?

# Part 2: Mini-Project -- Predicting Student Math Performance

This project uses the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance), collected from two Portuguese secondary schools during the 2005-2006 school year. The dataset records demographic, social, and academic attributes for students enrolled in a math course, along with their grades at three points in the year: G1 (first period), G2 (second period), and G3 (final grade, 0-20).

Your goal is to build a regression model that predicts G3 from student background and behavioral features -- without using G1 or G2. Predicting a final grade from earlier grades in the same class is a narrow task; predicting it from who students are and how they live is a much more interesting problem, and the kind of thing a real analyst or data engineer would be asked to do.

The data file `student_performance_math.csv` is in the course repository at `assignments/resources/`. Copy it into your `assignments_02/` folder before starting, and place your code in `project_02.py`.

## Pre-preprocessing

Before writing a single line of code, open `student_performance_math.csv` in a plain text editor (not Excel). Read the first few rows carefully.

Notice how fields are separated. Notice which values are quoted and which are not. Look at what G1, G2, and G3 look like in the raw file. If you were loading this with `pd.read_csv()`, what parameter would you need to specify beyond the filename? Write that observation as a comment at the top of your script before you write the load call.

## Feature Guide

The version of the dataset used here has been trimmed to 18 columns (the original has 33). The tables below describe everything included. Read through them before you start coding.

*Numeric features* -- use directly as numbers:

| Column | Description |
|---|---|
| `age` | Student age (15-22) |
| `Medu` | Mother's education: 0=none, 1=primary (4th grade), 2=5th-9th grade, 3=secondary, 4=higher education |
| `Fedu` | Father's education (same 0-4 scale as Medu) |
| `traveltime` | Home-to-school travel time: 1=under 15 min, 2=15-30 min, 3=30-60 min, 4=over 1 hour |
| `studytime` | Weekly study time: 1=under 2 hours, 2=2-5 hours, 3=5-10 hours, 4=over 10 hours |
| `failures` | Number of past class failures (0-3; values above 3 are coded as 3) |
| `absences` | Number of school absences (0-93) |
| `freetime` | Free time after school (1=very low to 5=very high) |
| `goout` | Time going out with friends (1=very low to 5=very high) |
| `Walc` | Weekend alcohol consumption (1=very low to 5=very high) |

*Binary features stored as "yes"/"no"* -- you will convert these to 1/0:

| Column | Description |
|---|---|
| `schoolsup` | Extra educational support from the school (remedial help, tutoring) |
| `internet` | Has internet access at home |
| `higher` | Wants to pursue higher education after secondary school |
| `activities` | Participates in extra-curricular activities |

*Binary feature stored as "F"/"M"* -- you will convert to 0/1:

| Column | Description |
|---|---|
| `sex` | Student sex (F=0, M=1). In this Portuguese dataset from 2005, male students show a modest math advantage. [PISA research](https://www.oecd.org/pisa/keyfindings/pisa-2012-results-gender.htm) shows this gap varies significantly by country and correlates with gender equality -- suggesting it reflects a social pattern in the educational context, not an inherent difference. The coefficient is worth noticing and discussing. |

*Grade columns:*

| Column | Description |
|---|---|
| `G1` | First period grade (0-20) |
| `G2` | Second period grade (0-20) |
| `G3` | Final period grade (0-20) -- your prediction target. Note: Some students have G3=0. This represents absence from the final exam, not an actual score of zero. You will need to decide how to handle these rows. |

G1 and G2 are in the dataset but are not used as features in the main tasks below.

## Task 1: Load and Explore

Load the dataset with the correct separator. Print the shape, the first five rows, and the data types of all columns.

Then plot a histogram of G3 with 21 bins (one per possible value, 0-20). Add a title `"Distribution of Final Math Grades"`, label both axes, and save to `outputs/g3_distribution.png`. You should see a cluster of zeros sitting apart from the main distribution. They represent the students who didn't take the final exam. 

## Task 2: Preprocess the Data

Handle the G3=0 rows first. Filter them out and save the result to a new DataFrame. Print the shape before and after to confirm how many rows were removed. Add a comment explaining your reasoning -- why would keeping these rows distort the model?

Then convert the yes/no columns to 1/0 and the sex column to 0/1.

Now check something interesting before moving on. Compute the Pearson correlation between `absences` and G3 on both the original dataset and the filtered one, and print both values. The difference is striking. Add a comment explaining why filtering changes the result: what were students with G3=0 doing in the original data that made `absences` look like a weak predictor? You might want to explore scatter plots to help understand this.

## Task 3: Exploratory Data Analysis

Compute the Pearson correlation between each numeric feature and G3 on the filtered dataset, and print them sorted from most negative to most positive. Which feature has the strongest relationship with G3? Are any results surprising?

Then create at least two visualizations of your own choosing and save them to `outputs/`. Use your judgment from previous weeks of data engineering to guide your use of plots. Use the correlation results to guide you -- what relationships seem worth a closer look? Add a comment for each plot describing what you see.

## Task 4: Baseline Model

Build the simplest possible model: use `failures` alone to predict G3. Split into training and test sets (80/20, `random_state=42`), fit a `LinearRegression` model, and print the slope, RMSE, and R² on the test set.

Add a comment: given that grades are on a 0-20 scale, what do the slopes and RMSE tell you in plain English? Is R² better or worse than you expected from exploratory data analysis?

## Task 5: Build the Full Model

Now build a regression model using all of the numeric and binary features from the Feature Guide:

```python
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_clean[feature_cols].values
y = df_clean["G3"].values
```

Split into training and test sets (80/20, `random_state=42`), fit a `LinearRegression` model, and print both train R² and test R², as well as RMSE on the test set. Compare the test R² to your baseline from Task 4 -- how much does adding more features help?

Print each feature name alongside its coefficient:

```python
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")
```

Look carefully at the coefficients. Sort them mentally from largest to smallest. Are any signs (positive or negative) surprising given what you know about the data? For any surprising result, add a comment with your best explanation. Then compare train R² to test R² -- are they close, or is there a gap? What does that tell you about the model?

Finally, add a comment answering: if you were deploying this model in production, which features would you keep and which would you drop? Justify your choices based on what you see in the numbers.

## Task 6: Evaluate and Summarize

A useful way to evaluate a regression model visually is a *predicted vs actual plot*. This is a scatter plot where each point in the test set becomes a dot, with the model's *prediction* (y_hat) on the x-axis and the true value (y) on the y-axis. If the model were perfect, every point would fall exactly on the diagonal (predicted = actual). Clusters or curves away from the diagonal reveal systematic errors that RMSE alone won't show you. Random scattering around the diagonal is expected, and acceptable, prediction error. 

Create this plot for your test set. Add a diagonal reference line (for y=y_predicted), a title `"Predicted vs Actual (Full Model)"`, labeled axes, and save to `outputs/predicted_vs_actual.png`. Add a comment: does the model seem to struggle more at the high end, the low end, or is error roughly uniform across grade levels? What does a value above or below the diagonal mean?

Then write a plain-language summary in your comments statements covering:

- The size of the filtered dataset and the test set
- The RMSE and R² of your best model in plain language -- on a 0-20 scale, what does a typical prediction error actually mean?
- Which two features have the largest positive and largest negative coefficients, and what those mean
- One result that surprised you

## Neglected Feature: The Power of G1

Add `G1` (first period grade) as a feature to the full model from Task 5 and refit. We kept it out because it is so powerful. Print the new test R². The jump will be large -- from roughly 0.30 to somewhere around 0.80.

Add a comment addressing these questions: does a high R² here mean G1 is *causing* G3? Is this a useful model for identifying students who might struggle? What might educators need to do if they wanted to intervene early, before G1 is even available?

Good luck on this mini-regression analysis project!
