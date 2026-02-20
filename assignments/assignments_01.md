## Submission instructions
In your `python200-homework` repository, create a folder called `assignments_01/`. Inside that folder, create three files:

1. warmup_01.py  : for the warmup exercises
2. prefect_warmup.py  : for the prefect pipeline warmup exercise
3. project_01.py  : for the project exercise

When finished, commit the files to your repo and open a PR as discussed in the assignment's [README page](README.md). 


## Part 1: Warmup Exercises
Put all warmup exercises  in a single file: `warmups_01.py`. Use comments to mark each section and question (e.g. `# --- Pandas ---` and `# Pandas Q1`). Use `print()` to display all outputs. 

### Pandas Review

#### Pandas Question 1

Create the following DataFrame and print the first three rows, the shape, and the data types of each column.

```python
import pandas as pd

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)
```

Print each result with a label (e.g. `print(f"Num Rows: {len(df)}")`).

#### Pandas Question 2

Using the DataFrame from Q1, filter the rows to show only students who passed *and* have a grade above 80. Print the result.

#### Pandas Question 3

Add a new column called `"grade_curved"` that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows).

#### Pandas Question 4

Add a new column called `"name_upper"` that contains each student's name in uppercase, using the `.str` accessor. Print the `"name"` and `"name_upper"` columns together.

#### Pandas Question 5

Group the DataFrame by `"city"` and compute the mean grade for each city. Print the result.

#### Pandas Question 6

Replace the value `"Austin"` in the `"city"` column with `"Houston"`. Print the `"name"` and `"city"` columns to confirm the change.

#### Pandas Question 7

Sort the DataFrame by `"grade"` in descending order and print the top 3 rows.

### NumPy Review

#### NumPy Question 1

Create a 1D NumPy array from the list `[10, 20, 30, 40, 50]`. Print its shape, dtype, and ndim.

#### NumPy Question 2

Create the following 2D array and print its shape and size (total number of elements).

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
```

#### NumPy Question 3

Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is `[[1, 2], [4, 5]]`.

#### NumPy Question 4

Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both.

#### NumPy Question 5

Create an array using `np.arange(0, 50, 5)`. First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation.

#### NumPy Question 6

Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use `np.random.normal()`). Print the mean and standard deviation of the result.

### Matplotlib Review

#### Matplotlib Question 1

Plot the following data as a line plot. Add a title `"Squares"`, x-axis label `"x"`, and y-axis label `"y"`.

```python
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
```

#### Matplotlib Question 2

Create a bar plot for the following subject scores. Add a title `"Subject Scores"` and label both axes.

```python
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
```

#### Matplotlib Question 3

Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.

```python
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
```

#### Matplotlib Question 4

Use `plt.subplots()` to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot `x` vs `y` from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call `plt.tight_layout()` before showing.

### Descriptive Statistics Review

#### Descriptive Stats Question 1

Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.

```python
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
```

#### Descriptive Stats Question 2

Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use `np.random.normal(65, 10, 500)`). Plot a histogram with 20 bins. Add a title `"Distribution of Scores"` and label both axes.

#### Descriptive Stats Question 3

Create a boxplot comparing the two groups below. Label each box (`"Group A"` and `"Group B"`) and add a title `"Score Comparison"`.

```python
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
```

Hint: pass `labels=["Group A", "Group B"]` to `plt.boxplot()`.

#### Descriptive Stats Question 4

Create the following DataFrame and use `groupby()` to compute the mean and standard deviation of `"salary"` for each `"department"`. Print the result.

```python
employees = {
    "department": ["Engineering", "Marketing", "Engineering", "Marketing", "Engineering"],
    "salary":     [95000, 72000, 105000, 68000, 89000]
}
df = pd.DataFrame(employees)
```

#### Descriptive Stats Question 5
Print the mean, median, and mode of the following:

data1 = [10, 12, 12, 16, 18]  
data2 = [10, 12, 12, 16, 150]

Why are the median and mean so different for data2? Add your answer as a comment in the code.


### Hypothesis Testing Review

#### Hypothesis Question 1

Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.

```python
from scipy import stats

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
```

#### Hypothesis Question 2

Using the p-value from Q1, write an `if/else` statement that prints whether the result is statistically significant at alpha = 0.05.

#### Hypothesis Question 3

Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.

```python
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
```

#### Hypothesis Question 4

Run a one-sample t-test to check whether the mean of `scores` is significantly different from a national benchmark of 70. Print the t-statistic and p-value.

```python
scores = [72, 68, 75, 70, 69, 74, 71, 73]
```

#### Hypothesis Question 5

Re-run the test from Q1 as a one-tailed test to check whether `group_a` scores are *less than* `group_b` scores. Print the resulting p-value. Use the `alternative` parameter.

#### Hypothesis Question 6

Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a `print()` statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.

### Correlation Review

#### Correlation Question 1

Compute the Pearson correlation between `x` and `y` below using `np.corrcoef()`. Print the full correlation matrix, then print just the correlation coefficient (the value at position `[0, 1]`).

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
```

What do you expect the correlation to be, and why? Add your answer as a comment in the code.

#### Correlation Question 2

Use `pearsonr()` from `scipy.stats` to compute the correlation between `x` and `y` below. Print both the correlation coefficient and the p-value.

```python
from scipy.stats import pearsonr

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
```

#### Correlation Question 3

Create the following DataFrame and use `df.corr()` to compute the correlation matrix. Print the result.

```python
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
```

#### Correlation Question 4

Create a scatter plot of `x` and `y` below, which have a negative relationship. Add a title `"Negative Correlation"` and label both axes.

```python
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
```

#### Correlation Question 5

Using the correlation matrix from Q3, create a heatmap with `sns.heatmap()`. Pass `annot=True` so the correlation values appear in each cell, and add a title `"Correlation Heatmap"`.

Hint:
```python
import seaborn as sns
```

### Pipelines

#### Pipeline Question 1

A data pipeline is a sequence of processing steps where each step takes in data, transforms it, and passes the result to the next. You don't need a special framework to build one -- chaining plain functions together is often enough.

Given the array below, which contains some missing values scattered throughout:

```python
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
```

Implement the following three functions and then connect them in a `data_pipeline()` function.

- `create_series(arr)` : takes a NumPy array and returns a pandas Series with the name `"values"`.
- `clean_data(series)` : takes the Series, removes any `NaN` values using `.dropna()`, and returns the cleaned Series.
- `summarize_data(series)` -- takes the cleaned Series and returns a dictionary with four keys: `"mean"`, `"median"`, `"std"`, and `"mode"`. For mode, use `series.mode()[0]` to get a single value.
- `data_pipeline(arr)` -- calls the three functions above in sequence and returns the summary dictionary.

Call `data_pipeline(arr)` and print each key and its value from the result.

This is the last answer to put in `warmups_01.py`. Congrats!!!

The next question will be in `prefect_warmup.py`, but will implement the same functionality using Prefect instead of plain Python.

#### Pipeline Question 2

The answer to this question should go in `prefect_warmup.py`, not `warmups_01.py`.

Rebuild the pipeline from Q1 using Prefect. Copy your three functions from Pipeline Question 1 (`create_series`, `clean_data`, `summarize_data`) into this file and turn them into Prefect tasks using `@task`.

Turn `data_pipeline()` into a Prefect flow using `@flow`. Inside the flow, call the three tasks in order and return the summary dictionary.

Add this block at the bottom of the file so the flow runs when you execute the script directly:

```python
if __name__ == "__main__":
    pipeline_flow()
```

Run your workflow from the terminal:

```bash
python prefect_warmup.py
```

The summary values should match what you got in Question 1.

Finally, add a comment block at the bottom of `prefect_warmup.py` answering these two questions:

1. This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?
2. Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.


