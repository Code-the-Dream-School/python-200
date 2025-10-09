
# Lesson 1 - Topic 1

## Recapping and Adding to What We Learned in Python 100

### Prerequisites
- Operators  
- Data Types  
- Loops  
- Functions  
- File types (.CSV, .Excel)

> 💡 All the code examples and "check for understanding" in this section can (and should!) be run by you in VS Code.  
> Try them out and see the output for yourself, it’s the best way to learn!


### Topics Covered
- **Pandas** (load, clean, and analyze data)  
- **Numpy** (complex numerical operations)  
- **Matplotlib** (visualization or plotting the data on graphs)  

---

## Pandas

Pandas is a Python library that makes it easy to work with structured data, like tables you’d see in Excel or a database. It helps you store, clean, analyze, and explore data quickly and efficiently.

There is a ton of data being generated and processed every second and most real-world data comes in messy formats, missing values, weird columns, extra info, or in files like CSVs or Excel spreadsheets.

**Pandas helps you:**
- Load that data into Python.  
- Clean it up.  
- Ask useful questions like:  
  - “What’s the average income in this dataset?”  
  - “How many people signed up last week?”  
  - “Which product had the most sales?”  

It’s used in data analytics, machine learning, finance, science, and pretty much anywhere people work with data.

---

## Getting Started with Pandas with some important resources
📺 [Learning Pandas for Data Analysis?](https://www.youtube.com/watch?v=DkjCaAMBGWM)  
📺 [What is Pandas?](https://www.youtube.com/watch?v=dcqPhpY7tWk&t=70s)
📺 [10 Minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) — official quickstart guide from the pandas developers

## Pro-tip: 
- You’ll definitely run into errors sometimes, that’s normal! Just read the error message on your screen carefully and search online. Errors are your best hints for fixing problems.

---

### What is a DataFrame and a Series?

**DataFrame**  
You can think of a DataFrame as a two-dimensional table or a spreadsheet in Python.  
- It has rows and columns.  
- Each column has a name.  
- Each row has an index number.  

<img width="644" height="344" alt="Screenshot 2025-09-04 at 6 39 39 PM" src="https://github.com/user-attachments/assets/db1dc11e-9eb2-40cb-9eef-e4947925b372" />


Rule of Thumb: a DataFrame has two axes, just like a table.  
- **Rows** (top to bottom) = Axis 0  
- **Columns** (left to right) = Axis 1  

👉 More on DataFrames: [GeeksforGeeks Pandas DataFrame](https://www.geeksforgeeks.org/pandas/python-pandas-dataframe/)  

**Series**  
You can think of a Series as just one column from a DataFrame.  
- It has values and an index, but only one column.  
- Can store heterogeneous data types.  
- A Series is like a fancy list that knows the name of each item.

<img width="640" height="300" alt="Screenshot 2025-09-04 at 6 40 01 PM" src="https://github.com/user-attachments/assets/a8d01a5b-0cfa-43fc-a21e-dfad2f04b4ac" />

<img width="640" height="300" alt="Screenshot 2025-09-04 at 6 40 17 PM" src="https://github.com/user-attachments/assets/b8b9a6e0-cca6-4e71-bb23-b91953cf6c99" />

👉 More on Series: [GeeksforGeeks Pandas Series](https://www.geeksforgeeks.org/python/python-pandas-series/)  

---

## Installing and Importing Pandas

```bash
pip install pandas
```

Once installed, import it into your Python code:

```python
import pandas as pd
```

---

## Reading from a CSV File

**What is a CSV file?**  
CSV = Comma-Separated Values. It’s a plain text file that looks like a spreadsheet, where each row is a line and columns are separated by commas.

```python
# Read data from a CSV file
df = pd.read_csv('customers_100.csv')
```

---

## Creating a DataFrame

You can create DataFrames in several ways:

**From a Dictionary (most common):**  
```python
data = {
    "name": ["Alice", "Bob", "Carol"],
    "age": [25, 30, 22],
    "grade": ["A", "B", "A"]
}
df = pd.DataFrame(data)
print(df)
```

**From a List of Dictionaries:**  
```python
students = [
    {"name": "Alice", "age": 25, "grade": "A"},
    {"name": "Bob", "age": 30, "grade": "B"}
]
df = pd.DataFrame(students)
print(df)
```

**From a List of Lists (with column names):**  
```python
data = [
    ["Alice", 25, "A"],
    ["Bob", 30, "B"],
    ["Carol", 22, "A"]
]
df = pd.DataFrame(data, columns=["name", "age", "grade"])
print(df)
```

- Now, Read a Series (pick a column)
```python
ages = df["age"]   # this is a Series
print(ages)
```

---

## Creating a Series

**From a Python list:**  
```python
numbers = [10, 20, 30, 40]
num_series = pd.Series(numbers)
print(num_series)
```

**From a list with custom labels:**  
```python
data = [85, 90, 95]
names = ["Alice", "Bob", "Carol"]
my_series = pd.Series(data, index=names)
print(my_series)
```

**From a Dictionary:**  
```python
data = {"Alice": 85, "Bob": 90, "Carol": 95}
my_dict_series = pd.Series(data)
print(my_dict_series)
```

---

## Reading a Series from a DataFrame

A **Series** in Pandas represents a single column of data. To understand this better, let’s first create a simple DataFrame and then select a column from it.

### Step 1: Create a DataFrame

```python
import pandas as pd

data = {
    "name": ["Alice", "Bob", "Carol"],
    "age": [25, 30, 22]
}
df = pd.DataFrame(data)
print(df)
```

- Now, Read a Series (pick a column)
```python
ages = df["age"]   # this is a Series
print(ages)
```

---


## Why Series are Important

- A DataFrame is really just a collection of Series.  
- Series can have **labels** (indexes) which make it easy to access values.  
- Series handle **missing data** with NaN gracefully.  
- Let’s test this concept
```python
s = pd.Series([85, 90, 95], index=["Alice", "Bob", "Carol"])
print(s["Bob"])  # Output: 90
```
- Another important reason, series handles missing data effortlessly. Missing Data = gaps or no values or None values
- Pandas allows NaN inside a Series. Instead of crashing or giving wrong results, it understands that the value is missing.
- Let’s test
```python
s = pd.Series([10, None, 30])
print(s)  # Handles missing values
```

---

## Exploring Your Data in Pandas

When you load a dataset, you might want to get a quick look at it. Pandas provides:  
- `.head()` → Peek at first few rows  
- `.tail()` → Peek at last few rows  
- `.info()` → Structure, datatypes, missing values  
- `.describe()` → Quick statistics summary  
- `.value_counts()` → Frequency of unique values in a column (or series)

Example with `students.csv`:Let’s Load the data using-import pandas as pd 
- optional if you already have imported pandas {students_df = pd.read_csv("students.csv")}
- Here’s what the dataset looks like:
<img width="653" height="323" alt="Screenshot 2025-09-04 at 6 46 31 PM" src="https://github.com/user-attachments/assets/4f17a569-dfd4-4314-9a79-3b126d059b39" />


- .head(): Take a Peek
Think of .head() as saying:
“Show me just the first few rows so I can see what’s inside.” By default, .head() shows the first 5 rows of your dataset. You can also peek at more rows: df.head(10) for 10 rows.

Let’s try: 
```python
print(students_df.head(2)) #prints first 2 rows
```

- .tail(): Look at the End. If .head() is a sneak peek at the start, then .tail() is a sneak peek at the end.
- By default, .tail() shows the last 5 rows of your dataset.
- Let’s try:
```python  
print(students_df.tail()) #prints last 5 rows
```

- .info():  Get the Blueprint. It is like asking Pandas: “Tell me the structure of this dataset.”
- Let’s try:
```python
print(students_df.info())
```
- When you run df.info(), Pandas gives you a summary report of your DataFrame.
<img width="475" height="348" alt="Screenshot 2025-09-04 at 7 18 57 PM" src="https://github.com/user-attachments/assets/03ea7d7c-a120-43b4-ba0d-c811c52e32e5" />

- Understanding .info() Output
- RangeIndex: The dataset has 6 rows (0 to 5).
- Data columns:  The dataset has 6 columns.
- Columns: id, name, age, grade, city, score.
- Missing values: Some columns have missing values: age, grade, city, score.
- Data types: Numbers (int, float) and text (object).
- .describe(): Quick Statistics
- .describe() gives you a quick summary of the numbers in your dataset.

- Let’s try: print(students_df.describe())

**- Gives us:**
<img width="380" height="205" alt="Screenshot 2025-09-04 at 7 19 28 PM" src="https://github.com/user-attachments/assets/81977e12-5865-48d5-ae2e-05887cf7fd43" />

- What `.describe()` tells us:
  - count: how many values (notice 5, not 6, because of missing data)
  - mean: the average
  - std: standard deviation (how spread out the values are)
  - min & max: smallest and biggest
  - 25%, 50%, 75%: quartiles (like checkpoints in the data)

So .describe() is like a quick health report of your numbers.


- Quick tip
Usually the experts suggest these three are always the first commands you should run after loading any dataset, so that you get a good knowledge of your data you will be working with.

---

## Check for Understanding 🎯

You have the dataset **students.csv** loaded into a DataFrame called `students_df`.  

**Q1. What’s the difference between a DataFrame and a Series?**  
<details>
<summary>Show Answer</summary>

A DataFrame is a 2D table (rows × columns). A Series is a single column of data.
</details>

**Q2. What is the role of the index in a DataFrame?**  
<details>
<summary>Show Answer</summary>

The index labels each row. It helps identify and access data but is not itself a data column.
</details>

**Q3. After creating a new DataFrame, what are the first steps you should take to understand its contents?**  
<details>
<summary>Show Answer</summary>

Use `.head()`, `.info()`, `.describe()`, and possibly `.value_counts()` for key columns.
</details>

**Q4. Which columns have missing values in `students_df`?**  
<details>
<summary>Show Answer</summary>

age, grade, city, score
</details>

**Q5 (Bonus). How many students are missing their score?**  
<details>
<summary>Show Answer</summary>

1 student
</details>

---

# Lesson 1 - Topic 2

## Selecting & Filtering Data

### Selecting Data with `.loc[]` and `.iloc[]`

Datasets can be huge, with millions of rows and many columns.  
To work effectively, we often need to grab **specific pieces** of data: rows, columns, or even individual cells.

Both `.loc[]` (label-based) and `.iloc[]` (index-based) are powerful tools for this.  
Think of them like coordinates:

- `.loc[row_label, column_label]`
- `.iloc[row_number, column_number]`

---

#### `.loc[]` — label-based selection  
- Use when your dataset has **meaningful labels** (names, IDs, dates, etc.).  

**Example:**

```python
# Optional: set the index to 'name' for label-based selection
students_df = students_df.set_index("name")

# Row-only: get the row for Carol
students_df.loc["Carol"]

# Row + column: get Carol’s score
students_df.loc["Carol", "score"]

# Column-only: get the 'score' column for all students
students_df.loc[:, "score"]
```

---

#### **`.iloc[]` — index-based selection**  
- Think **i for index number**.  
- Uses integer positions instead of labels.

**Example:**

```python
# Row-only: get the 3rd row (Carol, if index is reset)
students_df.iloc[2]

# Row + column: get the value at row 3, column 2
students_df.iloc[2, 1]

# Column-only: get the 2nd column for all rows
students_df.iloc[:, 1]
```

---

### Filtering Rows by Condition

Filtering = ask a question and get matching rows.  
This helps us find rows based on conditions.  

**Real analysis questions:**  
- “Which students scored above 90?”  
- “Who is younger than 25?”  

**Example: Students older than 30**  
```python
print(students_df[students_df["age"] > 30])
```

** Example: Students with grade “A”**  
```python
print(students_df[students_df["grade"] == "A"])
```

---

### Sorting Data

We can sort rows by any column. Sorting helps you see patterns.  

** Example:  
- Sort students by **score** to see the top performers.  
- Sort by **age** to see youngest/oldest.  

** Example Sort by score (descending):**  
```python
students_df.sort_values("score", ascending=False)
```

---

## Check for Understanding: Selecting & Filtering Data 🎯

**Q1. How do you select just the `name` column from `students_df`?**  
<details>
<summary>Show Answer</summary>

```python
students_df["name"]
```
</details>

**Q2. How do you select both `name` and `age` columns?**  
<details>
<summary>Show Answer</summary>

```python
students_df[["name", "age"]]
```
</details>

**Q3. How do you select the first row using `.iloc`?**  
<details>
<summary>Show Answer</summary>

```python
students_df.iloc[0]
```
</details>

**Q4. How do you select the third row using `.loc` (assuming labels)?**  
<details>
<summary>Show Answer</summary>

```python
students_df.loc[2]
```
</details>

**Q5. Show the rows where age is greater than 30.**  
<details>
<summary>Show Answer</summary>

```python
students_df[students_df["age"] > 30]
```
</details>

---

✅ This wraps up **Part 2 of Lesson 1**.


# Lesson 1 - Topic 3

## Cleaning Data  

Cleaning data is like tidying your desk before starting homework. 📚  
If your data is messy, your analysis will give wrong or unexpected answers. That’s why data cleaning is **super important**.  

This step makes sure no row is left “half-empty” when you do math or analysis.  

We’ll cover 4 big steps:  

---

### 1. Checking for and Filling Missing Values  

Why? Real-world data is never perfect, sometimes values are missing. Pandas helps us spot and fix them.  

**Check missing values:**  
```python
print(students_df.isnull())
```


👉 Output: A table of `True`/`False`.  
- `True` = value is missing (NaN).  
- `False` = value is present.  

**Count missing values in each column:**  
```python
print(students_df.isnull().sum())
```

👉 Easier to read than the True/False table.  

**Fix missing values with `.fillna()`:**  
```python
students_df["age"] = students_df["age"].fillna(students_df["age"].mean())
print(students_df.isnull())
```

Before: row 3 had `True` (missing age).  
After filling: row 3 now shows `False` (no missing value).  

---

### 2. Renaming Columns  

Why rename columns?  
- Sometimes column names are too long, unclear, or contain spaces.  
- Renaming makes them easier to use and remember.  

**Example:**  
```python
students_df = students_df.rename(columns={"score": "final_score", "grade": "class_grade"})
print(students_df.head())
```

---

### 3. Changing Data Types  

Why?  
- Sometimes numbers are stored as text (`"25"` instead of `25`).  
- This prevents math operations unless fixed.  

**Check data types first:**  
```python
print(students_df.dtypes)
```

**Don’t you think age should be an integer instead of a float?**
**Fix the type of `age` to integer:**  
```python
students_df["age"] = students_df["age"].astype("int")
print(students_df.dtypes)
```

---

### 4. Removing Duplicates  

Sometimes the same record gets entered twice, which messes up results.  

**Example:**  
```python
print(students_df.duplicated())
```

**Count duplicates:**  
```python
print(students_df.duplicated().sum())
```

**Remove duplicates:**  
```python
students_df = students_df.drop_duplicates()
```

---

✅ Wrapping up **Topic 3 of Lesson 1**  

Cleaning data may feel boring, but it’s like sharpening your pencil ✏️ before writing an exam — it makes sure you don’t get stuck later.  

📖 More on cleaning data: [W3Schools Pandas Cleaning](https://www.w3schools.com/python/pandas/pandas_cleaning.asp)


# Topic 5 of Lesson 1: Modifying Data

So far, we’ve learned how to look at data, select/filter it, and clean it.  
Now comes the fun part, modifying the data to make it more useful!

---

## Creating New Columns

Sometimes, we need new information that doesn’t exist in the dataset.

**Example:**  
Let’s say we want to give every student 5 extra points on their score. We can create a new column for this:

```python
students_df["bonus_score"] = students_df["final_score"] + 5

# print specific columns and see the data
print(students_df[["name", "final_score", "bonus_score"]])
```

Here we created a new column called **bonus_score**:  
- It simply takes everyone’s final_score and adds 5 extra points.  
- If Bob’s final_score was 70, his bonus_score will show 75.  
- If Carol’s final_score was NaN (missing), the bonus_score will also be NaN.  

---

## Applying Functions to Columns (.apply())

`.apply()` is like telling pandas:  
"Please run this function on every element in this column."

**Example: Make all student names uppercase**

```python
students_df["name_upper"] = students_df["name"].apply(str.upper)
print(students_df[["name", "name_upper"]])
```

---


## Replacing Values

Sometimes, data has wrong values or needs standardizing.  
We can use `.replace()` for this.

**Example:**  
Suppose some students’ grade column has `"A"`, `"B"`, etc.  
We want to replace `"A"` with `"A+"`.

```python
students_df["class_grade"] = students_df["class_grade"].replace("A", "A+")
print(students_df[["name", "class_grade"]])
```

---

## Exercise

Create a new column called **name_lower** that converts all names to lowercase.

(Hint: Use `str.lower` with `.apply()`)  

**Solution Code (avoid unhiding before trying):**  

<details>
<summary>Show Solution</summary>  

```python
students_df["name_lower"] = students_df["name"].apply(str.lower)
```  
</details>

---

# Topic 6 of Lesson 1: Grouping & Aggregating

When working with data, sometimes we don’t want to look at every row, instead, we want summaries.

For instance:  
- What is the average score for each grade?  
- How many students are there in each city?  

That’s where **grouping & aggregating** comes in!

---

## Grouping Data with .groupby()

`.groupby()` is like telling pandas:  
“Please organize my data by this column.”

**Example: Group students by grade:**  

```python
grouped = students_df.groupby("class_grade")
print(grouped)
```

Note: This won’t immediately show results, it just tells pandas how the data should be grouped.  
We usually combine it with an aggregation function (like mean, sum, count).

---

## Aggregating Data

Aggregation means applying a summary calculation to each group.

**1: Average score by grade**

```python
# This will show the average score for each grade.
print(students_df.groupby("class_grade")["final_score"].mean())
```

**2: Average score of students in each city**

```python
# This counts how many students are from each city.
print(students_df.groupby("city")["id"].count())
```
> 💡 Note: Please try this code in your own VS Code (or Jupyter) to see the actual results based on the dataset.


Grouping & aggregation helps you summarize big datasets into meaningful insights without looking row by row.

---

# Topic 7 of Lesson 1: Combining Data from Different DataFrames

When working with data in the real world, we often don’t get one “perfect” dataset with everything in it.  
Instead, information is spread across multiple tables, and we need to bring it together. That’s where combining data comes in.

A simple way to think about it:  
- **Merge** → adds *columns* together from different DataFrames (makes your data wider).  
- **Concat** → stacks *rows* together from different DataFrames (makes your data longer).  

*(This is a simplified view, but it’s a useful starting point.)*

**Scenario:**  
- Table 1: students’ names and IDs.  
- Table 2: their test results.  
To do meaningful analysis, we need to bring them together.

---

## Merging Data with .merge()

Think of `.merge()` as when you want to bring two different lists together based on a common identifier.

> 🚀 **Reminder:** Open VS Code and try each Example and Check for understanding section.  
> Seeing the results yourself is the fastest way to learn.


**Example:**  

```python
# List one: Student info
students_info = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Carol"]
})
print(students_info)
```


# List two: Exam scores
exam_scores = pd.DataFrame({
    "id": [1, 2, 3],
    "score": [85, 90, 78]
})
print(exam_scores)


# Merge the two lists on "id"
merged_df = pd.merge(students_info, exam_scores, on="id")
print(merged_df)


Here, the `on="id"` tells pandas to use the id column as the matching key in both tables.

---

## Combining Data with .concat()

Imagine you have two separate lists of students, and you just want to stack them together into one list.  
Here, we don’t need a common key. We just want to put one list below the other.

**Example:**  

```python
# Class A students
class_a = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "score": [85, 90]
})
print(class_a)
```

```python
# Class B students
class_b = pd.DataFrame({
    "name": ["Carol", "David"],
    "score": [78, 92]
})
print(class_b)
```

```python
# Combine the two classes into one big list
all_students = pd.concat([class_a, class_b])
print(all_students)
```

---

## Check for Understanding 🎯

**Q1. You have one list with student IDs and names, and another list with student IDs and exam scores. You want to bring the scores next to the names.**  
Options: Should you use **merge** or **concat**?  

<details>
<summary>Show Answer</summary>  
merge (because we’re matching rows using the common key student_id).  
</details>

---

**Q2. You have two lists of students: one for Class A and another for Class B. You just want to put them together into one list of all students.**  
Options: Should you use **merge** or **concat**?  

<details>
<summary>Show Answer</summary>  
concat (because we’re stacking two datasets without matching, one below the other).  
</details>

📺 [Follow the YouTube video on merging and concatenating](https://www.youtube.com/watch?v=4mHm32u4zbQ)

---

# Topic 8 (Wrap-up) of Lesson 1: Exporting & Saving Data

After we’re done working with our data, whether it’s cleaning, exploring, or applying operations, we’ll often want to save the dataset.

**Why?**  
- To share with others.  
- To avoid repeating all steps later.  
- To use in other tools (Excel, SQL, ML libraries, etc.).  

---

## Example

Save to CSV:  
```python
students_df.to_csv("students_cleaned.csv", index=False)
```

Save to Excel:  
```python
students_df.to_excel("students_cleaned.xlsx", index=False)
```

---

## With this we are wrapping up Pandas, do not forget to try-out all the examples in your VsCode and refer attached resources for better understanding.

