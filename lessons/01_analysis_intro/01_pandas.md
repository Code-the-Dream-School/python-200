
# Lesson 1 - Topic 1

## Recapping and Adding to What We Learned in Python 100

### Prerequisites
- Operators  
- Data Types  
- Loops  
- Functions  
- File types (.CSV, .Excel)

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

## Getting Started with Pandas

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

<img width="640" height="291" alt="Screenshot 2025-09-04 at 6 40 01 PM" src="https://github.com/user-attachments/assets/a8d01a5b-0cfa-43fc-a21e-dfad2f04b4ac" />

<img width="638" height="408" alt="Screenshot 2025-09-04 at 6 40 17 PM" src="https://github.com/user-attachments/assets/b8b9a6e0-cca6-4e71-bb23-b91953cf6c99" />

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
<img width="454" height="211" alt="Screenshot 2025-09-04 at 6 40 42 PM" src="https://github.com/user-attachments/assets/4757d3e1-81f4-4f03-b2bc-64738a8b9189" />

**From a List of Dictionaries:**  
```python
students = [
    {"name": "Alice", "age": 25, "grade": "A"},
    {"name": "Bob", "age": 30, "grade": "B"}
]
df = pd.DataFrame(students)
print(df)
```
<img width="498" height="217" alt="Screenshot 2025-09-04 at 6 40 57 PM" src="https://github.com/user-attachments/assets/d2a626d4-f7c4-48d9-aa2b-c71cbc9412f8" />

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
<img width="464" height="220" alt="Screenshot 2025-09-04 at 6 41 18 PM" src="https://github.com/user-attachments/assets/720e22bb-f2a5-4bbd-b76d-24ff728634ff" />

- Now, Read a Series (pick a column)
```python
ages = df["age"]   # this is a Series
print(ages)
```
<img width="497" height="231" alt="Screenshot 2025-09-04 at 7 09 56 PM" src="https://github.com/user-attachments/assets/d1e0ba57-cb07-4f36-9fbc-7b6c808eee43" />

---

## Creating a Series

**From a Python list:**  
```python
numbers = [10, 20, 30, 40]
num_series = pd.Series(numbers)
print(num_series)
```
<img width="298" height="209" alt="Screenshot 2025-09-04 at 6 41 29 PM" src="https://github.com/user-attachments/assets/2a063e18-7225-46aa-850b-e75c34a96bab" />

**From a list with custom labels:**  
```python
data = [85, 90, 95]
names = ["Alice", "Bob", "Carol"]
my_series = pd.Series(data, index=names)
print(my_series)
```
<img width="286" height="207" alt="Screenshot 2025-09-04 at 6 41 48 PM" src="https://github.com/user-attachments/assets/0dd95702-65bb-4439-beb8-44e5d9cca183" />

**From a Dictionary:**  
```python
data = {"Alice": 85, "Bob": 90, "Carol": 95}
my_dict_series = pd.Series(data)
print(my_dict_series)
```
<img width="282" height="226" alt="Screenshot 2025-09-04 at 6 42 02 PM" src="https://github.com/user-attachments/assets/e8771a74-4755-43f1-ae0c-3ec630c46ee0" />

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
<img width="312" height="213" alt="Screenshot 2025-09-04 at 7 07 44 PM" src="https://github.com/user-attachments/assets/e0ecca13-57e7-45c7-9919-f9da6e56cf63" />

- Now, Read a Series (pick a column)
```python
ages = df["age"]   # this is a Series
print(ages)
```
<img width="494" height="208" alt="Screenshot 2025-09-04 at 7 12 11 PM" src="https://github.com/user-attachments/assets/95fad01c-8d9b-46e9-8baa-2db4372b9b55" />

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
<img width="317" height="182" alt="Screenshot 2025-09-04 at 6 44 59 PM" src="https://github.com/user-attachments/assets/0e8fb902-6bdc-43bd-9b7f-ce9c46f1ee44" />

---

## Exploring Your Data in Pandas

When you load a dataset, you might want to get a quick look at it. Pandas provides:  
- `.head()` → Peek at first few rows  
- `.tail()` → Peek at last few rows  
- `.info()` → Structure, datatypes, missing values  
- `.describe()` → Quick statistics summary  

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
<img width="653" height="323" alt="Screenshot 2025-09-04 at 7 00 25 PM" src="https://github.com/user-attachments/assets/19088215-7512-418a-af85-025592f52775"/>

- .tail(): Look at the End. If .head() is a sneak peek at the start, then .tail() is a sneak peek at the end.
- By default, .tail() shows the last 5 rows of your dataset.
- Let’s try:
```python  
print(students_df.tail()) #prints last 5 rows
```
<img width="520" height="280" alt="Screenshot 2025-09-04 at 7 17 34 PM" src="https://github.com/user-attachments/assets/cb0e1303-ea1b-43a1-9b73-145935213a33" />

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
Gives us:
<img width="380" height="205" alt="Screenshot 2025-09-04 at 7 19 28 PM" src="https://github.com/user-attachments/assets/81977e12-5865-48d5-ae2e-05887cf7fd43" />


- What .describe() tells us:
- count: how many values (notice 5, not 6, because of missing data)
- mean: the average
- min & max: smallest and biggest 25%, 50%, 75%: quartiles (like checkpoints in the data)

So .describe() is like a quick health report of your numbers.


- Quick tip
Usually the experts suggest these three are always the first commands you should run after loading any dataset, so that you get a good knowledge of your data you will be working with.

---

## Assessment Time 🎯

You have the dataset **students.csv** loaded into a DataFrame called `students_df`.  

**Q1. How many rows are in the dataset?**  
<details>
<summary>Show Answer</summary>

6 rows
</details>

**Q2. How many columns are there?**  
<details>
<summary>Show Answer</summary>

6 columns
</details>

**Q3. Which columns have missing values?**  
<details>
<summary>Show Answer</summary>

age, grade, city, score
</details>

**Q4. What are the data types of the age and score columns?**  
<details>
<summary>Show Answer</summary>

Both are float64
</details>

**Q5 (Bonus). How many students are missing their score?**  
<details>
<summary>Show Answer</summary>

1 student
</details>

---


# Lesson 1 - Topic 2

## Selecting & Filtering Data

---

### Selecting Columns

Think of a column like a single list/Series inside the table.  
Sometimes you don’t need the whole table, only a few columns.  

For instance: In a dataset with student info, maybe you only care about **name** and **score** for a leaderboard.

**Select one column:**  
```python
print(students_df["name"])  # This gives you a Series (just one column)
```
<img width="502" height="197" alt="Screenshot 2025-09-04 at 6 27 02 PM" src="https://github.com/user-attachments/assets/b204acbf-137c-4018-be41-655f20593a12" />

**Select multiple columns:**  
```python
print(students_df[["name", "age"]])  # This gives you a DataFrame (2 columns)
```
<img width="371" height="214" alt="Screenshot 2025-09-04 at 6 27 17 PM" src="https://github.com/user-attachments/assets/507d6b65-d88d-4b62-bcf1-528a2f8ef35f" />

---

### Selecting Rows

Datasets can be huge, millions of rows. Being able to grab just the row(s) you need helps when debugging, checking a specific entry, or sampling data.

#### `.loc[]`  
- Great when your dataset has **meaningful labels** (names, IDs, dates, etc.).  
- Instead of remembering row numbers, you can just use the label.  

**Example: Get the row for Carol**  
```python
students_df = students_df.set_index("name")
print(students_df.loc["Carol"])
```
<img width="315" height="136" alt="Screenshot 2025-09-04 at 6 22 54 PM" src="https://github.com/user-attachments/assets/86e23600-f251-457f-8296-666bae5e2cd2" />

#### `.iloc[]`  
- Think **i for index number**.  
- Uses numbers (row position).  
- Helpful when you know the position/index of the element.  

**Example: Get the 3rd row (Carol):**  
```python
students_df = students_df.reset_index()
print(students_df.iloc[2])
```
<img width="208" height="137" alt="Screenshot 2025-09-04 at 6 23 17 PM" src="https://github.com/user-attachments/assets/fb9f9759-093e-441f-995d-3254415cd839" />

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
<img width="496" height="75" alt="Screenshot 2025-09-04 at 6 23 45 PM" src="https://github.com/user-attachments/assets/f95415cc-b165-4ee4-86c4-3dc1eaedfc33" />

**Example: Students with grade “A”**  
```python
print(students_df[students_df["grade"] == "A"])
```
<img width="484" height="93" alt="Screenshot 2025-09-04 at 6 24 01 PM" src="https://github.com/user-attachments/assets/4fd6d2bc-7041-41b9-b867-e146f9a93b45" />

---

### Sorting Data

We can sort rows by any column. Sorting helps you see patterns.  

For example:  
- Sort students by **score** to see the top performers.  
- Sort by **age** to see youngest/oldest.  

**Example: Sort by score (descending):**  
```python
students_df.sort_values("score", ascending=False)
```
<img width="532" height="239" alt="Screenshot 2025-09-04 at 6 24 12 PM" src="https://github.com/user-attachments/assets/d4bd9642-d929-4c77-ae31-90ff17bd68a8" />

---

## Quiz: Selecting & Filtering Data 🎯

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
<img width="453" height="148" alt="Screenshot 2025-09-04 at 6 34 00 PM" src="https://github.com/user-attachments/assets/5bdddf7f-e8a3-4942-bfb7-eff9184dd8b7" />


👉 Output: A table of `True`/`False`.  
- `True` = value is missing (NaN).  
- `False` = value is present.  

**Count missing values in each column:**  
```python
print(students_df.isnull().sum())
```
<img width="419" height="193" alt="Screenshot 2025-09-04 at 6 34 45 PM" src="https://github.com/user-attachments/assets/daefbded-9c50-4736-81e3-d9a3bddd7c6d" />

👉 Easier to read than the True/False table.  

**Fix missing values with `.fillna()`:**  
```python
students_df["age"] = students_df["age"].fillna(students_df["age"].mean())
print(students_df.isnull())
```
<img width="437" height="138" alt="Screenshot 2025-09-04 at 6 35 14 PM" src="https://github.com/user-attachments/assets/c31bfd08-2bd9-4f10-927c-72ae98fde20f" />

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
<img width="592" height="202" alt="Screenshot 2025-09-04 at 6 35 41 PM" src="https://github.com/user-attachments/assets/5794420c-68b1-411d-a35c-d8eb82e77348" />

---

### 3. Changing Data Types  

Why?  
- Sometimes numbers are stored as text (`"25"` instead of `25`).  
- This prevents math operations unless fixed.  

**Check data types first:**  
```python
print(students_df.dtypes)
```
<img width="620" height="225" alt="Screenshot 2025-09-04 at 6 35 54 PM" src="https://github.com/user-attachments/assets/9cc9683b-02d0-4b62-beca-7f3cae2da94c" />

**Don’t you think age should be an integer instead of a float?**
**Fix the type of `age` to integer:**  
```python
students_df["age"] = students_df["age"].astype("int")
print(students_df.dtypes)
```
<img width="555" height="192" alt="Screenshot 2025-09-04 at 6 36 53 PM" src="https://github.com/user-attachments/assets/17d90d35-e437-4160-990f-756120a02d48" />

---

### 4. Removing Duplicates  

Sometimes the same record gets entered twice, which messes up results.  

**Check for duplicates:**  
```python
print(students_df.duplicated())
```
<img width="324" height="182" alt="Screenshot 2025-09-04 at 6 37 07 PM" src="https://github.com/user-attachments/assets/4b031ca5-639c-4f4d-9ae2-31e7aaf5e4f3" />

**Count duplicates:**  
```python
print(students_df.duplicated().sum())
```
<img width="174" height="51" alt="Screenshot 2025-09-04 at 6 37 35 PM" src="https://github.com/user-attachments/assets/0cbdeb74-28b9-4684-a729-cd4bc58087a0" />

**Remove duplicates:**  
```python
students_df = students_df.drop_duplicates()
```

---

## 📝 Exercise: Clean a Small DataFrame

**Goal:** Create a small DataFrame with missing values and duplicates, then:  
1. Find the missing values (`.isnull()` and `.isnull().sum()`).  
2. Count duplicates (`.duplicated().sum()`).  
3. Fill missing numeric values with the median (`.median()`).  
4. Remove duplicates (`.drop_duplicates()`).  
5. Verify your work by re-running the checks.  

**Starter Code:**  
```python
# import pandas
import pandas as pd

# Create a dataframe with duplicates and missing values
data = {
    "id":   [101, 102, 102, 103, 104, 105],
    "age":  [25, 30, 30, None, 22, None],
    "score":[88, 92, 92, 75, None, 85]
}

practice_df = pd.DataFrame(data)

# See the created dataframe
print(practice_df)

# Write your code here
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
<img width="430" height="159" alt="Screenshot 2025-09-09 at 3 44 19 PM" src="https://github.com/user-attachments/assets/3c15f083-05a8-4bed-afa3-43c76fcfc542" />

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
<img width="304" height="184" alt="Screenshot 2025-09-09 at 3 45 27 PM" src="https://github.com/user-attachments/assets/a034f773-7107-4e5d-a590-9bd7204bbadd" />

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
<img width="263" height="178" alt="Screenshot 2025-09-09 at 3 45 38 PM" src="https://github.com/user-attachments/assets/dc283845-8858-441f-8d31-2d2cd14330c7" />

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

**Example 1: Average score by grade**

```python
# This will show the average score for each grade.
print(students_df.groupby("class_grade")["final_score"].mean())
```
<img width="406" height="144" alt="Screenshot 2025-09-09 at 3 46 15 PM" src="https://github.com/user-attachments/assets/f58b548c-b062-4318-867c-74bf196d7e8a" />

**Example 2: Number of students in each city**

```python
# This counts how many students are from each city.
print(students_df.groupby("city")["id"].count())
```
<img width="286" height="191" alt="Screenshot 2025-09-09 at 3 46 26 PM" src="https://github.com/user-attachments/assets/175bcf71-13ca-46d4-b0d8-90d1bb3cac84" />

Grouping & aggregation helps you summarize big datasets into meaningful insights without looking row by row.

---

# Topic 7 of Lesson 1: Merging & Combining Data

When working with data in the real world, we often don’t get one “perfect” dataset with everything in it.  
Instead, information is spread across multiple tables and we need to bring it together. That’s where merging and combining come in.

**Scenario:**  
- Table 1: students’ names and IDs.  
- Table 2: their test results.  
To do meaningful analysis, we need to bring them together.

---

## Merging Data with .merge()

Think of `.merge()` as when you want to bring two different lists together based on a common identifier.

**Example:**  

```python
# List one: Student info
students_info = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Carol"]
})
print(students_info)
```
<img width="234" height="132" alt="Screenshot 2025-09-09 at 3 46 50 PM" src="https://github.com/user-attachments/assets/bf61c077-246f-44be-aa7b-02695288f7ba" />


# List two: Exam scores
exam_scores = pd.DataFrame({
    "id": [1, 2, 3],
    "score": [85, 90, 78]
})
print(exam_scores)

<img width="201" height="128" alt="Screenshot 2025-09-09 at 3 47 13 PM" src="https://github.com/user-attachments/assets/6986efda-0a19-4404-94d2-ad2de26a0277" />

# Merge the two lists on "id"
merged_df = pd.merge(students_info, exam_scores, on="id")
print(merged_df)

<img width="456" height="170" alt="Screenshot 2025-09-09 at 3 49 35 PM" src="https://github.com/user-attachments/assets/c730d5d6-bb61-4d19-bd04-98fc3e1319bc" />

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
<img width="321" height="125" alt="Screenshot 2025-09-09 at 3 50 18 PM" src="https://github.com/user-attachments/assets/8f5fa2d8-863b-459a-b168-bdcae3824685" />

```python
# Class B students
class_b = pd.DataFrame({
    "name": ["Carol", "David"],
    "score": [78, 92]
})
print(class_b)
```
<img width="290" height="124" alt="Screenshot 2025-09-09 at 3 50 35 PM" src="https://github.com/user-attachments/assets/3e654800-0443-4eee-87ee-af38672625c3" />

```python
# Combine the two classes into one big list
all_students = pd.concat([class_a, class_b])
print(all_students)
```
<img width="221" height="146" alt="Screenshot 2025-09-09 at 3 53 51 PM" src="https://github.com/user-attachments/assets/4e6825e6-58f5-455a-9725-df3b4b6ea623" />

---

## Quiz Time 🎯

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

# Topic 8 of Lesson 1: Pandas "EDA Superpower" using ydata-profiling

Imagine you’ve just received a large dataset and don’t know where to start.  
Instead of manually checking column by column, you can run a single function that summarizes:  
- Missing values  
- Data types  
- Distributions of each column  
- Correlations between columns  
- Basic visualizations  

Pandas can create an **Exploratory Data Analysis (EDA)** report for you automatically.

---

## Example with ydata-profiling

```python
import pandas as pd  

# load the students data
students_df = pd.read_csv("students.csv")
print(students_df.head())
```

### Step 2: Install the library

```bash
pip install ydata-profiling
```

### Step 3: Import the library

```python
from ydata_profiling import ProfileReport
```

### Step 4: Create a profile report

```python
profile = ProfileReport(students_df, title="Students Data Report", explorative=True)
```

### Step 5: View the report

- Save as HTML:  
```python
profile.to_file("students_report.html")
```

- Or view inline in Jupyter/Colab:  
```python
profile.to_notebook_iframe()
```

---

# Topic 9 (Wrap-up) of Lesson 1: Exporting & Saving Data

After we’re done working with our data, whether it’s cleaning, exploring, or applying operations, we’ll often want to save the dataset.

**Why?**  
- To share with others.  
- To avoid repeating all steps later.  
- To use in other tools (Excel, SQL, ML libraries, etc.).  

---

## Examples

Save to CSV:  
```python
students_df.to_csv("students_cleaned.csv", index=False)
```

Save to Excel:  
```python
students_df.to_excel("students_cleaned.xlsx", index=False)
```

---

## Final Tip for Pandas

You’ll definitely run into errors sometimes, that’s normal!  
Just read the error message carefully and search online. Errors are your best hints for fixing problems.

📺 [Learning Pandas for Data Analysis?](https://www.youtube.com/watch?v=DkjCaAMBGWM)  
📺 [What is Pandas?](https://www.youtube.com/watch?v=dcqPhpY7tWk&t=70s)
