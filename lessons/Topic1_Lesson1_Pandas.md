
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
