
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
