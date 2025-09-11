
# Topic: NumPy - Lesson 1

## Installing, Importing and Why NumPy Arrays over Python Lists?

### Installing and Importing NumPy

Before we learn why NumPy is so powerful, let’s quickly set it up.

**Step 1:** To Install NumPy, open your terminal (in VS Code) and run:

```bash
pip install numpy
```

This installs NumPy in your Python environment.  
(You only need to do this once.)

**Step 2:** To Import NumPy use:

```python
import numpy as np
```

Here, we give NumPy the nickname `np` (a common convention).  
Now, whenever we want to use NumPy, we write `np.something`.

---

### Why NumPy Arrays instead of Python Lists?

When we work with data, we often deal with large collections of numbers. Real-world data is usually large (thousands or millions of values).

NumPy arrays make your work faster, more memory-efficient, and less error-prone.  
This is also the reason why NumPy is the foundation of data science — almost every other library (like Pandas, Scikit-learn, TensorFlow) is built on top of NumPy.

For example: exam scores for 10 students is fine with a list. But what if you had 10,000 students? Or some kind of sensor readings collected every second with millions of data points?  
👉 This is where NumPy arrays shine.

---

### Memory Efficiency & Speed

A Python list can store different types (numbers, strings, objects). That’s flexible, but it takes more memory and is slower.

NumPy arrays store only **one type** (all ints/integers, or all floats) in a continuous block of memory.

Think of it like:  
- Python lists = a bag with many boxes inside (lots of overhead).  
- NumPy arrays = a straight row of boxes (compact and fast).  

**Example: Memory usage of Python list vs NumPy array**

```python
# Memory Efficiency & Speed
import numpy as np
import sys

# Python list of 1000 numbers
py_list = list(range(1000))

# NumPy array of 1000 numbers
np_array = np.arange(1000)

print("Size of Python list:", sys.getsizeof(py_list))
print("Size of NumPy array:", np_array.nbytes)
```

As we can see in the above output, NumPy uses less memory, which means faster calculations, especially with large datasets.

---

### Easier Math with Arrays

Imagine you have a list of student exam scores:  
In a Python list, if you want to add 5 bonus points to each student’s score, you must write a loop and update each element one by one.  

With NumPy arrays, you just say “add 5 to the whole array” and NumPy does it for you.

**Python List Example:**  
```python
# Python list
scores = [50, 60, 70, 80]

# Add 5 bonus points using a loop
new_scores = []
for s in scores:
    new_scores.append(s + 5)

print(new_scores)   # [55, 65, 75, 85]
```

By seeing the code you might have noticed how we needed a loop (`for s in scores`) to do this.

**NumPy Array Example:**  
```python
# NumPy array
scores = np.array([50, 60, 70, 80])

# Add 5 bonus points directly
new_scores = scores + 5
print(new_scores)   # [55 65 75 85]
```

✅ No loops needed! NumPy does the math for the whole array in one shot.

---

# Topic 11 of Lesson 1: Array Creation, Indexing & Slicing

We just learned that the NumPy array is like a super-efficient solution, where you can quickly store and work with numbers. Let’s see how to create arrays, access values, and pick slices (subsets).

---

### Creating Arrays

NumPy gives us many ways to make arrays easily.

**Example 1: From a Python list**  
```python
# Convert a list into a NumPy array
arr = np.array([1, 2, 3, 4, 5])
print(arr) # Output: [1 2 3 4 5]
```

This is just like your list, but super-powered with NumPy’s speed.

**Example 2: Special arrays**  
Sometimes, you don’t want to type numbers manually. NumPy can create arrays for you:

```python
# Array of numbers from 0 to 9
print(np.arange(10)) # Output: [0 1 2 3 4 5 6 7 8 9]

# Array of all zeros
print(np.zeros(5)) # Output: [0. 0. 0. 0. 0.]

# Array of all ones
print(np.ones(5)) # Output: [1. 1. 1. 1. 1.]
```

👉 Follow the article for more on creating NumPy arrays:  
[GeeksforGeeks - Different Ways to Create NumPy Arrays](https://www.geeksforgeeks.org/numpy/different-ways-to-create-numpy-arrays-in-python/)

---

### NumPy Array Indexing

Think of a NumPy array like a row of seats in a movie theater.  
Each seat has a number (starting from 0), and each number tells you who’s sitting there.

**Example:**  
```python
arr = np.array([10, 20, 30, 40, 50])

print(arr[0])   # Seat 0: output is 10
print(arr[3])   # Seat 3: output is 40
```

So if you want the student in seat #0, you get 10.  
If you ask for seat #3, you get 40.

---

### Indexing in 2D NumPy Arrays (Tables)

Now imagine instead of one row of seats, we have a classroom seating chart.

- Rows = rows of benches  
- Columns = seats in each row  

**Example:**  
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(arr_2d[0, 0])  # Row 0, Col 0 → 1
print(arr_2d[1, 2])  # Row 1, Col 2 → 6
```

Explanation:  
- `arr_2d[0, 0]` → first row, first column (front-left seat).  
- `arr_2d[1, 2]` → second row, third column (second row, far-right seat).  

---

### Slicing in 2D Arrays (Tables)

Indexing is like pointing to one exact student’s seat.  
But sometimes, you want a whole row or block. That’s where slicing comes in.

**Example:**  
```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(arr_2d[0, :])     # All columns from Row 0
print(arr_2d[:, 1])     # All rows from Column 1
print(arr_2d[0:2, 0:2]) # Rows 0-1, Cols 0-1
```

Output:  
```
[1 2 3]    # whole first row
[2 5 8]    # whole second column
[[1 2]     # top-left 2x2 block
 [4 5]]
```

---

### Broadcasting in NumPy

Broadcasting is NumPy’s smart way of handling math when arrays have different shapes. NumPy repeats the smaller array automatically so it matches the bigger one during operations.

**Example 1: Adding a Number to an Array**  
```python
arr = np.array([1, 2, 3])
print(arr + 5)   # [6 7 8]
```

**Example 2: Adding a Vector to a Matrix**  
```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
vector = np.array([10, 20, 30])

print(matrix + vector)
```
Output:  
```
[[11 22 33]
 [14 25 36]]
```

NumPy saw that the matrix has shape (2,3) and the vector has shape (3,).  
It stretched the vector across both rows so the math works.

---

# Topic 12 of Lesson 1: Basic Statistics with NumPy

When you collect numbers (like exam scores, sales, or daily temperatures), you often want to summarize them.  
That’s where statistics come in.

NumPy makes this super easy — no manual math, just one function call.

Think of it this way:  
Imagine you’re a teacher with scores of 100 students.  
Instead of checking every score one by one, you want quick answers like:  
- What’s the average score? (**Mean**)  
- Who got the highest and lowest? (**Max/Min**)  
- How spread out are the scores? (**Standard Deviation**)  
- What’s the total marks for the class? (**Sum**)  

**Example:**  
```python
scores = np.array([60, 72, 85, 90, 45, 100, 77])

print("Mean (average):", np.mean(scores)) 
print("Maximum:", np.max(scores)) 
print("Minimum:", np.min(scores)) 
print("Sum:", np.sum(scores)) 
print("Standard Deviation:", np.std(scores)) 
```

Output:  
```
Mean (average): 75.57
Maximum: 100
Minimum: 45
Sum: 529
Standard Deviation: 17.93
```

**Key Idea:**  
- Mean → Average performance of the class  
- Max/Min → Best and worst student  
- Sum → Total marks (or sales revenue, etc.)  
- Standard Deviation → How much scores differ from the average  

📌 This is super useful in real life: businesses use it for sales trends, sports teams for performance consistency, and teachers for class analysis.
