
# Data Visualization using Matplotlib

## Why Visualization Matters?

Imagine you have exam scores of 100 students. You could look at the raw numbers in a table, but that would feel overwhelming.  
Instead, a single chart can instantly show you trends, like who’s performing better, how many failed, or whether scores are clustered around a certain range.

👉 That’s the power of Matplotlib — it turns numbers into pictures.

---

## Installing and Importing Matplotlib

In VS Code, run:  
```bash
pip install matplotlib
```

Then import the library:  
```python
import matplotlib.pyplot as plt
```

---

## Basic Plotting

### 1. Line Plot (Trends over time)
**Use case:** Tracking your monthly expenses.

```python
months = ["Jan", "Feb", "Mar", "Apr"]
expenses = [200, 250, 300, 220]

plt.plot(months, expenses)
plt.title("Monthly Expenses")
plt.xlabel("Month")
plt.ylabel("Amount ($)")
plt.show()
```

Output:
<img width="439" height="347" alt="Screenshot 2025-09-09 at 3 02 18 PM" src="https://github.com/user-attachments/assets/4389bc0b-443d-4204-989a-a6161397aad7" />


---

### 2. Bar Plot (Comparisons)
**Use case:** Comparing marks of different students.

```python
students = ["Alice", "Bob", "Charlie"]
marks = [85, 90, 78]

plt.bar(students, marks, color="orange")
plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")
plt.show()
```

Output:
<img width="512" height="383" alt="Screenshot 2025-09-09 at 3 02 34 PM" src="https://github.com/user-attachments/assets/ab4ab418-b5cb-4a28-95d0-e1a2c0a92cfc" />


---

### 3. Scatter Plot (Relationships between two things)
**Use case:** Hours studied vs. exam score.

```python
hours = [2, 4, 6, 8, 10]
scores = [50, 60, 70, 80, 90]

plt.scatter(hours, scores, color="green")
plt.title("Hours Studied vs Score")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()
```

Output:
<img width="385" height="299" alt="Screenshot 2025-09-09 at 3 02 51 PM" src="https://github.com/user-attachments/assets/d63b3bfb-fa1a-481f-a64b-17cecce0b97c" />


---

### 4. Histogram (Distribution of data)
**Use case:** How exam scores are spread out in a class.

```python
scores = [55, 60, 61, 62, 65, 70, 72, 80, 85, 90, 90, 92, 95]

plt.hist(scores, bins=5, color="purple", edgecolor="black")
plt.title("Score Distribution")
plt.xlabel("Score Range")
plt.ylabel("Number of Students")
plt.show()
```

Output:
<img width="459" height="358" alt="Screenshot 2025-09-09 at 3 03 06 PM" src="https://github.com/user-attachments/assets/b0e3937f-1872-44fd-b677-ff233cbb4342" />


---

## Customization (Making Charts Readable)

Adding titles, labels, legends, colors makes your chart easy to understand.

**Example:**  
```python
plt.plot(months, expenses, label="Expenses", color="red", marker="o")
plt.title("Monthly Expenses")
plt.xlabel("Month")
plt.ylabel("Amount ($)")
plt.legend()
plt.show()
```

Output:
<img width="394" height="297" alt="Screenshot 2025-09-09 at 3 03 23 PM" src="https://github.com/user-attachments/assets/07030e5b-766b-4983-9b0b-513c84183790" />

---

## Quiz: Data Visualization with Matplotlib 🎯

**Q1. Which Python library do we use for data visualization in this lesson?**  
<details>
<summary>Show Answer</summary>  

Matplotlib (`import matplotlib.pyplot as plt`)  

</details>

**Q2. Which type of plot is best for showing trends over time?**  
<details>
<summary>Show Answer</summary>  

Line Plot  

</details>

**Q3. Which plot would you use to compare marks of students?**  
<details>
<summary>Show Answer</summary>  

Bar Plot  

</details>

**Q4. If you want to see the relationship between hours studied and exam scores, which plot should you use?**  
<details>
<summary>Show Answer</summary>  

Scatter Plot  

</details>

**Q5. Which plot helps you understand the distribution of exam scores in a class?**  
<details>
<summary>Show Answer</summary>  

Histogram  

</details>

**Q6. How do you add a legend to a chart in Matplotlib?**  
<details>
<summary>Show Answer</summary>  

Use `plt.legend()`  

</details>


---

## More Matplotlib Features to Know

The four basic plots (line, bar, scatter, histogram) are the foundations.  
But you’ll often see and use a few additional features in practice. Here’s a quick taste:

### 1. Object-Oriented Interface

So far we’ve used the **pyplot** style (`plt.plot(...)`).  
Matplotlib also has an **object-oriented** style, which is more flexible and what you’ll see in most real projects.  
It separates the **Figure** (overall canvas) from the **Axes** (individual plots).

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()  # Create a figure and axis
ax.plot(x, y)
ax.set_title("Sine Wave")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
plt.show()



### 2. Visualizing 2D Arrays with `imshow`

NumPy arrays can represent images or grids of data.  
You can visualize them directly with `imshow()` — very useful when working with image data later in the course.

```python
arr = np.random.rand(100, 100)

plt.imshow(arr, cmap="viridis")
plt.colorbar()  # Show color scale
plt.title("Random 2D Array (like an image)")
plt.show()

### 3. Subplots

Sometimes you want to show multiple plots side by side.  
`subplots()` makes it easy to lay out multiple Axes in the same Figure.

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(x, y1, color="blue")
axes[0].set_title("Sine")

axes[1].plot(x, y2, color="green")
axes[1].set_title("Cosine")

plt.show()


### 4. Customizing Line Styles

Matplotlib gives you fine control over **line style, color, markers, labels, and more**.  
Here’s a slightly fancier plot that combines several options.

```python
x = np.arange(0, 100)
y1 = np.sin(x / 10)
y2 = np.cos(x / 10)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, linewidth=1, color='r', linestyle='--', label="t1", zorder=3)
plt.plot(x, y2, linewidth=0.5, color='b', marker='.', markersize=4, label="t2")
plt.axis([0, 100, -1.8, 1.8])
plt.xlabel("Location", fontsize=12)
plt.ylabel("Height", fontsize=12)
plt.title("Wave Height at Different Times", fontsize=16)
plt.legend(loc='upper right')
plt.show()

---

## Additional Resources 📚

If you’d like to go further with Matplotlib, here are some excellent resources:

- 🎥 [Corey Schafer’s YouTube Series on Matplotlib](https://www.youtube.com/watch?v=UO98lJQ3QGI&list=PL-osiE80TeTvipOqomVEeZ1HRrcEvtZB_) — great video walkthroughs.  
- 📖 [DataCamp Matplotlib Tutorial](https://www.datacamp.com/tutorial/matplotlib-tutorial-python) — step-by-step written guide.  

These will give you extra practice and deeper insights into data visualization.

📌 Keep practicing, the more you play with data, the more confident you’ll get. You’re off to a great start!

---

## Lesson 1 Wrap-Up

🎉 Congrats! You’ve completed Lesson 1.  

You now know the **three pillars of data analysis**:  
- **Pandas** → handle and explore data tables  
- **NumPy** → fast math & calculations  
- **Matplotlib** → turn numbers into visual stories  

With just these, you can already clean data, analyze trends, and make charts. This is the foundation every data scientist uses daily. 

---

