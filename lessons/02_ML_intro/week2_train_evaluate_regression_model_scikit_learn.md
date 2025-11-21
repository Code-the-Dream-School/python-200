# CTD Python 200 2

## Weeks 2–4: Machine Learning  
### Week 2: Introduction to Machine Learning and Regression  
#### 4. Train and evaluate a regression model using scikit-learn

### Let’s pause and recap

In Week 2 (Machine Learning), by now you might have learned most of the foundational ideas:

- Heard what **machine learning** is  
- Seen examples of **ML problems**  
- Learned about the **types of ML**, like  
  - supervised vs. unsupervised learning  
  - regression vs. classification  
- Explored **scikit-learn**

---

### Essential Resource 
These will help you follow along:

- **YouTube:** https://youtu.be/ukZn2RJb7TU?si=Jz65OxZeGuUKDbO7

---

### Now what?

Now we’re moving from *understanding what ML is*  
to actually **doing ML**.

We’ll teach the computer to learn from data and make predictions —  
this process is called **training and evaluation**.

---

### Let’s understand what is training and evaluation (testing) in ML

![week_2](resources/image_1.png)

**Training** is the process where a machine learning model learns patterns and relationships from the data we give it.

It looks at many examples of inputs (features) and their correct answers (labels or targets), and tries to find a rule or formula that connects them.

In other words, during training the model is learning how changes in the input affect the output, so it can make good predictions on new, unseen data later.

---

### Think of training this way:

If we show the model house sizes (input) and their prices (output), it learns how price usually changes when the size changes — **that’s training**.

Think of training like *studying from examples*.  
The model “studies” the training data to learn how inputs relate to outputs.

In scikit-learn, this happens when we call:

```python
model.fit(X_train, y_train)
```
---

### Evaluation (Testing) in Machine Learning

And, **evaluation** is the process of checking how well a trained model performs, in other words, how good it is at making predictions on new, unseen data.

After the model has learned from the training data, we test it using a separate part of the data called the **test set**. This helps us see if the model has really learned the pattern, or if it just memorized the training examples.

Think of evaluation this way:  
After studying (training), a student takes an exam (evaluation) to see how well they can apply what they’ve learned to new questions they haven’t seen before.

We measure the model’s performance using evaluation metrics, such as:

- **MSE (Mean Squared Error)** — shows how far the predictions are from the actual values (smaller is better).  
- **R² Score** — shows how well the model explains the variation in the data (closer to 1 means a better fit).

---

### Training and Test Sets-Why We Split the Data

![week_2](resources/image_2.png)

When we train a model, we want to know how well it can make predictions on **new data**, not just the data it has already seen.

Think of it this way:  
When you study for a test, you practice using your notes and homework problems (training). Then, during the real exam (testing), you get new questions you haven’t seen before — that’s how we see if you truly understand the topic!

---

### Splitting the Dataset into Training and Test Sets

**To do that, we divide our dataset into two parts:**

**Training set:** The data the model learns from. The model uses this data to find patterns and relationships.

**Test set:** The data we keep aside to check how well the model learned. This simulates how the model will perform on real-world, unseen and new data.

We usually use around **70–80%** of the data for training and **20–30%** for testing.

**In scikit-learn, we can easily do this with:**

```python
from sklearn.model_selection import train_test_split (imports a function from scikit-learn)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\

```

### This line splits our data into four parts:

- **X_train:** features for training  
- **X_test:** features for testing  
- **Y_train:** labels (answers) for training  
- **Y_test:** labels for testing  

### What the parameters mean

**test_size=0.2**  
Means **20%** of the data will be used for testing, and the rest (**80%**) for training.

**random_state=42**  
Just makes the split repeatable, so we get the same result every time we run the code.


### Quick note

Sometimes, we even split the data into **three parts**:  
**training**, **validation**, and **test**.

The **validation set** helps us fine-tune and improve the model before doing the final test.

We’ll explore that later when we learn how to improve model performance.

---

### Let’s also quickly recap Regression Model & scikit-learn

#### What is a Regression Model?

A regression model is used when we want to predict a number, a value that can go up and down. This kind of value is called a **continuous value**.

**Examples of things we might want to predict:**

- House price (in dollars)  
- Car mileage (in miles per gallon)  
- Temperature (in degrees)  
- Time something will take (in minutes)

In all these cases, the answer is a **number**, not a label. That’s why we use regression.

---
![week_2](resources/image_3.png)

### Let’s understand with a House Prices example:

Suppose we want to predict the price of a house.

**We know that things like:**
- House size (square feet)  
- Number of bedrooms  
- Neighborhood  

usually affect the price.

**A regression model looks at many past houses and learns:
**
- “If size increases, price tends to increase.”  
- “If located in area A, the price might be higher.”

It learns the relationship between the **inputs (features)** and the **price (target)**.

---

### Let’s also recap: What is scikit-learn?

**scikit-learn** is a Python library that gives us easy-to-use tools for building machine learning models.  
Instead of writing math from scratch, we use scikit-learn to handle the heavy lifting.

---

### Why we use scikit-learn

- It provides ready-to-use models (like Linear Regression).  
- It provides tools to split data into training and test sets.  
- It provides evaluation metrics to measure how good our model is.  
- It keeps our code simple, clear, and consistent.

---

### Connecting to Our Housing Example

![week_2](resources/image_4.png)


Think about predicting house prices.

**If we tried to figure this out ourselves, we’d need to:**

- Study a lot of past house sales  
- Notice patterns (like bigger houses cost more)  
- Build a formula  
- Test whether our formula works on new houses  

scikit-learn helps us do exactly this, but automatically.  
We just tell scikit-learn:

- What data to learn from (house size, number of bedrooms…)  
- What value we want to predict (house price)  

And scikit-learn learns the relationship for us.

---

### The scikit-learn general workflow, in simple:

**1. Choose a model:**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

**2. Train the model on the training data:**
```python
model.fit(X_train, y_train)
```

**3. Make predictions on new data:**
```python
predictions = model.predict(X_test)
```

**4. Evaluate how close the predictions are to the real prices:**
```python
from sklearn.metrics import mean_squared_error, r2_score
```
---

### Evaluation metrics for regression

When we build a regression model (such as predicting house prices), we need a way to measure how well the model is performing. We compare the model’s predicted prices with the actual prices from the dataset.

![week_2](resources/image_4.png)

---

**Here are the two most common evaluation metrics we use:**

#### 1. Mean Squared Error (MSE)

MSE tells us how far off the model’s predictions are, on average.  
It works by taking the squared difference between predicted and actual prices.

**Lower MSE = Better model**

A high MSE means the model’s predictions are very different from the real prices.

If the model keeps making big mistakes, MSE becomes large.

**Example:**  
If the model predicts a house should cost **$5,000,000** but the real price is **$4,000,000**,  
the error is large — MSE increases.


#### 2. R² Score (R-Squared)

R² tells us how well the model explains the pattern in the data.

- **R² = 1 → Perfect prediction**  
- **R² = 0 → Model learned nothing useful**  
- **Higher R² = Better fit**

R² shows how much of the variation in house prices the model can explain.

---

### Introduction to overfitting and underfitting

When we train a machine learning model, our goal is for it to learn real patterns in the data so it can make good predictions on new houses (not just the ones it has already seen).

However, sometimes the model learns too little or too much.  
This leads to **underfitting** or **overfitting**.

![week_2](resources/image_5.png)

### Underfitting

The model didn’t learn enough.  
The model is too simple and misses important patterns in the data.

**In the Housing Example:**  
If the model only uses area to predict price and ignores features like bedrooms, bathrooms, or location, its predictions will be very rough and inaccurate.

**Signs of Underfitting:**
- Low accuracy on training data  
- Low accuracy on test data  

**Simple way to think of it:**  
Like a student who didn’t study enough — they perform poorly on homework *and* on the test.

---

### Overfitting

The model learned too much — it **memorized** the training data.  
The model becomes too complex and tries to remember every detail, even noise or random patterns.

**In the Housing Example:**  
If the model tries to fit every small fluctuation in price (like whether the house has a specific shade of paint), it will perform great on training data but fail when predicting a new house.

**Signs of Overfitting:**
- High accuracy on training data  
- Low accuracy on test data  

**Simple way to think of it:**  
Like a student who memorized answers instead of understanding — they do well on practice, but poorly on a new test.

---

### Goal: Find the Balance

**We want a model that:**
- Learns the important patterns  
- Ignores noise and random details  
- Performs well on both training and test data  

This is called **generalization** — being able to make good predictions on new, unseen houses.

---

### Next Step

Now we can move into the actual training step:

**Creating the model → Fitting → Predicting → Evaluating**

![week_2](resources/image_6.png)

---

### Hands-on activity:  
#### Train a Regression Model to Predict House Prices

In this activity, we will use real housing data to build a Linear Regression model that can predict the price of a house based on its features (like size, bedrooms, etc.).  
We will train the model, test how well it performs, and visualize the results to understand how accurate our predictions are.

Before we start training a machine learning model, we need to understand the data we’ll be working with and the process we will be following:

We will use a **Housing Dataset** (included in the resources folder).

It can also be downloaded directly from Kaggle.com:  
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset?resource=download

This dataset contains information about houses (like size, number of bedrooms, amenities, etc.) along with their selling prices.

### Steps We Will Follow

**Explore the dataset**  
Look at the data and understand what information we have.

**Select features (X) and target (y)**  
Decide which columns will be used to predict the house price.

**Split the data into training and test sets**  
Train the model on one portion of the data and test it on the rest.

**Train a Linear Regression model using scikit-learn**  
Teach the model to learn the relationship between features and price.

**Make predictions on the test data**  
See what prices the model predicts for unseen houses.

**Evaluate and visualize predictions**  
- Plot predicted vs actual prices  
- Look at residuals (errors)  
- Use metrics like MSE and R² to measure performance  

**Compare training vs testing performance**  
Check for overfitting or underfitting.

**Discuss what we learned**  
Interpret the results and reflect on model performance.

---
### Now let's complete this activity inside a Jupyter Notebook.
You can find the notebook for this lesson in the Resources folder (or the Lessons folder for this week).

Please open the notebook titled: **Week2_House_Price_Regression.ipynb**

---

### Student Reflection / Observations

After completing the notebook activity, take a moment to reflect on your experience and observations from the Housing Price Prediction model.

Write 4–5 short points about what you noticed. You can think about:

- Patterns in the data and how features affected price  
- Whether the model's predictions were close to the actual prices  
- Which parts of the model seemed to work well  
- What could be improved, such as adding more features or trying a different model  
- Any interesting or surprising findings  

Here are some sentence starters to help you reflect:

- "I noticed that the model did/did not capture the relationship between ______ and price."  
- "The model struggled when predicting houses with ______."  
- "Adding more features might improve the accuracy because ______."  
- "The residuals show that ______."  
- "I was surprised to see that ______ seemed to influence price the most."  

Take your time and write your observations clearly. The goal is to understand what happened and think about how the model could be improved.

---

## Check for Understanding  
### Introduction to Machine Learning & Regression

---

### 1. What is the goal of training a machine learning model?

a. To memorize the dataset  
b. To learn patterns in the data and make predictions  
c. To generate random numbers  
d. To sort data alphabetically  

<details>
<summary><strong>Show Answer</strong></summary>
b — The model learns patterns so it can make useful predictions.
</details>

---

### 2. Why do we split data into training and test sets?

a. To make the dataset smaller  
b. To make the model run faster  
c. To check if the model can handle new, unseen data  
d. Because scikit-learn requires it  

<details>
<summary><strong>Show Answer</strong></summary>
c — The test set helps us evaluate if the model generalizes, not just memorizes.
</details>

---

### 3. In a regression problem, what type of value are we trying to predict?

a. A category (like cat vs. dog)  
b. A continuous value (like price or temperature)  
c. A color  
d. A True/False answer  

<details>
<summary><strong>Show Answer</strong></summary>
b — Regression predicts continuous values.
</details>

---

### 4. What does the R² score tell us?

a. How random the model is  
b. How close the model is to making perfect predictions  
c. The number of errors the model made  
d. The size of the dataset  

<details>
<summary><strong>Show Answer</strong></summary>
b — R² closer to 1 means the model fits the data well.
</details>

---

### 5. If the model performs well on training data but poorly on test data, what is happening?

a. Underfitting  
b. Overfitting  
c. Perfect learning  
d. The dataset is broken  

<details>
<summary><strong>Show Answer</strong></summary>
b — Overfitting means the model memorized the training data but cannot generalize.
</details>

---

### 🎉 Congratulations & Thank You!

Great job completing Week 2 of the Machine Learning module!  
You’ve learned ML concepts, explored real data, trained a regression model, and evaluated its performance.

Your effort, curiosity, and consistency are what make you a stronger developer each week.  
Keep going, the skills you’re building now will open many opportunities ahead.

**See you in the next lesson!**
