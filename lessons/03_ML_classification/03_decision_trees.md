# Lesson 3  
## CTD Python 200  
### Decision Trees and Ensemble Learning (Random Forests)

---

## Before We Begin: Optional Learning Resources

If you’d like extra intuition before or after this lesson, these resources are very helpful:

**Text (Displayr – very intuitive):**  
https://www.displayr.com/what-is-a-decision-tree/

**Video (StatQuest, ~8 minutes):**  
https://www.youtube.com/watch?v=7VeUPuFGJHk  

Seeing trees explained visually makes everything that follows much easier to understand.

---

## From Distance to Decisions

In the previous lesson, we learned **K-Nearest Neighbors (KNN)**. KNN makes predictions by measuring *distance* between data points. That works well for small, clean datasets like Iris. But many real-world problems don’t behave like points in space.

Now imagine how *you* decide whether an email is spam:

> “Does the email contain lots of dollar signs?”  
> “Does it use words like *free* or *winner*?”  
> “Are there long blocks of capital letters?”

This style of reasoning — asking a **sequence of yes/no questions** — is exactly how a **Decision Tree** works.

![how_decision_trees_work_](https://github.com/user-attachments/assets/6d8c032d-2579-47e9-a1cc-bc571268c209)


**Image credit:** Geeks For Geeks

Decision trees are powerful because they resemble **human decision-making**. They feel like flowcharts rather than equations.

---

## What Is a Decision Tree?

A decision tree repeatedly asks questions like:

> “Is this feature greater than some value?”

Each question splits the data into smaller and more focused groups. Eventually, the tree reaches a **leaf**, where it makes a prediction.

Unlike KNN:
- Trees do **not** use distance
- Trees do **not** need feature scaling
- Trees work very well with real-world tabular data

A decision tree makes predictions based on a series of questions. The outcome of each question determines which branch of the tree to follow. They can be constructed manually (when the amount of data is small) or by algorithms, and are naturally visualized as a tree. 
The decision tree is typically read from top (root) to bottom (leaves). A question is asked at each node (split point) and the response to that question determines which branch is followed next. The prediction is given by the label of a leaf.

The diagram below shows a decision tree which predicts how to make the journey to work.

<img width="577" height="335" alt="Screenshot 2026-01-24 at 5 35 29 PM" src="https://github.com/user-attachments/assets/cd1cbe6a-24a2-453d-82e0-210a27c21922" />

**Image credit:** Displayr

The first question asked is about the weather. If it’s cloudy, then the second question asks whether I am hungry. If I am, then I walk, so I can go past the café. However, if it’s sunny then my mode of transport depends on how much time I have.

The responses to questions and the prediction may be either:

Binary, meaning the response is yes/no or true/false as per the hungry question above
Categorical, meaning the response is one of a defined number of possibilities, e.g. the weather question
Numeric, an example being the time question

---

## How a Decision Tree Works

### A Simple Example: Deciding What to Eat for Lunch

Imagine you’re trying to decide **what to eat for lunch** each day.

You don’t calculate distances or probabilities.
Instead, you ask yourself a sequence of simple questions.

For example:

> “Do I have a lot of time?”
> “Am I very hungry?”
> “Do I want something healthy?”

Based on your answers, you end up choosing something like *salad*, *sandwich*, or *takeout*.

Now imagine we collect a small dataset of past lunch decisions.

| Lunch Choice | Time Available | Very Hungry | Healthy Mood |
| ------------ | -------------- | ----------- | ------------ |
| Takeout      | Little         | Yes         | No           |
| Salad        | Plenty         | No          | Yes          |
| Sandwich     | Little         | Yes         | Yes          |
| Takeout      | Little         | No          | No           |
| Salad        | Plenty         | Yes         | Yes          |

Each row is an **example**:

* The **predictor variables** are the questions (time, hunger, health mood)
* The **outcome** is the lunch choice

---

## How a Decision Tree Learns from This Data

In real life, we usually *don’t know the rules ahead of time*.
Instead, the decision tree **learns the rules from examples** like the table above.

Conceptually, the tree is built like this:

1. Start with all lunch examples together at the top (the root).
2. Ask: *Which question best separates the lunch choices?*
   For example: “Do I have plenty of time?”
3. Split the data based on the answer (Yes / No).
4. For each group, ask another question that improves separation.
5. Stop once a group mostly leads to the same lunch choice.

At the end, the tree might represent rules like:

> “If I have little time and I’m very hungry → Takeout”
> “If I have plenty of time and want to eat healthy → Salad”

These rules were **not programmed by hand** — they were discovered from data.

---

## Why This Works So Well

Decision trees are powerful because they:

* Mimic how humans naturally make decisions
* Turn data into understandable rules
* Can model complex behavior using simple yes/no questions

However, there’s a catch.

Because trees are so flexible, they can sometimes learn **noise** instead of real patterns.
That’s why, later, we introduce **Random Forests**, which combine many trees to make more reliable decisions.

For now, the key idea is this:

> A decision tree learns a sequence of questions that best explains the examples it sees.

That’s the core intuition — everything else builds on that.

---

## Gini Impurity (How a Decision Tree Measures “Messiness”)

The Gini index measures how mixed the classes are in a group of data. A low Gini value means the group is mostly one class (more certain), while a high Gini value means the group is mixed (more uncertain). Decision trees choose splits that reduce the Gini index the most, creating purer groups at each step. Think of each group of data as a bar showing how mixed it is.
To make this idea more concrete, imagine visualizing each group as a bar showing how much spam and not-spam it contains.

### Before Any Split (Highly Mixed Group)

```

Spam      ██████████
Not Spam  ██████████

```
Before any split, the emails are evenly mixed between spam and not spam. In this situation, the model has very little confidence. If you randomly pick an email from this group, it could easily belong to either class. This corresponds to **high Gini impurity**.

---

### After a Good Split (More Pure Groups)

```

Group A:
Spam      ████████████████
Not Spam  ██

Group B:
Spam      ██
Not Spam  ████████████████

```

After the tree asks a good question, the data separates into smaller groups. One group is now mostly spam, while the other is mostly not spam. These groups are much easier to classify because one class clearly dominates in each. This means the **Gini impurity is much lower** after the split.

---

### What the Tree Learns From This

A decision tree tries many possible questions and evaluates each one by checking how much it reduces the Gini impurity. The tree always chooses the question that creates the **purest groups overall**. In other words, it looks for splits that turn messy, uncertain data into cleaner, more predictable pieces.

---

### Key Intuition

Decision trees do not understand emails, words, or meaning the way humans do. Instead, they repeatedly ask simple yes-or-no questions that make the data less mixed. By continuously reducing Gini impurity, the tree builds a structure that becomes more confident in its predictions at each step.
This intuition will later help explain both the strengths of decision trees and why combining many trees in a **Random Forest** leads to better performance.

---

## Dataset: Spambase — Learning From Real Emails

So far, we’ve worked with clean, well-structured datasets like Iris.
Now we take our **next big step** toward real-world machine learning.

In this lesson, we use the **Spambase dataset**, which is built from **real emails**.
Each row represents **one email message**, and our goal is simple:

> **Can we tell whether an email is spam or not based on how it looks?**

This is a realistic and motivating problem — it’s the same idea behind the spam filters you use every day.

---

## What Do the Columns Actually Mean?

At first glance, this dataset looks intimidating: just lots of numbers.
But these numbers are *not arbitrary* — each one measures something meaningful about an email.

Each column captures a **specific signal**, such as:

- How often certain words appear  
  (for example: `"free"`, `"credit"`, `"remove"`, `"your"`)
- How often special characters appear  
  (such as `"!"` or `"$"`)
- Patterns of **capital letter usage**, which is common in spam  
  (for example: emails written in ALL CAPS)

The **final column** is the label:
- `1` → spam  
- `0` → not spam (often called *ham*)

So instead of thinking:

> “These are just abstract numbers”

think:

> “Each number describes a behavior of an email sender”

That perspective is essential for understanding what the model is learning.

---

## Why Spambase Is a Good Learning Dataset

This dataset is especially useful because it is:

- Messier than Iris  
- High-dimensional (many features)
- Much closer to real-world machine learning problems

It forces us to interpret features and results carefully — not just trust accuracy scores.

---

## Setup

Before loading the data, we import the tools we’ll use.
Don’t worry if some of these feel unfamiliar — we’ll use them step by step.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
````

---

## Loading the Dataset (Step by Step)

The Spambase dataset lives online at the UCI Machine Learning Repository.
We download it directly and load it into a pandas DataFrame.

```python
from io import BytesIO
import requests

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
```

What just happened?

* We downloaded the raw data file
* Loaded it into a table structure (a DataFrame)
* Told pandas there is **no header row**, because the dataset stores only numbers

At this stage, it’s completely normal if the data feels overwhelming.

---

## First Look at the Data

Before building *any* model, we stop and explore.

```python
print(df.shape)
df.head()
```

You should see:

* **4,601 emails** (rows)
* **58 columns**

  * 57 feature columns
  * 1 label column (spam or not spam)

This confirms that we’re working with a moderately large, real dataset.

---

## Exploratory Data Analysis (EDA): Making the Numbers Meaningful

EDA is where we start turning numbers into understanding.

### 1. Are Spam and Non-Spam Balanced?

We first check how many emails fall into each class.

```python
df.iloc[:, -1].value_counts()
```

This tells us how many emails are spam versus not spam.

Seeing both classes well represented is important — it means our models will get to learn from **both types of emails**, not just one.

---

### 2. Capital Letters as a Spam Signal

One feature measures how intense capital letter usage is in an email.
Spam messages often use ALL CAPS to grab attention.

```python
plt.hist(df.iloc[:, 54], bins=30)
plt.title("Capital Letter Run Length Average")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

Most emails have low values, meaning normal capitalization.
A small number have very high values — these are often aggressive, spam-like emails.

This is a great example of how a **simple numeric feature can encode human behavior**.

---

### 3. Interpreting Features Like a Human

At this point, it’s important to pause and reflect:

* These features are **not spatial**
* They don’t represent physical distance
* Each column measures a different behavior

This explains why some models struggle and others perform well.
Understanding the *nature of the features* helps us understand the models later.

---

### 4. Why EDA Comes Before Modeling

Before we train anything, EDA helps us answer:

* What does each feature represent?
* Are there strong signals?
* Does the data reflect real-world behavior?

Without this step, models become black boxes and accuracy numbers lose meaning.

---

### 5. Big Picture So Far

At this stage, you should understand:

* What one row represents (an email)
* What columns represent (email behaviors)
* What the label means (spam or not spam)

Only **after** this foundation is built does it make sense to train classifiers.

---
## Looking Ahead

Next, we will:

* Tune tree depth and forest size
* Use cross-validation
* Interpret feature importance
* Discuss real-world trade-offs

---

### 🚀 You’ve just taken a major step toward real-world machine learning.

```

---
