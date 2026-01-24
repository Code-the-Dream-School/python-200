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

### What the Tree Learns From This

A decision tree tries many possible questions and evaluates each one by checking how much it reduces the Gini impurity. The tree always chooses the question that creates the **purest groups overall**. In other words, it looks for splits that turn messy, uncertain data into cleaner, more predictable pieces.

### Key Intuition

Decision trees do not understand emails, words, or meaning the way humans do. Instead, they repeatedly ask simple yes-or-no questions that make the data less mixed. By continuously reducing Gini impurity, the tree builds a structure that becomes more confident in its predictions at each step.
This intuition will later help explain both the strengths of decision trees and why combining many trees in a **Random Forest** leads to better performance.

---


## Dataset: Spambase — Learning From Real Emails

So far, we’ve worked with clean, well-structured datasets like Iris. Now we take our **next big step** toward real-world machine learning.
In this lesson, we use the **Spambase dataset**, which is built from **real emails**. Each row represents **one email message**, and our goal is simple: **Can we tell whether an email is spam or not based on how it looks?** This is a realistic and motivating problem — it’s the same idea behind the spam filters you use every day.


## What Do the Columns Actually Mean?

At first glance, this dataset looks intimidating: just lots of numbers. But these numbers are *not arbitrary* — each one measures something meaningful about an email. Each column captures a **specific signal**, such as:

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

This dataset is especially useful because it is:
- High-dimensional (many features)
- Much closer to real-world machine learning problems
- It forces us to interpret features and results carefully — not just trust accuracy scores.

---

## Setup

Before loading the data, we import the tools we’ll use. Don’t worry if some of these feel unfamiliar — we’ll use them step by step.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
````
---

## Loading the Dataset (Step by Step)

The Spambase dataset lives online at the UCI Machine Learning Repository. We download it directly and load it into a pandas DataFrame.

```python
from io import BytesIO
import requests

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()
df = pd.read_csv(BytesIO(response.content), header=None)
```

**What just happened?**

We downloaded the raw data file.Loaded it into a table structure (a DataFrame). Told pandas there is **no header row**, because the dataset stores only numbers. At this stage, it’s completely normal if the data feels overwhelming.

---

## First Look at the Data

Before building *any* model, we stop and explore.

```python
print(df.shape)
df.head()
```

<img width="940" height="245" alt="Screenshot 2026-01-24 at 6 05 37 PM" src="https://github.com/user-attachments/assets/47d4f94f-0215-466e-bd90-129c87566e6a" />

**Image Credits: Google Colab**

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

<img width="211" height="187" alt="Screenshot 2026-01-24 at 6 06 39 PM" src="https://github.com/user-attachments/assets/7218b9c3-db19-46b4-9c76-3349550b8daf" />

**Image Credits: Google Colab**

This tells us how many emails are spam versus not spam. Seeing both classes well represented is important — it means our models will get to learn from **both types of emails**, not just one.

To make this even clearer, let’s visualize it.
```python
counts = df.iloc[:, -1].value_counts()
plt.bar(["Not Spam (0)", "Spam (1)"], counts.sort_index())
plt.title("Spam vs Not Spam Counts")
plt.xlabel("Email Type")
plt.ylabel("Number of Emails")
plt.show()
```
Seeing the balance visually makes it easier to trust later accuracy and F1 scores.

<img width="711" height="481" alt="Screenshot 2026-01-24 at 6 16 46 PM" src="https://github.com/user-attachments/assets/382b4968-80aa-45a2-a2d4-520fabdde75d" />

**Image Credits: Google Colab**

---

### 2. Capital Letters as a Spam Signal

One feature measures how intense capital letter usage is in an email. Spam messages often use ALL CAPS to grab attention.

```python
plt.hist(df.iloc[:, 54], bins=30)
plt.title("Capital Letter Run Length Average")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

<img width="618" height="451" alt="Screenshot 2026-01-24 at 6 07 36 PM" src="https://github.com/user-attachments/assets/bbe77c31-4d1f-42ef-b576-44e4d8911321" />

**Image Credits: Google Colab**

Because a few emails have extremely large values, the histogram is heavily right-skewed. This compresses most of the data near zero, but that skew itself is a strong spam signal. Most emails have low values, meaning normal capitalization. A small number have very high values — these are often aggressive, spam-like emails. This is a great example of how a **simple numeric feature can encode human behavior**.

---

### 3.Comparing Spam vs Non-Spam for One Feature

A boxplot helps us compare how a feature behaves across classes.

```python
temp = df[[54, 57]].copy()
temp.columns = ["cap_run_avg", "is_spam"]

plt.boxplot(
    [
        temp[temp["is_spam"] == 0]["cap_run_avg"],
        temp[temp["is_spam"] == 1]["cap_run_avg"]
    ],
    labels=["Not Spam (0)", "Spam (1)"]
)
plt.title("Capital Letter Run Length Avg: Spam vs Not Spam")
plt.ylabel("Value")
plt.show()
```
<img width="678" height="444" alt="Screenshot 2026-01-24 at 6 18 32 PM" src="https://github.com/user-attachments/assets/070c02b1-f76f-4d3e-8310-2581100c41df" />

**Image Credits: Google Colab**

**What to notice:**
Spam emails tend to have higher and more extreme capitalization values than non-spam emails. This is exactly the kind of rule a decision tree can learn naturally.

---

### 4. How Do Features Relate to Each Other?

Next, let’s check whether some features move together. We’ll keep this simple by looking at just the first 12 features.

```python
subset = df.iloc[:, :12]
corr = subset.corr()

plt.imshow(corr)
plt.title("Feature Correlation Heatmap (First 12 Features)")
plt.colorbar()
plt.xticks(range(12), range(12))
plt.yticks(range(12), range(12))
plt.show()
```

<img width="536" height="449" alt="Screenshot 2026-01-24 at 6 19 51 PM" src="https://github.com/user-attachments/assets/30b4cc91-c7bc-42c6-8e4f-3cb6ad5264b4" />

**Image Credits: Google Colab**

Correlation patterns like these help explain: Why some features may be redundant and Why models that rely on rules (trees) can outperform distance-based models (KNN).

---

### 5. Which Features Vary the Most?

Some features barely change across emails. Others vary wildly — and those often carry stronger signals.

```python
subset = df.iloc[:, :12]
variances = df.iloc[:, :-1].var().sort_values(ascending=False)
top10 = variances.head(10)

plt.bar(range(len(top10)), top10.values)
plt.title("Top 10 Most Variable Features")
plt.xlabel("Feature Index")
plt.ylabel("Variance")
plt.xticks(range(len(top10)), top10.index)
plt.show()
```
<img width="696" height="447" alt="Screenshot 2026-01-24 at 6 21 37 PM" src="https://github.com/user-attachments/assets/5b0483c2-ad09-4101-a067-1f43f92adda6" />

**Image Credits: Google Colab**

High-variance features are often more informative because they capture real behavioral differences between emails.

---

### 6. Interpreting Features Like a Human

At this point, it’s important to pause and reflect.

These features are: Not spatial, Not physical measurements, Independent behavioral signals, Each column measures one aspect of how an email was written.

This explains why:

- Distance-based models struggle

- Rule-based models shine

- Trees feel very natural for this problem

### Why EDA Comes Before Modeling

Before we train any classifiers, EDA helps us answer:

- What does each column actually represent?

- Which features look meaningful?

- Does the data reflect real-world behavior?

Without this step, models become black boxes and accuracy numbers lose their meaning. Now that we understand the data, we’re finally ready to model it.

---

## Our First Decision Tree

Now that we understand the data and what the features represent, we are ready to try our **first rule-based model**. Unlike KNN, which compares distances, a **decision tree asks questions** like:

> “Is capital letter usage unusually high?”
> “Does this email contain certain word patterns often seen in spam?”

Let’s see if this kind of reasoning actually works on our data.

## Step 1: Create the Model

We start by creating a single **Decision Tree classifier**. At this stage, we use all default settings. Our goal is not to tune or optimize — it’s simply to see how well a tree can learn from the data.

```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
```
---

## Step 2: Train the Tree

Next, we fit the model using the training data. For a decision tree, this means:
* trying many possible yes/no questions,
* choosing the ones that best separate spam from non-spam and building a sequence of rules.

```python
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0,
    stratify=y
)

tree.fit(X_train, y_train)
```
---

## Step 3: Make Predictions on New Emails

Now we ask the tree to classify emails it has **never seen before**.
```python
pred_tree = tree.predict(X_test)
```
Each prediction follows a path through the tree, answering questions about features like capitalization, word frequency, and symbol usage.

---

## Step 4: Evaluate the Results

Finally, we check how well the tree performed.

```python
print(classification_report(y_test, pred_tree, digits=3))
```

<img width="556" height="166" alt="Screenshot 2026-01-24 at 6 34 16 PM" src="https://github.com/user-attachments/assets/1569cffc-a19c-4beb-85a4-d59ec6535673" />

**Image Credits: Google Colab**

You should see **strong performance**, often noticeably better than KNN on this dataset.

---

## What Just Happened?

Even with a single tree and no tuning, the model performs well because each feature represents a meaningful email behavior. The tree examines one feature at a time and builds rules that closely match how spam is actually written, such as aggressive capitalization or specific word patterns. The table summarizes how well our single decision tree performed on emails it has never seen before. The overall accuracy is about 92%, meaning the model correctly classified roughly 9 out of every 10 emails. That’s a strong result for a simple, interpretable model.

Looking class by class:

Not Spam (0):
Precision and recall are both around 93%, which means the tree is very good at recognizing legitimate emails and rarely mislabeling them as spam.

Spam (1):
Precision and recall are just under 90%, meaning the model successfully catches most spam emails, though a small number still slip through or get misclassified. The F1-scores (which balance precision and recall) are high for both classes, showing that the tree performs well overall and not just on one category.

---

## Key Takeaway So Far

> A decision tree can learn **human-interpretable rules** directly from data
> and apply them effectively to real-world problems like spam detection.

Before moving on, take a moment to appreciate this: You’ve now trained **two very different classifiers** and seen how data structure affects model performance. Next, we’ll explore **why this powerful approach has a weakness** — and how Random Forests solve it.
