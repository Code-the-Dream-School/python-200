# LOGISTIC REGRESSION

When you first hear the term “Logistic Regression,” it sounds like something complicated from a math textbook. In reality, it is one of the most approachable and friendly machine learning algorithms you will ever learn. In fact, most data scientists and ML engineers start with Logistic Regression because it gives you a simple, interpretable, and surprisingly powerful way to make predictions about real-world problems. The goal of this lesson is to guide you step-by-step, so by the end you’ll feel confident explaining how Logistic Regression works without relying on formulas or heavy math.

## What problem does Logisic Regression solve? 

Let’s start with a simple question: when would someone even use Logistic Regression? Imagine you are building a system that reads emails and decides whether each one is spam or not spam. Or maybe you are trying to predict whether a customer will buy a product, whether a patient has a disease, or whether a transaction is fraudulent. All of these tasks share the same structure: there are only two possible outcomes. Logistic Regression is built exactly for this kind of problem. It is a binary classifier, meaning it tries to split the world into two groups, like yes/no, true/false, spam/not spam, or 1/0. But what makes Logistic Regression special is that it doesn’t just say “yes” or “no.” It produces a probability, which makes its predictions smoother, more interpretable, and incredibly useful in practical settings.

## THE BIG IDEA 

At its heart, logistic regression asks:
“How strongly do the input features suggest that this example belongs to class 1 instead of class 0?”

To answer this question, logistic regression starts with something familiar:
it uses the same idea as **linear regression**.

Before it becomes a classifier, it behaves like a linear model: taking all your input features, multiplying them by learned weights, adding them together, and producing a single score.

Something like this:

 ```
 z=b0​+b1​x1​+b2​x2​+…
 ```

 If this were ordinary linear regression, the model would output that number directly.

 But we can’t use raw numbers for classification. A score like 4.3 or –2.8 doesn't tell us “spam” or “not spam,” let alone the probability. So logistic regression takes this linear score, feeds it through a special transformation, and turns it into a clear probability between 0 and 1.

That transformation is the star of this algorithm: the **sigmoid function.**

## The Sigmoid Function

Right after the linear part produces a number like –5 or +7, logistic regression passes that value to the sigmoid.

![Sigmoid graph](<resources/Sigmoid graph.png>)

Now look at the curve. It’s smooth, soft, and shaped like the letter “S.” What this function does is beautifully simple:

Large negative numbers get squeezed near 0
Large positive numbers get pushed toward 1
Numbers near zero end up close to 0.5

The formula is:  ![Formula](resources/Formula.png)

but you don’t need to memorize it. What you need to understand is its *behavior*.

If the weighted sum of features produces a small value (something like –3), the sigmoid outputs a tiny probability, maybe around 0.05. If the model produces a large value (like +4), the sigmoid outputs something like 0.98. These are clean, intuitive probabilities.

This is what makes logistic regression special. You don’t just get a hard “yes/no.”
You get how confidently the model believes its answer.

A probability of 0.52 might mean “a toss-up, but slightly leaning toward class 1,” while 0.98 means “I’m almost certain.”

This probabilistic nature is why logistic regression is loved in fields like medicine, finance, and social sciences. You can interpret and explain what the model is thinking.

If you prefer to read a gentle text explanation with intuition and analogies, this Medium article breaks it down very clearly: https://medium.com/%40roseserene/back-to-basics-logistic-regression-less-math-more-intuition-e473aebcf64a

## Logistic Regression as a Classifier (The Decision Boundary)

Once we have a probability, we need to make a final decision: should this example be class 0 or class 1?

Most commonly, logistic regression uses 0.5 as a threshold:
- Probability > 0.5 → class 1
- Probability < 0.5 → class 0

But here is where logistic regression shows its structure. Because the “z” value inside the sigmoid comes from a linear combination of features, the surface separating the two classes is a straight line (or a plane, in higher dimensions). 

This means logistic regression is looking for one clean dividing boundary.

If KNN makes decisions by looking at the closest neighbors,
and Decision Trees build decisions by splitting the data into rule-based steps,
logistic regression draws one smooth linear boundary between the classes.

This makes it incredibly easy to interpret:
- If a weight is positive, that feature pushes the example toward class 1
- If it’s negative, that feature pushes it toward class 0
- The magnitude of the weight tells you how influential the feature is 

You can watch this YouTube video for another high-level, visual explanation of logistic regression and how it works: https://www.youtube.com/watch?v=yIYKR4sgzI8


### An Example Intuition (Spam Classifier)

Suppose you’re building a spam detector. Logistic regression might notice:
- Emails with lots of exclamation marks → more likely spam
- Emails with frequent business terms → less likely spam
- Emails with “free,” “credit,” or “money” → more likely spam

Each of these gets a weight:

- A positive weight means “more of this pushes toward spam”
- A negative weight means “more of this pushes toward not spam”

All these weighted pieces get added together into the “z” score. The sigmoid then converts that score into something like 0.89, meaning an 89% chance of spam.

## How Logistic Regression Learns

The model starts with random weights. It sees each training email and asks:

`“How wrong was my guess?”`

It then adjusts the weights a little bit to reduce future errors. This is done using a method called gradient descent, which is basically the model taking tiny steps downhill on an error curve.

Even without the math:
think of the model learning like a student practicing flashcards,  improving a bit every time it makes a mistake.


## When Logistic Regression is a great choice

Logistic Regression is usually a great choice when the relationship between your features and the outcome is *roughly linear*. That doesn’t mean perfect straight lines everywhere, but it does mean that as a feature increases, the likelihood of the outcome tends to increase or decrease in a steady, predictable way. In those situations, Logistic Regression fits naturally. It also shines when you want something fast to train, simple to understand, and not too computationally heavy. If your dataset isn’t enormous or if the classes are separated reasonably well in the feature space, Logistic Regression can perform surprisingly well. One of its strengths is that it gives probabilities rather than just a yes/no answer, which helps it express uncertainty in a realistic way.

But like any model, it has limitations. Logistic Regression can only draw linear boundaries between classes, which means if the data falls into swirling or curved patterns, the model will struggle unless you create new features or use polynomial transformations. It can also become unstable when your features are highly correlated with each other, which is why regularization is often helpful. Outliers can pull the decision boundary too far in one direction, and extremely imbalanced classes can cause the model to assign misleading probabilities unless you handle the imbalance with care.

Now that you understand the intuition, strengths, and weaknesses of Logistic Regression, we’re ready to put everything into action. In the next section, we’ll walk through a complete coding example where we load a dataset, train a logistic regression classifier, visualize some results, and interpret what the model learned.

## From Idea to Code: Our Logistic Regression Workflow

Before we dive into the code, let’s take a step back and look at the entire journey our data will take.

Below is a workflow diagram that shows the full pipeline of what we’re about to build using Logistic Regression - from raw data all the way to model evaluation.

You don’t need to memorize this diagram, and we are not going to explain every box right now. Instead, think of it as a *map*. As we move through the code, we’ll keep coming back to different parts of this map so you always know where you are and why you’re doing something.

### The Workflow 

![Workflow](resources/workflow.png)

Everything we do in code will follow this exact path. If at any point you feel confused later, you can come back to this diagram and ask yourself, “Which step am I in right now?” 

### Let’s Start Coding: 1. Importing the Tools We Need

We’ll begin by importing the libraries that help us work with data, visualize patterns, and build our Logistic Regression model.

As you read through this code, don’t worry if some imports feel unfamiliar. What matters is understanding why each group of tools exists. We’ll see them in action very soon.

```python
# Import Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```
NumPy helps us work with numbers, Pandas helps us work with tables of data, and Matplotlib lets us create visualizations so we can actually see what our model is doing. 

We also include a small piece of code to suppress warnings. This doesn’t change how the model works, it simply keeps the output clean and easier to read while we’re learning.

```python
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

```

Next, we import the machine learning tools from scikit-learn. This is where Logistic Regression and most of the modeling utilities come from.

```python
# Import Scikit-Learn Components
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay

```

## 2. Loading the Dataset

For this lesson, we’ll use a classic dataset for spam detection. Each row represents an email, and each feature describes something measurable about that email like how often certain words appear or how many capital letters it contains.

We start by downloading the dataset directly from an online source.

```python
import requests
from io import BytesIO

```

We also define column names so the data is easier to understand once it’s loaded.

```python
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
df.head()

```

When you run this cell, you’ll see the first few rows of the dataset.

![first_5_rows](resources/first_5_rows.png)

This is your first chance to visually confirm that the data looks reasonable and structured.

At this stage, we are still in the “**Load Dataset**” box of our workflow diagram.

## 3. Seperating Features and Target

Now we split the data into two parts:
- X: the input features (what the model learns from)
- y: the target label (spam or not spam)

```python
X = df.drop("spam_label", axis=1)
y = df["spam_label"]

```
This step is simple, but extremely important. Logistic Regression learns a relationship between X and y, so they must be clearly separated.

## 4. Protecting Ourselves from Data Leakage: Train–Test Split

Next, we split the data into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

This is one of the most important steps in machine learning.

The model will only learn from the training data. The test data is kept hidden until the very end, so we can honestly evaluate how well the model performs on unseen data.

We also use stratification to make sure the proportion of spam and non-spam emails stays consistent across both sets. This ensures a fair evaluation.

Once the split is done, the test data is off-limits until evaluation time.

## 5. Feature Scaling 

Logistic Regression relies on **weighted sums of features**. If one feature has values in the thousands and another stays between 0 and 1, the larger-scale feature can dominate unfairly.

That’s why we scale.

```python
scaler = StandardScaler()
# fit only on training data
X_train_scaled = scaler.fit_transform(X_train)
# Transform test with same scaling
X_test_scaled = scaler.transform(X_test)

```
Here we use something called `StandardScaler`, which is a very common and beginner-friendly way to scale data.

When we call `scaler.fit_transform(X_train)`, the scaler learns the statistics only from the training set. This prevents information from the test set from leaking into the model. If we were to fit the scaler on all the data, the model would indirectly “peek” at the test set, which would make our evaluation unreliable.

After fitting the scaler on `X_train`, we use the same scaler to transform both the training and test sets. This keeps them comparable while maintaining fairness.

## 6. Training the Logistic Regression

Now the model finally learns.

```python
log_reg = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)
log_reg.fit(X_train_scaled, y_train)
```
The parameters we pass in, like `max_iter` and `solver`, control how that machine learns, not the data itself.

The real learning happens when we call the `.fit()`function. In machine learning,`.fit()` is a very common function name, and it always means the same idea: *learn patterns from the training data*. When we write `log_reg.fit(X_train_scaled, y_train)`, we are explicitly telling the model, “Here are the inputs, and here are the correct answers, figure out the best way to connect them.”

Behind the scenes, Logistic Regression is learning a single linear decision boundary that separates spam from non-spam emails by combining all features.

During the process, it is learning weights (coefficients) that tell us which features push predictions toward spam and which push them away.

After .fit() completes, the model has officially learned from the training data.

## 7. Feature Importance

One of the biggest strengths of Logistic Regression is interpretability.

We can inspect the learned coefficients directly.

```python
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": log_reg.coef_[0]
})
```
When we sort these coefficients by absolute value, we can see which features had the strongest influence on the model’s decisions. Larger coefficients mean the feature pushes the prediction more strongly toward spam or non-spam.

```python
coef_df["importance"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values(by="importance", ascending=False)
coef_df.head(10)
```
The output here gives us real insight into how the model thinks, something that’s much harder to do with models like KNN or deep neural networks.

![Top_coefficients](resources/Top_coefficients.png)

From the output, we can see that `word_freq_george` has the largest absolute coefficient, making it the most influential feature in the model. This means that the presence of the word *“George”* strongly affects whether an email is predicted as spam or not, more than any other feature in this dataset. 

## 8. 1-Feature Decision Boundary(Using only the most important feature)

To build intuition, we temporarily step away from the full model and zoom in.

First, we look at the single most important feature and visualize how Logistic Regression separates spam from non-spam using just that one dimension. 

```python
# 1-Feature Decision Boundary
top_feature = coef_df.iloc[0]["feature"]
feature_index = list(X.columns).index(top_feature)

X_top_train = X_train_scaled[:, feature_index]

plt.scatter(X_top_train, y_train, alpha=0.3)
plt.axvline(0, color='red')
plt.xlabel(top_feature)
plt.ylabel("Spam")
plt.title("1-Feature Logistic Regression Decision Threshold")
plt.show()

```

We take the most important feature (word_freq_george) and plot its scaled values against the spam labels. Each dot represents one email. Emails labeled 0 are not spam, and 1 means spam.

![1-feature decision boundary](<resources/1-feature decision boundary.png>)

The red vertical line represents the **decision threshold**. On one side of this line, the model predicts “not spam.” On the other side, it predicts “spam.”

## 9. 2-Feauture Decision Boundary 

Next, we add the second most important feature and retrain a new Logistic Regression model using two features instead of one. 

```python
# 2-Feature Decision Boundary
top2_features = list(coef_df.head(2)["feature"])
indices = [list(X.columns).index(f) for f in top2_features]

X2_train = X_train_scaled[:, indices]

model2 = LogisticRegression()
model2.fit(X2_train, y_train)

DecisionBoundaryDisplay.from_estimator(
    model2,
    X2_train,
    response_method="predict",
    cmap="coolwarm",
    alpha=0.3
)

plt.scatter(X2_train[:, 0], X2_train[:, 1],
            c=y_train, cmap="coolwarm", edgecolor="k")
plt.xlabel(top2_features[0])
plt.ylabel(top2_features[1])
plt.title("Decision Boundary (Top 2 Features)")
plt.show()

```

This time, instead of a single vertical line, we get a 2D decision boundary. The background colors show which regions of the feature space are classified as spam or not spam, and each dot is still a real email.

To create this visualization, we used Scikit-Learn’s **DecisionBoundaryDisplay**, which automatically shows how the trained Logistic Regression model separates the feature space into spam and non-spam regions. This saves us from manually computing predictions and makes the decision boundary easy to interpret.

![Decision Boundary](<resources/Decision Boundary.png>)

You can now clearly see that the model has more flexibility. By combining two features, it draws a diagonal boundary that separates the classes better than before. This visually demonstrates an important idea: adding meaningful features gives the model more context and improves its decisions.

## 10. Final step: Evaluating the Model on Unseen Data

So far, everything we’ve seen used training data. Now comes the most important moment: evaluation on untouched test data.

```python
y_pred = log_reg.predict(X_test_scaled)
```

We then compute multiple evaluation metrics.

```python
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
```

Each metric tells a slightly different story about model performance, especially in a spam-detection problem where false positives and false negatives matter.

OUTPUT: 

![Output](resources/Output.png)

Now let’s interpret how well our Logistic Regression model performed on the unseen test data.

We’ll start with the confusion matrix: Out of all non-spam emails (class 0), the model correctly identified 530 emails as not spam, but 28 non-spam emails were mistakenly flagged as spam. For spam emails (class 1), the model correctly detected 326 spam messages, while 37 spam emails slipped through and were predicted as non-spam. This tells us the model is strong overall, but like any real system, it still makes a small number of mistakes in both directions.

Next, let’s look at the accuracy, which is **0.93**. This means the model correctly classified about 93% of all test emails. While accuracy gives a quick high-level view, it doesn’t tell the full story by itself especially for problems like spam detection where false positives and false negatives matter differently.

That’s why precision and recall are important. The precision score is 0.92, which means that when the model predicts an email is spam, it is correct 92% of the time. This is important because it shows the model is not overly aggressive in labeling emails as spam. 

The recall score is 0.90, meaning the model successfully catches about 90% of all actual spam emails. A small portion of spam still gets through, but most of it is detected.

The F1 score is 0.91, which balances precision and recall into a single number. This tells us that the model maintains a good trade-off between catching spam and avoiding false alarms.

The classification report breaks this down further by class. For non-spam (class 0), the model achieves 95% recall, meaning it almost always correctly recognizes legitimate emails. For spam (class 1), the recall is 90%, which is still strong and expected in real-world spam filtering. The weighted averages across both classes remain around 0.93, showing consistent performance across the dataset.

Overall, these results show that Logistic Regression performs very well on this problem. It learns a clean linear boundary, provides strong predictive performance, and most importantly gives us interpretable metrics that help us understand exactly how and where the model succeeds or fails.

## Check for Understanding: 

1. What is the main role of the sigmoid function in Logistic Regression?

- A. To scale features between −1 and 1
- B. To convert a linear combination of features into a probability between 0 and 1
- C. To select the most important features
- D. To draw multiple decision boundaries
<details> <summary><strong>Click to reveal answer</strong></summary>
Correct answer: B
</details>

2. Why do we fit the StandardScaler only on the training data and not on the test data?
A. To make training faster
B. To reduce model complexity
C. To prevent information leakage from the test set
D. To increase accuracy artificially
<details> <summary><strong>Click to reveal answer</strong></summary>
Correct answer: C
Fitting on the test set would leak future information into the model, making evaluation unfair.
</details>


## Lesson Wrap up: 

Well done 🎉 Congratulations on completing the Logistic Regression lesson!
In this lesson, we built a full intuition for Logistic Regression, from how it turns linear combinations of features into probabilities using the sigmoid function, to how it learns a single, interpretable decision boundary for binary classification. We trained a real spam classifier end-to-end, explored feature importance to see how the model “thinks,” visualized decision boundaries with one and two features, and evaluated performance using accuracy, precision, recall, F1 score, and the confusion matrix. Along the way, we emphasized best practices like train–test splitting, scaling correctly to avoid data leakage, and interpreting results rather than treating the model as a black box.

 **Next up:** Now that we understand how machine learning works with structured, numeric data, we’ll move into Computer Vision, where models learn from images and pixels instead of rows and columns.

