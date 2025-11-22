# ML: Introduction to Data Preprocessing and Feature Engineering
Machine learning algorithms expect data in a clean, numeric, consistent format, but as we have seen in Python 100, real datasets rarely arrive that way. Preprocessing makes features easier for models to learn from and is required for most `scikit-learn` workflows.

This lesson will be a review of many concepts from Python 100, which are secretly very important for Machine Learning. We will cover:

- Numeric vs categorical features
- Feature scaling (standardization, normalization)
- Encoding categorical variables 
- Creating new features
- How to do all of this with scikit-learn transformers and pipelines

This lesson prepares you for next week’s classifiers (KNN, logistic regression, decision trees).

## 1. Numeric vs Categorical Features

[Video on numeric vs categorical (and more!)](https://www.youtube.com/watch?v=rodnzzR4dLo)

Before we can train a classifier, we need to understand the kind of data we are giving it. Machine learning models only know how to work with _numbers_, so every feature in our dataset eventually has to be represented numerically. Different types of features require different kinds of preprocessing, which is why this distinction matters right at the start.

**Numeric features** are things that are already numbers: age, height, temperature, income. They are typically represented as floats or ints in Python. Models generally work well with these, but many algorithms still need the numbers to be put on a similar scale before training. We will cover scaling next.

**Categorical features** describe types or labels instead of quantities. They are often represented as `strings` in Python: city name, animal species, shirt size. These values mean something to _us_, but such raw text is not useful to a machine learning model. We will need to convert these categories into a numerical format that a classifier can learn from. That is where things like one-hot encoding come in (which we will cover below). Even large language models do not work with strings: as we will see in future weeks when we cover AI, linguistic inputs must be converted to numerical arrays before large language models can get traction with them. 

> Most categorical features have no natural order (dog, cat, bird; red, green, blue). These are known as *nominal* categories, and one-hot encoding works perfectly for them. Some categories do have an order (`small` < `medium` < `large`). These are known as *ordinal* categories. For these, the ordering matters but the spacing does not: medium is not "twice" small. In practice, ordinal features often need extra thought. Sometimes an integer mapping is fine; sometimes one-hot encoding is still safer. There is no universal answer for how to answer ordinal categories. 

## 2. Scaling Numerical Features

[Video overview of feature scaling](https://www.youtube.com/watch?v=dHSWjZ_quaU)

When we have data in numerical form, we might think we are all set to feed it to a machine learning algorithm. However, this is not always true. Even though numeric features are already numbers, we still have to think about how they behave in a machine learning model. Many algorithms do not just look at the numberical features themselves, but at how large they are relative to each other. If one feature uses much bigger units than another, the model may unintentionally focus on the bigger one and ignore the smaller one.

For example, imagine a dataset with two numeric features:

- age (18 to 70)
- income (25,000 to 180,000)

Both features matter, but income is measured in much larger units. Many ML algorithms will treat the income differences as more important than the age differences simply because the numbers are bigger. The model is not being clever here, it is just reacting to scale.

Scaling helps put numeric features on a similar footing so that models can consider them more fairly.

## Normalization (Min-Max Scaling)
Normalization, aka min-max scaling, rescales each feature so that it falls into the range [0, 1]. This helps ensure that no feature overwhelms another just because it uses larger numbers.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```
Now each column of X will have values that fall into the desired range. 

## Standardization
Another common approach is standardization, which transforms each numeric feature so that:

- Its mean becomes 0
- Its standard deviation becomes 1

This keeps the general shape of the data but puts all features on comparable units. 

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
These standardized units are also known as *z-scores*. 

### When scaling matters

Scaling is especially important for algorithms that use distances or continuous optimization algorithms to minimize errors, such as:

- KNN
- Logistic regression
- Neural networks

Some models are much less sensitive to scale:

- Decision trees and random forests

Even numeric features require thoughtful preparation. Scaling helps many models learn fairly from all features instead of being overwhelmed by a few large numbers.

### Standardization example 
To make this concrete, let us look at the distributions of two numeric features:

- `age` (in years)  
- `income` (in dollars)  

First we will plot them separately on their original scales. Then we will standardize them and plot both together to see how they compare when measured in standardized units (z score):

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Synthetic data: age (years) and income (dollars)
age = np.array([22, 25, 30, 35, 42, 50, 60])
income = np.array([28000, 32000, 45000, 60000, 75000, 90000, 150000])

X = np.column_stack([age, income])

# Scale using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

age_scaled = X_scaled[:, 0]
income_scaled = X_scaled[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: original values
axes[0].scatter(age, income)
axes[0].set_title("Original data")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Income")

# Right: scaled values
axes[1].scatter(age_scaled, income_scaled)
axes[1].set_title("Scaled data")
axes[1].set_xlabel("Age (standardized)")
axes[1].set_ylabel("Income (standardized)")

plt.tight_layout()
plt.show()
```
On the first two plots, you can see that age and income are on completely different numeric scales. On the bottom plot, after standardization, both features live in the same z-score space and can be compared directly.

A z-score tells you how many standard deviations a value is above or below the mean of that feature:

- z = 0 means "right at the average"
- z = 1 means "one standard deviation above average"
- z = -2 means "two standard deviations below average"

So a negative age or income after standardization does not mean a negative age or negative dollars. It simply means that value is below the average for that feature.

## 3. One-Hot Encoding for Categorical Features

[Video about one-hot encoding](https://www.youtube.com/watch?v=G2iVj7WKDFk)

Categorical features (like "dog", "cat", "bird") must be converted into numbers before a machine learning model can use them. But we cannot simply assign numbers like:

```
'dog' -> 1
'cat' -> 2
'bird' -> 3
```

If we did this, the model would think that "cat" is somehow bigger than "dog", or that the distance between categories carries meaning. That is not true. These numbers would create a false ordering that does not exist in the real categories.

To avoid this, we use one-hot encoding. One-hot encoding represents each category as an array where:

- all elements are 0
- except for one element, which is 1
- the position of the 1 corresponds to the category

So the categories become:

```
dog  -> [1, 0, 0]
cat  -> [0, 1, 0]
bird -> [0, 0, 1]
```

Each category is now represented cleanly, without implying any ordering or distance between them. This is exactly what we want for most categorical features in classification.

### One-hot encoding in scikit-learn

Because this step is so common, scikit-learn has a built-in one-hot encoder.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

X = [["dog"], ["cat"], ["bird"], ["dog"]]
X_encoded = encoder.fit_transform(X)

print("one-hot encoded features:")
print(X_encoded.toarray())
```

Output:

```
one-hot encoded features:
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]]
```

Note: You may see the output as a sparse matrix. Calling `.toarray()` converts it into a plain NumPy array so you can print or inspect it easily.

We will see more practical examples of one-hot encoding in future lessons. 

# 4. Creating New Features (Feature Engineering)

[Video overview of feature engineering](feature engineering vid)

Sometimes the data we start with is not the data that will help a model learn best. A big part of machine learning is noticing when you can create new data, or new features, that capture something useful about the data. This is called *feature engineering*, and it can make a massive difference in how well a classifier performs.

You have already learned about this idea in Python 100 in the context of your lessons about Pandas dataframes (you created new columns from existing columns). Here we revisit the idea with an ML mindset: Can we create features that make patterns easier for the model to see?

Below are a few common and intuitive examples.

## Combining features

Sometimes two columns together make a more meaningful measurement than either one alone. For example, height and weight individually may sometimes combine to form a more useful feature than either feature alone, into BMI:

```python
df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
```

A classifier may learn more easily from BMI than from raw height and weight (for instance when predicting heart disease risk).

## Extracting parts of a feature

If you have a datetime column, you can often pull out the pieces that matter for prediction:

```python
df["weekday"] = df["date"].dt.weekday  
```

A model might not care about the full timestamp, but it might care about whether something happened on a weekday or weekend.

## Binning continuous values

Sometimes a numeric feature is easier for a model to understand if we convert it into categories. For example, instead of raw ages, we can group them into age *groups*:

```python
df["age_group"] = pd.cut(df["age"], bins=[0, 18, 30, 50, 80])
```

This can help when the exact number is less important than the general range.

## Boolean features

A simple yes/no feature can sometimes capture a pattern that the raw values obscure. For example:

```python
df["is_senior"] = (df["age"] > 65).astype(int)
```

Even though age is already numeric, the idea of "senior" might be more directly meaningful for the model.

## Final thoughts on feature engineering
There are no strict rules for feature engineering. It is a creative part of machine learning where your intuitions and understanding of the data matters a great deal. Good features often come from visualizing your data, looking for patterns, and thinking about the real-world meaning behind each column. Domain-specific knowledge helps a lot here: someone who knows the problem well can often spot new features that make the model's job easier. As you work with different datasets, you will get better at recognizing when a new feature might capture something important that the raw inputs miss. Feature engineering is less about following a checklist and more about exploring, experimenting, and trusting your intuition as it develops.