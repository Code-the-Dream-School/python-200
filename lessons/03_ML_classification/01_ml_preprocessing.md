# ML: Introduction to Data Preprocessing and Feature Engineering
Machine learning algorithms expect data in a clean, numeric, consistent format, but as we have seen in Python 100, real datasets rarely arrive that way. Preprocessing makes features easier for models to learn from and is required for most `scikit-learn` workflows.

This lesson will be a review of many concepts from Python 100, which are secretly very important for Machine Learning. We will cover:

- Numeric vs categorical features
- Feature scaling (standardization, normalization)
- Encoding categorical variables (one hot encoding)
- Creating new features (feature engineering)
- Dimensionality reduction and Principal component analysis

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

## 4. Creating New Features (Feature Engineering)

[Video overview of feature engineering](feature engineering vid)

Sometimes the data we start with is not the data that will help a model learn best. A big part of machine learning is noticing when you can create new data, or new features, that capture something useful about the data. This is called *feature engineering*, and it can make a massive difference in how well a classifier performs.

You have already learned about this idea in Python 100 in the context of your lessons about Pandas dataframes (you created new columns from existing columns). Here we revisit the idea with an ML mindset: Can we create features that make patterns easier for the model to learn?

To make this concrete, let’s create a tiny dataframe with ten fictional people:

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Ana","Ben","Cara","Dev","Elle","Finn","Gia","Hank","Ivy","Jules"],
    "weight_kg": [55, 72, 68, 90, 62, 80, 70, 95, 50, 78],
    "height_m": [1.62, 1.80, 1.70, 1.75, 1.60, 1.82, 1.68, 1.90, 1.55, 1.72],
    "birthdate": pd.to_datetime([
        "1995-04-02","1988-10-20","2001-01-05","1990-07-12","1999-12-01",
        "1985-05-22","1993-09-14","1978-03-02","2004-11-18","1992-06-30"
    ])
})

df
```
This gives us:
```
    name  weight_kg  height_m  birthdate
0    Ana         55      1.62 1995-04-02
1    Ben         72      1.80 1988-10-20
2   Cara         68      1.70 2001-01-05
3    Dev         90      1.75 1990-07-12
4   Elle         62      1.60 1999-12-01
```
We will apply common types of feature engineering to this dataframe to illustrate the concepts.

### Combining features

Sometimes two columns together make a more meaningful measurement than either one alone. For example, height and weight individually may sometimes combine to form a more useful feature than either feature alone, into BMI:

```python
df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2)
```

A classifier may learn more easily from BMI than from raw height and weight (for instance when predicting heart disease risk).

### Extracting parts of a feature

If you have a datetime column, you can often pull out the pieces that matter for prediction:

```python
df["weekday"] = df["date"].dt.weekday  
df["birth_year"] = df["birthdate"].dt.year
```

A model might not care about the full timestamp, but it might care about whether something happened on a weekday or weekend. This might matter for costs of healthcare, for instance.

### Binning continuous values

Sometimes a numeric feature is easier for a model to understand if we convert it into categories. For example, instead of raw ages, we can group them into age *groups*:

```python
current_year = 2025
df["age"] = current_year - df["birth_year"]

df["age_group"] = pd.cut(df["age"], bins=[0, 20, 30, 40, 60], labels=["young","20s","30s","40+"])
df[["name","age","age_group"]]
```
This can help when the exact number is less important than the general range.

### Boolean features

A simple yes/no feature can sometimes capture a pattern that the raw values obscure. For example:

```python
df["is_senior"] = (df["age"] > 65).astype(int)
df[["name","age","is_senior"]]
```

Even though age is already numeric, the idea of "senior" might be more directly meaningful for a model (for instance if you are thinking about pricing for restaurants).

### Final thoughts on feature engineering
There are no strict rules for feature engineering. It is a creative part of machine learning where your intuitions and understanding of the data matters a great deal. Good features often come from visualizing your data, looking for patterns, and thinking about the real-world meaning behind each column. Domain-specific knowledge helps a lot here: someone who knows the problem well can often spot new features that make the model's job easier. As you work with different datasets, you will get better at recognizing when a new feature might capture something important that the raw inputs miss. Feature engineering is less about following a checklist and more about exploring, experimenting, and trusting your intuition as it develops.


## 5. Dimensionality reduction and PCA
Many real datasets contain far more dimensions, or features, than we truly need (in pandas, represented by columns in a dataframe). Some features are almost duplicates of each other, or they carry very similar information -- this is known as *redundancy*. When our feature space gets large, models can become slower, harder to interpret, harder to fit to data, and become prone to overfitting. Dimensionality reduction is a set of techniques that help us simplify a dataset by creating a smaller number of informative features. 

As discussed previously (add link), one helpful way to picture this is to think about images. A high-resolution photo might have millions of pixels, but you can shrink it down to a small thumbnail and still recognize the main shape and structure. You will lose some detail, but you keep the big picture. Dimensionality reduction works the same way for datasets: the goal is to keep the important structure while throwing away the noise and redundancy. ML algorithms, and visualization tools can work while throwing away a great deal of raw data, and this can speed things up tremendously. 

We see redundancy all the time in real data. For example, if a dataset includes height, weight, and BMI, one of these is technically redundant because BMI is literally just a function of the other two: if you calculated BMI, then you might want to get rid of weight and height if you are estimating certain health risks. Machine learning models can still train with redundant features, but it is often helpful to compress the dataset into a smaller number of non-overlapping dimensions (features). 

Dimensionality reduction can be a helpful visualization tool (we will demonstrate this below) help fight overfitting, and can eliminate noise from our data. We saw last week that overfitting comes from model complexity (a model with too many flexible parameters can memorize noise in the training set). However, high-dimensional data can make overfitting more likely because it gives the model many opportunities to chase tiny, meaningless variations. Reducing feature dimensionality can sometimes help by stripping away that noise and highlighting the core structure the model should learn.

### Principal component analysis (PCA)
Before moving on, consider watching the following introductory video on PCA:
[PCA concepts](https://www.youtube.com/watch?v=pmG4K79DUoI)

PCA is the most popular dimensionality reduction technique: it provides a way to represent numerical data using much fewer features (dimensions), which helps us visualize extremely complex datasets. It also can help as a preprocessing step. 

A helpful way to understand PCA is to return to the image example. A raw image may have millions of pixel values, but many of those pixels move together. Nearby pixels tend to be highly correlated: if a region of the image is bright, most pixels in that region will be bright too. This means the dataset looks very high dimensional on paper, but the underlying structure is much simpler. As you know from resizing images on your phone, you can shrink an image dramatically and still instantly recognize what it depicts. You rarely need all original pixels to preserve the important content.

PCA directly exploits this correlation-based redundancy. It looks for features that vary together and combines them into a single *new feature* that captures their shared variation. These new features are called *principal components*. One nice feature is that principal components are ordered: the first principal component captures the single strongest pattern of variation in the entire dataset. For example, imagine a video of a room where the overall illumination level changes. That widespread, correlated fluctuation across millions of pixels is exactly the kind of pattern PCA will automatically detect. The entire background trend will be extracted as the first principal component, replacing millions of redundant pixel-by-pixel changes with a single number. It will basically represent the "light level" in the room.

![PCA Room](resources/jellyfish_room_pca.jpg)

Now imagine that on the desk there is a small jellyfish toy with a built-in light that cycles between deep violet and almost-black. But the group of pixels that represent the jellyfish all brighten and darken together in their own violet rhythm, independently of the room's background illumination. This creates a localized cluster of highly correlated pixel changes that are not explained by the global brightness pattern. Because this fluctuation is coherent within that region and independent from the background illumination, PCA will naturally identify this jellyfish pixel cluster as the *second* principal component.

In this way, PCA acts like a very smart form of compression. Instead of throwing away random pixels or selecting every third column of the image, it builds new features that preserve as much of the original information as possible based on which pixels are correlated with each other. 

Interestingly, PCA offers a way to reconstruct the original dataset from these compressed features. By weighting and combining the principal components, you can approximate the original pixel values. In the jellyfish room example, knowing only two numbers (background brightness level and brightness of the jellyfish toy) would be enough to recreate the essential content of each frame, even though the full image contained millions of pixels. This would be let us represent an entire image with two numbers instead of millions!

In real datasets, the structure is not usually this clean, so you will typically need more than two components to retain the information in such high-dimensional datasets. PCA provides a precise way to measure how much variability each component captures, which helps you decide how many components to keep while maintaining an accurate, compact version of the original data. 

We are not going to go deeply into the linear algebra behind PCA, but will next go into a code example to show how this works in practice. 



 ### PCA Demo Using the Olivetti Faces Dataset

In this demo, we will use the Olivetti faces dataset from scikit-learn to see how PCA works on a high-dimensional dataset. Each face image is 64x64 pixels, which means each image has 4096 pixel values. That means each sample lives in a 4096-dimensional space. Many of those pixels are correlated with each other, because nearby pixels tend to have similar intensity values (for instance, the values around the eyes tend to fluctuate together). This makes the Olivetti dataset a great example for dimensionality reduction with PCA.

First, some imports.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
```

Next, load the Olivetti faces dataset

```python
faces = fetch_olivetti_faces()
X = faces.data        # shape (n_samples, 4096)
images = faces.images # shape (n_samples, 64, 64)
y = faces.target      # person IDs (0 to 39)

print(X.shape)
print(images.shape)
print(y.shape)
```

There are 400 faces in the dataset. Each row of `X` is one face, flattened into a 4096-dimensional vector. The `images` array stores the same data in image form, as 64x64 arrays that are easier to visualize.

Visualize some sample faces

```python
fig, axes = plt.subplots(4, 10, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(images[i], cmap="gray")
    ax.axis("off")
plt.suptitle("Sample Olivetti Faces")
plt.tight_layout()
plt.show()
```
This gives you a quick look at the variety of faces in the dataset. Remember that each one of these images is a single point in a 4096-dimensional space.

#### Fit PCA and look at variance explained
Here we fit PCA to the full dataset. We will look at how much of the total variance is explained as we add more and more components.

```python
pca_full = PCA().fit(X)

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.title("PCA Variance Explained on Olivetti Faces")
plt.grid(True)
plt.show()
```

This curve shows how quickly we can capture most of the variation in the dataset with far fewer than 4096 components. Within 50 components, well over 80 percent of the variance in the dataset has been accounted for. 

#### Plot eigenfaces
We can plot the principal components to get a sense for what the correlated features look like in our image set, and we can visualize them as images. Note these are often called *eigenfaces* (this is for technical reaons: the linear algebra used to generate principal components uses something called eigenvector decomposition):

```python
mean_face = pca_full.mean_.reshape(64, 64)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

# Mean face
axes[0, 0].imshow(mean_face, cmap="gray")
axes[0, 0].set_title("Mean face")
axes[0, 0].axis("off")

# First 9 principal components (eigenfaces)
for i in range(9):
    ax = axes[(i + 1) // 5, (i + 1) % 5]
    ax.imshow(pca_full.components_[i].reshape(64, 64), cmap="bwr")
    ax.set_title(f"PC {i+1}")
    ax.axis("off")

plt.suptitle("Mean Face and First Eigenfaces")
plt.tight_layout()
plt.show()
```

The mean face is the average of all faces in the dataset. You can think of these eigenfaces as basic building blocks for constructing individual faces. PC1 is the eigenface that captures the most correlated activity among the pixels, PC2 the second most, and so on. Each eigenface shows the discovered pattern of pixel intensity changes. Red regions mean "add brightness to the mean" when you move in the direction of that component, and blue regions mean "subtract brightness here".

#### Reconstructions with different numbers of components
We discussed above how you can use principal components to reconstruct or approximate the original data. We will show this now. The following code will: 

- Choose 10 random faces from the dataset.
- Reconstruct them using different numbers of components.
- Compare these reconstructions to the original faces.

```python
rng = np.random.default_rng(42)
rand_indices = rng.choice(len(X), size=10, replace=False)

components_list = [0, 5, 15, 50, 100]

fig, axes = plt.subplots(len(components_list), len(rand_indices), figsize=(10, 7))

for i, n in enumerate(components_list):

    if n == 0:
        X_recon = X.copy()
        row_label = "Original"
    else:
        pca = PCA(n_components=n)
        X_proj = pca.fit_transform(X)
        X_recon = pca.inverse_transform(X_proj)
        if n == 1:
            row_label = "PCs: 1"
        else:
            row_label = f"PCs: 1-{n}"

    for j, idx in enumerate(rand_indices):
        ax = axes[i, j]
        ax.imshow(X_recon[idx].reshape(64, 64), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

        # Row labels on the left
        if j == 0:
            ax.set_ylabel(
                row_label,
                rotation=0,
                ha="right",
                va="center",
                fontsize=10,
            )

        # Column labels with person ID on top row
        if i == 0:
            ax.set_title(f"ID {y[idx]}", fontsize=8, pad=4)

plt.suptitle("Olivetti Face Reconstructions with Different Numbers of PCs", y=0.97)
plt.subplots_adjust(left=0.20, top=0.90)
plt.show()
```

The top row shows the original faces. Each lower row shows reconstructions using an increasing number of principal components:

- `PCs: 1-5` keeps only a very small number of components, so the faces look blurry but still recognizable.
- `PCs: 1-15` and `PCs: 1-50` look progressively sharper.
- `PCs: 1-100` usually looks very close to the original, even though we are using far fewer than 4096 numbers.

This demonstrates how PCA can dramatically reduce the dimensionality of the data while still preserving the essential structure of the faces. In sum:
- Each face lives in a very high-dimensional space (4096 features).
- PCA finds directions (eigenfaces) that capture the main patterns of variation.
- A relatively small number of principal components can capture most of the meaningful information.

## Summary
In this lesson you saw that good machine learning does not start with fancy models: it starts with good data. Choosing the right feature types, scaling numeric values, encoding categories, and creating new features all help your models see patterns more clearly. Often it means reducing the dimensionality of our data. There are no magic rules here: you will learn the most by exploring your data, visualizing it, and trying small experiments. The habits you build around preprocessing and feature engineering now will pay off with classifiers you build later in this lesson.

## Check for Understanding

### Question 1 
Which of the following are **categorical** features?

- A) age  
- B) t-shirt size ("S", "M", "L")  
- C) temperature  
- D) city name  

<details>
<summary>View answer</summary>
**Answer:** B and D
</details>

### Question 2  
True or false: A categorical feature always has a natural order.

<details>
<summary>View answer</summary>
**Answer:** False. Most categorical features are nominal and have no natural order.
</details>


### Question 3  
Why might a classifier pay more attention to a feature ranging from 0–1000 than to one ranging from 0–10?

<details>
<summary>View answer</summary>
**Answer:** Because the larger numbers dominate distance-based calculations unless we scale the features.
</details>


### Question 4  
What does a z-score of **-1** mean?

<details>
<summary>View answer</summary>
**Answer:** The value is one standard deviation below the feature’s mean.
</details>


#### Question 5  
Which model *does not* require scaling?

- A) KNN  
- B) Neural networks  
- C) Logistic regression  
- D) Decision trees  

<details>
<summary>View answer</summary>
**Answer:** D) Decision trees
</details>


### Question 6  
What problem occurs if we encode categories like:

```
dog -> 1
cat -> 2
bird -> 3
```

<details>
<summary>View answer</summary>
**Answer:** It creates a false ordering and suggests numeric relationships that do not exist.
</details>


### Question 7  
If you one-hot encode the categories ["shirt", "dog", "plane"], how many output columns will you get?

<details>
<summary>View answer</summary>
**Answer:** 3 columns
</details>


#### Question 8  
What does PCA do with datasets that have many correlated features?

<details>
<summary>View answer</summary>
**Answer:** Reduce redundancy by combining correlated features into new components.
</details>

