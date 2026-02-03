# ML: Introduction to Data Preprocessing and Feature Engineering
Machine learning algorithms expect data in a clean numerical format, but real datasets often do not arrive that way. *Preprocessing* data for machine learning makes features easier for models to learn from, and is required for most `scikit-learn` workflows. This assumes the data has already been cleaned up using techniques used in Python 100 (for instance, missing data has been handled). 

We will cover:

- Numeric vs categorical features
- Feature scaling (standardization, normalization)
- Encoding categorical variables (one hot encoding)
- Creating new features (feature engineering)
- Dimensionality reduction and Principal component analysis

Some of the material will be review of what you learned in Python 100, but specifically geared toward optimizing data for consumption by classifiers. 

## 1. Numerical vs Categorical Features

[Video on numeric vs categorical data](https://www.youtube.com/watch?v=rodnzzR4dLo)

Before we can train a classifier, we need to understand the kind of data we are giving it. Machine learning models only know how to work with _numbers_, so every feature in our dataset eventually has to be represented numerically. 

**Numerical features** are things that are *already* numbers: age, height, temperature, income. They are typically represented as floats or ints in Python. Models generally work well with these, but many algorithms still need the numbers to be put on a similar scale before training. We will cover scaling next.

**Categorical features** describe types or labels instead of quantities. They are often represented as `strings` in Python: city name, animal species, shirt size. These values mean something to *us*, but such raw text is not useful to a machine learning model. We need to convert these categories into a numerical format that a classifier can learn from. That is where things like one-hot encoding come in (which we will cover below). 

Even large language models (LLMs) do not work with strings: as we will see in future weeks when we cover LLMs, linguistic inputs must be converted to numerical arrays before large language models can get traction with them. 

## 2. Scaling Numeric Features

[Video overview of feature scaling](https://www.youtube.com/watch?v=dHSWjZ_quaU)

When we have data in numerical form, we might think we are all set to feed it directly to a machine learning algorithm. However, this is not always true. Even though numeric features are already numbers, we still have to think about how they behave in a machine learning model. Many algorithms do not just look at the numerical features themselves, but at how large they are *relative to each other*. If one feature uses much bigger units than another, the model may unintentionally focus on the bigger one and ignore the smaller one.

For example, imagine a dataset with two numeric features:

- age (with range 18 to 70)
- income (with range 15,000 to 350,000)

Both features matter, but income numbers vary on a much larger scale. Many ML algorithms will be sensitive to this, especially those that depend on distance calculations, will end up weighting income more heavily than age, just because of this difference in scale.  

Scaling helps put numeric features on a similar footing so that models can consider them more fairly. There are two main scaling methods, normalization and standardization.

## Normalization (Min-Max Scaling)
Normalization, aka min-max scaling, rescales each feature so that it falls into the range [0, 1]. This helps ensure that no feature overwhelms another just because it uses larger numbers.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```
Now X will have values that fall into the desired range. 

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

Some models are much less sensitive to scale, such as decision trees. 

Even numeric features require thoughtful preparation. Scaling helps many models learn fairly from all features instead of just listening to the biggest numbers.

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
You can see that age and income are on completely different numeric scales. On the bottom plot, after standardization, both features live in the same z-score space and can be directly compared.

A z-score tells you how many standard deviations a value is above or below the mean of that feature:

- z = 0 means "right at the average"
- z = 1 means "one standard deviation above average"
- z = -2 means "two standard deviations below average"

So a negative age or income after standardization does not mean a negative age or negative dollars. It just means that value is below the average for that feature.

## 3. One-Hot Encoding for Categorical Features

[Video about one-hot encoding](https://www.youtube.com/watch?v=G2iVj7WKDFk)

Assume you have a categorical feature that you are feeding to an ML model. For instance, musical genre (maybe the model is predicting whether the music will contain electric guitar or not, for our instrument sales site).  Categorical features (like "jazz", "classical", "rock") must be converted into numbers before a machine learning model can use them. But we cannot simply assign integers like:

```
'classical' -> 1
'jazz' -> 2
'rock' -> 3
```

If we did this, the model would think that "classical" is smaller than "rock", or that the distance between genres carries meaning. These numbers would create a false ordering that does not exist in the real categories. To avoid this, we use one-hot encoding. One-hot encoding represents each category as an array where:

- all elements are 0 except for one element, which is 1
- the position of the 1 corresponds to the category

So the categories from above would become:

```
classical  -> [1, 0, 0]
jazz  -> [0, 1, 0]
rock -> [0, 0, 1]
```

Each category is now represented cleanly, without implying any ordering or distance between them. This is exactly what we want for most categorical features in classification.

> Side note: Most categorical features have no natural order (dog, cat, bird; red, green, blue). These are known as *nominal* categories: one-hot encoding works great for them. Some categories do have an order (`small` < `medium` < `large`). These are known as *ordinal* categories. For a discussion of some of the nauances of this case, see [this page](https://www.datacamp.com/tutorial/categorical-data).   

### One-hot encoding in scikit-learn

Because one-hot encoding is so important, it a built-in class in scikit-learn: 

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

y = [["jazz"], ["rock"], ["classical"], ["jazz"]]
y_encoded = encoder.fit_transform(y)

print("one-hot encoded categories:")
print(y_encoded)
```

Output will be something like:

```
one-hot encoded categories:
[[0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
```

We will see more practical examples of one-hot encoding in future lessons. 

> One important thing to notice about one-hot encoding is that it increases the number of features in your dataset. If a categorical feature has N unique categories, then one-hot encoding replaces that single column with N new columns. This is usually fine for small categorical features, but it can cause problems when a feature has many unique values. For example, if you have a feature representing ZIP codes, there may be thousands of unique values. One-hot encoding this feature would create thousands of new columns, which can make things very unwieldy in practice. In such cases, alternative encoding methods (like  embedding techniques, which we will cover in the AI lessons) may be more appropriate.

## 4. Creating New Features (Feature Engineering)

[Video overview of feature engineering](https://www.youtube.com/watch?v=4w-S6Hi1mA4)

Sometimes the data we start with is not the data that will help a model learn best. A big part of machine learning is noticing when you can create new data, or new features, that capture something useful about the data. This is called *feature engineering*, and it can make a big difference in how well a classifier performs.

You have already learned about this idea in Python 100 when learning about Pandas (you created new columns from existing columns). Here we revisit the idea with an ML mindset: Can we create features that make patterns easier for the model to learn?

To make this concrete, let’s create a synthetic dataframe with ten fictional people:

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

df.head()
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

If you have a datetime column (as we do above), you can often pull out the pieces that matter for prediction:

```python
df["weekday"] = df["birthdate"].dt.weekday  
df["birth_year"] = df["birthdate"].dt.year
```

A model might not care about the full timestamp, but it might care about whether something happened on a weekday or weekend. This might matter when predicting costs of healthcare, for instance.

### Binning continuous values

Sometimes a numeric feature is more predictive for a classification task if we convert it into categories. For example, instead of raw ages, we can group them into age *groups*:

```python
current_year = 2025
df["age"] = current_year - df["birth_year"]

df["age_group"] = pd.cut(df["age"], bins=[0, 20, 30, 40, 60], labels=["young","20s","30s","40+"])
df[["name","age","age_group"]].head()
```
This can help when the exact number is less important than the general range. Individual ages may be noisy, but age groups might capture broader patterns more effectively.

Before actually feeding such newly created categorical features to an ML model, you would need to one-hot encode them as described above. Let's look at how that would work for the `age_group` feature:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

age_groups = df[["age_group"]]        # must be 2D for scikit-learn
encoded = encoder.fit_transform(age_groups) # one-hot encoded age groups

# create a dataframe for easy viewing of the one-hot encoded age-group columns
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["age_group"])
)

encoded_df.head()
```
This would give you a one-hot encoded version of the age groups that you could feed to a model, or attach to the original dataframe.

### Boolean features

A simple yes/no feature can sometimes capture a pattern that the raw values obscure. For example:

```python
df["is_senior"] = (df["age"] > 65).astype(int)
df[["name","age","is_senior"]].head()
```

Even though age is already numeric, the idea of "senior" might be more directly meaningful for a model (for instance if you are thinking about pricing for restaurants).

Note because this is Boolean, you would not need to one-hot encode it: 0 and 1 are already perfect numeric representations for a binary feature.

### Final thoughts on feature engineering
There are no strict rules for feature engineering. It is a creative part of machine learning where your intuitions and understanding of the data matters a great deal. Good features often come from visualizing your data, looking for patterns, and thinking about the real-world meaning behind each column. Creativity and domain-specific knowledge helps a lot here: someone who knows the problem well can often spot new features that make the model's job easier. As you work with different datasets, you will get better at recognizing when a new feature might capture something important that the raw inputs miss. Feature engineering is less about following a checklist and more about exploring, experimenting, and trusting your intuition as it develops.


## 5. Dimensionality reduction 
Many real datasets contain far more dimensions, or features, than we truly need (in pandas, individual features are represented by separate columns in a dataframe). Some features are almost duplicates of each other, or they carry very similar information -- this is known as *redundancy*. When our feature space gets large, models can become slower, harder to interpret, harder to fit to data, and become prone to overfitting. Dimensionality reduction lets us simplify a dataset by creating a smaller number of informative features. 

As discussed in the [Introduction to Machine Learning](../02_ML_intro/01_ML_Intro.md), one helpful way to picture this is to think about images. A high-resolution photo might have hundreds of millions of pixels, and each pixel in an image is a feature. But you can shrink an image down to a small thumbnail and still recognize that it is a picture of your friend. Dimensionality reduction works the same way in general: the goal is to keep the important structure while throwing away noise and redundancy. ML algorithms can work while throwing away a great deal of raw data, and this can speed things up tremendously. 

Dimensionality reduction is useful for many reasons. One, it can be a useful tool for visualizing high-dimensional data (our plots are typicically in 2D or 3D, so if we want to visualize a 1000-dimension dataset, it can be helpful to project it to a 2D or 3D space for visualization). Also, for ML, dimensionality reduction can help fight overfitting and eliminate noise from our data. We saw last week that overfitting comes from model complexity (a model with too many flexible parameters can memorize noise in the training set). However, high-dimensional data can make overfitting more likely because it gives the model many opportunities to chase tiny, meaningless variations. Reducing feature dimensionality can sometimes help by stripping away that noise and highlighting the core structure the model should learn.

### Principal component analysis (PCA)
Before moving on, consider watching the following introductory video on PCA:
[PCA concepts](https://www.youtube.com/watch?v=pmG4K79DUoI)

PCA is one of the most widely used dimensionality reduction techniques.

To build intuition for how PCA works, it helps to return to the image example. A raw image may contain millions of pixel values, but many of those pixels fluctuate together. Nearby pixels tend to be highly correlated: if a region of an image becomes brighter, most of the pixels in that region brighten together. This means that while the dataset appears extremely high dimensional, the underlying structure is often much simpler.

PCA directly exploits this correlation-based redundancy. It looks for groups of features that vary together and combines them into new features that capture their shared variation. These new features are called principal components. A key property of PCA is that these components are ordered: the first principal component captures the strongest shared pattern in the dataset, the second captures the next strongest pattern, and so on.

For example, imagine a video of a room where the overall illumination level slowly changes because sunlight enters through a window. That widespread, correlated change across millions of pixels is exactly the kind of pattern PCA will detect. PCA would extract this global brightness fluctuation as the first principal component, replacing millions of redundant pixel-level changes with a single number.

![PCA Room](resources/jellyfish_room_pca.jpg)

You can see on the desk in the room there is a small jellyfish lamp. We can imagine this lamp cycles between deep violet and almost-black. The pixels corresponding to the jellyfish brighten and darken together in their own rhythm, independently of the room's background illumination. This creates a second, independent cluster of correlated pixel changes. PCA would identify this pattern as the second principal component.

In this simplified example, PCA reduces millions of pixel features down to just two numbers: one capturing the room's overall brightness changes, and one capturing the jellyfish's independent light fluctuations. This kind of dramatic dimensionality reduction is possible because of strong correlations in the data. In real datasets, the structure is rarely this clean, so more than two components are usually needed to capture all of the essential information in the dataset.

 We are not going to focus on the linear algebra behind PCA. Our goal here has been to build intuition. To continue, and convince you how well it works, we will next walk through a hands-on demo that shows how powerful PCA can be for feature extraction and dimensionality reduction.

 ## PCA Demo

In this demo, we will work with a synthetic dataset designed to closely mirror the intuition-building example from the lesson: a movie with *massive* redundancy. The movie shows a room where the background lighting slowly rises and falls over time, while a jellyfish lamp on a table fluctuates independently and much more rapidly.

This demo is designed to show three key things:

1. PCA can automatically separate the room and the jellyfish into different components without being told they exist (source separation).
2. PCA can reduce a dataset with over a million dimensions per frame down to just two meaningful numbers.
3. Using those two numbers, PCA can reconstruct the original movie with almost no visible loss.

We strongly recommend running this demo in Jupyter or Kaggle. Several steps involve animations that are meant to be viewed in a browser.

### Imports and inspect original data
First, import the required libraries. We will use `gdown` to download the movie from Google Drive (the file is about 50MB):

```python
import gdown
from IPython.display import YouTubeVideo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
```

Next, load and plot the *actual* brightness signals that were used to generate the movie. These values are stored separately so that we can later compare them to what PCA discovers on its own. 

```python
# Load actual  brightness values
brightness_df = pd.read_csv("resources/brightness_values.csv")
frame_idx = brightness_df["frame"].to_numpy()
room_brightness = brightness_df["room_brightness"].to_numpy()
jelly_brightness = brightness_df["lamp_brightness"].to_numpy()

# Plot brightness
plt.plot(room_brightness, color='k', label="room")
plt.plot(jelly_brightness, color='purple', label="lamp")
plt.legend()
plt.xlabel("Frame (Time)");
plt.ylabel("Brightness");
```

When you view the movie, you will see the dynamics described above: the room's brightness slowly changes from light to dark, while the jellyfish lamp ramdomly fluctuates independently of the room. You can [view it on YouTube](https://www.youtube.com/watch?v=uEFkp0mLzgI), or if you are running this notebook in Jupyter or Kaggle, you can view it inline below:

```python
YouTubeVideo("uEFkp0mLzgI", width=600, height=350)
```   

### Download movie data
Next, download the movie itself. To keep the repository lightweight, we download the file to a local `ctd_data` directory in your home folder. Feel free to change this path if you prefer.

```python
ctd_data_dir = Path.home() / "ctd_data"
ctd_data_dir.mkdir(parents=True, exist_ok=True)

filename = "jellyfish_movie.npz"
data_path = ctd_data_dir / filename
data_url = r"https://drive.google.com/uc?id=1JDNoc1ojz3_MqJURy_KjW4hkOKJ5LtiA"

if not data_path.exists():
    print(f"Downloading {filename}. This can take a minute.")
    gdown.download(
        data_url,
        str(data_path),
        quiet=False,
        fuzzy=True,
        use_cookies=True,
    )
else:
    print(f"{filename} already downloaded. Download skipped.")
```

Once the file is downloaded, load it and inspect its shape:

```python
data = np.load(data_path)
frames = data["frames"] 

num_frames, num_rows, num_cols = frames.shape
print(num_frames, num_rows, num_cols)
print(f"Pixels per frame: {num_rows*num_cols}")
```
Each frame contains over **one million pixels**, meaning each frame lives in a space with over one million dimensions. Despite this, the movie is driven by just two underlying signals: room brightness and jellyfish brightness.

Let's check out the mean movie image from the stack:

```python
mean_image = frames.mean(axis=0)
print(mean_image.shape)
plt.imshow(mean_image, cmap='grey');
plt.axis('off');
```
In the mean image we see the room and the lamp at their mean luminance. This average image represents the baseline scene that PCA uses as a starting point for reconstructing individual frames. We will do this below. 

### Perform PCA and inspect components
Before performing PCA, we need to put data into a format that scikit-learn algorithm expects. We reshape each image into a one-dimensional array. This gives us a data matrix of shape `(num_samples, num_dimensions)`. Then, we will follow the standard scikit-learn API, creating a PCA object before running the `fit()` method. Note we won't worry here about all the subtleties of PCA in this demo -- our goal is to illustrate how it works in practice:

```python
X = frames.reshape(num_frames, -1).astype(np.float32)  # num_samples x num_dimensions
print(X.shape)
pca = PCA(n_components=4, svd_solver="randomized", random_state=0)
pca.fit(X)

components = pca.components_ 
print(components.shape)
```
In the above code cell, we ran PCA, and only had it return four components (otherwise it would have taken a *long* time to run). 

Each principal component has the same dimensionality as one flattened image, which means we can reshape and visualize them:

```python
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for k, ax in enumerate(axes):
    ax.imshow(components[k].reshape(num_rows, num_cols), cmap="gray")
    ax.set_title(f"PC{k+1}")
    ax.axis("off")
plt.show()
```
Without any supervision, PCA has separated the room structure and the jellyfish lamp into distinct components. This is a clear example of **source separation** driven purely by correlation across pixels. The first principal component (PC1) captures the room's overall brightness changes, while the second principal component (PC2) captures the jellyfish lamp's independent fluctuations. This is exactly what we expected based on our understanding of the movie's structure.

### Explained variance 
We can see percentage of the variance in the original dataset is explained by the four components:

```python
perc_exp_vals = pca.explained_variance_ratio_ * 100
total_explained = perc_exp_vals.sum()
print("Explained variance (%):", ", ".join(f"{v:.2f}" for v in perc_exp_vals))
print(f"Total (%): {total_explained:.2f}")
```
There might be some variability here, but the typical outputs look like this:

    Explained variance (%): 98.31, 1.52, 0.04, 0.02
    Total (%): 99.88

The first component explains 98.3 percent of the variance, which is pretty amazing, but also makes sense because the room occupies most of the pixels, so its correlated brightness changes account for the majority of the variance. The second component mops up most of the remaining variance (1.5%). The first two components combined capture 99.8% of the variance in the dataset! The remaining two components capture less than 0.06%, which is negligible (feel free to plot them, they look like ghosts of the jellyfish lamp). :smile: 

### Scores and reconstruction

PCA also provides *scores*, which tell us how strongly each component is expressed in each frame:

```python
scores = pca.transform(X)
print(scores.shape)
```
We see that the shape of the scores is `(250,4)`, which makes sense -- there are 250 frames in the movie, and four components. Let's compare the first two PC scores to the original brightness signals:

```python
frame_indices = [29, 231] # to recreate later

f, axes = plt.subplots(2,2)
axes = axes.ravel() # create 1d array of axis

axes[0].plot(room_brightness, color='grey', label='Actual Room Brightness')
axes[0].set_title('Original Room Brightness')
axes[0].set_ylabel('Brightness')

axes[2].plot(scores[:,0], color='grey', zorder=1 )
axes[2].set_title('Component 1 Score')
axes[2].scatter(frame_indices, scores[frame_indices, 0], s=15, color='black', zorder=3)
# for frame_ind in frame_indices:
#     axes[2].axvline(frame_ind, color="black", linewidth=0.5)
axes[2].axhline(color='k', linewidth=0.5)
axes[2].set_xlabel('Frame')
axes[2].set_ylabel('Score')

axes[1].plot(jelly_brightness, color='plum', label='Actual Lamp Brightness')
axes[1].set_title('Original Lamp Brightness')

axes[3].plot(scores[:,1], 'plum', zorder=1)
axes[3].scatter(frame_indices, scores[frame_indices, 1], s=15, color='purple', zorder=3)
# for frame_ind in frame_indices:
#     axes[3].axvline(frame_ind, color="purple", linewidth=0.5)
axes[3].axhline(color='k', linewidth=0.5)
axes[3].set_title('Component 2 Score')
axes[3].set_xlabel('Frame')

plt.tight_layout()
```
You can see the amazing job PCA did extracting the shape of the "source" signals for the room and lamp brightness. Note that while the *shape* is captured almost exactly, the *values* are not the same: the original signals were positive, but there are positive and negative values for the scores. PCA scores represent deviations from the mean, not raw brightness values.

PCA reconstruction starts with the mean image and adds deviations along each principal component. We can write a function to reconstruct any frame using the first two PCs and their scores:

```python

component1 = components[0].reshape(num_rows, num_cols)
component2 = components[1].reshape(num_rows, num_cols)

def reconstruct_from_scores(frame_idx, scores):
    """
    Reconstruct frame `frame_idx` using the first 2 PCs

    Returns PC1 estimate and PC1+PC2 estimate
    """
    # First-order reconstruction
    x_hat1 = mean_image + scores[frame_idx, 0]*component1

    # Second_order reconstruction
    x_hat2 = x_hat1 + scores[frame_idx, 1]*component2
      
    x_hat1 = np.clip(x_hat1, 0, 255)
    x_hat2 = np.clip(x_hat2, 0, 255)
    return x_hat1, x_hat2

```
We can see from that function the simplicity of PCA-based reconstruction. Start with the mean image, and then add a weighted version of the first component (the room component). Then add a weighted version of the second component (the jellyfish component) to get the full second-order reconstruction. 

Let's see how things go for two frames from our movie (frames 29 and 131). The following plots the original frame on the left, and the reconstruction using PC1 only in the second column, and PC1 *and* PC2 on the right:

```python
fig, axes = plt.subplots(2, 3, figsize=(8, 5))

for row, idx in enumerate(frame_indices):
    # Original
    ax = axes[row, 0]
    ax.imshow(frames[idx], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Frame {idx}: original")
    ax.axis("off")

    pc1_reconstruction, pc12_reconstruction = reconstruct_from_scores(idx, scores)
    
    # PC1 only
    ax = axes[row, 1]
    ax.imshow(pc1_reconstruction, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Frame {idx}: PC1 only")
    ax.axis("off")

    # PC1 + PC2
    ax = axes[row, 2]
    ax.imshow(pc12_reconstruction, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Frame {idx}: PC1 + PC2")
    ax.axis("off")

plt.tight_layout()
plt.show()
```
These plots reveal a bit more about how PCA works. Let's focus on the bottom row (Frame 231), which has a dark room and bright lamp. With PC1 alone, the room becomes darker but the jellyfish remains flat and dull. PC1 only controls how bright or dark the whole room is, so when we add only PC1 we are really just turning the overall room lighting up or down around that average scene: the room becomes dark, but the dull mean jellyfish stays. However, when we add in PC2, the jellyfish suddenly pops back to life as in the original frame.

In this demo, PCA reduced a dataset with over one million dimensions per frame down to just *two meaningful numbers* while still allowing us to reconstruct the movie with little visible loss. Most real-world datasets are much more messy and noisy, and don't contain *this* much redundancy, this example is useful to demonstrate just how powerful PCA can be at discovering hidden correlated structure in your data. This is why it is typically one of the first tools people use to simplify very high-dimensional datasets. 

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
**Answer:** Reduce redundancy by combining correlated features into individual components that can be used as new features.
</details>

