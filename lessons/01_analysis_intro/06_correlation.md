# Correlation: How Variables Move Together 📈

## Useful Resources

If you would like a quick external overview before diving in, this
article provides a gentle introduction:

https://www.geeksforgeeks.org/data-analysis/exploring-correlation-in-python/
You can also watch this video for a visual explanation of correlation:
https://www.youtube.com/watch?v=99kC5neKvkE

Now let us build a deeper and more structured understanding.

## What Is Correlation?

Correlation measures how two variables vary together. If both tend to be above their averages at the same time, the relationship is positive. If one tends to be above its average while the other is below, the relationship is negative. If knowing one variable tells you nothing about the other, the relationship is close to zero.

The most common measure is Pearson correlation, written as r. Its value always falls between -1 and +1. Values near +1 indicate a strong positive linear relationship. Values near -1 indicate a strong negative linear relationship. Values near 0 indicate little to no linear relationship.

Under the hood, correlation is built from a quantity called covariance. Covariance measures how two variables move together, but it depends on the scale of the data. For example, measuring height in centimeters versus meters would change the covariance value.

To make the relationship comparable across different datasets and units, we divide covariance by the variability of each variable. This gives us Pearson’s formula for r:

$$
r = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}
$$

The numerator measures how X and Y move together. The denominator adjusts for how much X and Y vary on their own, forcing r to stay within -1 and +1. This way, r reflects only the strength and direction of their linear relationship.

Don't worry about the math for covariation, but you can find it in the links above if you are curious. The key idea is that correlation quantifies how two variables move together, while adjusting for their individual variability.


## Why Correlation Matters

Correlation helps us discover patterns in data.

In business, we might ask whether advertising spend rises with sales. In
education, we might ask whether study time tracks exam performance. In
health research, we might ask whether exercise time relates to blood
pressure.

In machine learning, correlation helps us choose useful features. If two
variables are almost identical, such as height measured in centimeters
and inches, we do not need both. Correlation helps us detect redundancy.

Beware of inferring causality from correlation. Ice cream sales and drowning
incidents both rise in summer, but one does not cause the other.
Correlation reveals a *relationship* exists, not causation. Sometimes the relationship could be caused by some third underlying factor (summer weather), or it could be a coincidence. Always reason about the mechanism behind the relationship and visualize your data to understand its structure.

## Visualizing Correlation

Numbers tell us strength. Plots tell us structure.

### Scatterplots

Consider study hours and exam scores.

``` python
import matplotlib.pyplot as plt

study_hours = [1, 2, 3, 4, 5, 6, 7, 8]
exam_scores = [48, 55, 58, 67, 72, 74, 81, 84]

plt.scatter(study_hours, exam_scores, color='teal')
plt.title("Study Hours vs Exam Scores")
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores")
plt.show()
```

![Scatterplot of Two
Variables](resources/04_correlation_1_scatter_two_variables.png)

As study hours increase, exam scores increase. The upward pattern shows
positive correlation.

### Heatmaps

When working with many variables, we often compute a correlation matrix
and visualize it as a heatmap.

``` python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    "hours_study": [1, 2, 3, 4, 5, 6, 7, 8],
    "exam_score": [50, 54, 63, 66, 72, 77, 80, 86],
    "sleep_hours": [8.2, 7.8, 7.3, 6.7, 6.2, 5.8, 5.4, 4.9],
    "stress_level": [30, 35, 45, 55, 60, 65, 72, 80],
    "screen_time": [2.5, 3, 3.5, 4, 4.5, 5.2, 6, 6.5]
}

df = pd.DataFrame(data)
corr = df.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```

![Correlation Heatmap](resources/04_correlation_2_heatmap.png)

The heatmap lets us quickly see which variables move together and which
move in opposite directions.

## Calculating Correlation in Python

### Using NumPy

``` python
import numpy as np

study_hours = [1, 2, 3, 4, 5, 6, 7, 8]
exam_scores = [48, 53, 60, 66, 72, 76, 80, 84]

corr_matrix = np.corrcoef(study_hours, exam_scores)
print(corr_matrix)
```

The off-diagonal value is the correlation coefficient between the two
variables.

### Using Pandas

``` python
import pandas as pd

data = {
    "hours_study": [1, 2, 3, 4, 5, 6, 7, 8],
    "exam_score": [52, 57, 62, 66, 71, 75, 79, 83],
    "sleep_hours": [8.1, 7.8, 7.3, 6.9, 6.4, 6.0, 5.6, 5.1]
}

df = pd.DataFrame(data)
print(df.corr())
```

### Including a p-value with SciPy

Sometimes we want to know whether an observed correlation is significant, or likely due
to chance. The pearsonr function gives both r and a p-value.

``` python
from scipy.stats import pearsonr
import numpy as np

study_hours = [1, 2, 3, 4, 5]
exam_scores = [52, 56, 63, 64, 71]

r, p = pearsonr(study_hours, exam_scores)
print("Correlation:", round(r, 2))
print("p-value:", round(p, 4))
```

A small p-value suggests the relationship is unlikely to be due to random
noise.

## Seeing Correlation Strengths in Action

To build deeper intuition, we can generate data with different
correlation values and visualize them.

``` python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
correlations = np.arange(-1.0, 1.1, 0.2)

fig, axes = plt.subplots(3, 4, figsize=(15, 10))
axes = axes.flatten()

for i, r in enumerate(correlations):
    x = np.random.randn(100)
    y = r * x + np.sqrt(1 - r**2) * np.random.randn(100)
    axes[i].scatter(x, y, alpha=0.6)
    axes[i].set_title(f"r = {r:.1f}")

plt.tight_layout()
plt.show()
```

Seeing the scatter patterns change as r moves from -1 to +1 builds
strong intuition about what correlation actually represents.

## Limits of Correlation

Correlation only measures linear relationships. Two variables can have
the same correlation value and completely different shapes.

A famous example is Anscombe's Quartet. The four datasets below have
nearly identical correlation values, yet their scatterplots look very
different.

``` python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("anscombe")

sns.lmplot(x="x", y="y", col="dataset", data=df,
           col_wrap=2, ci=None,
           scatter_kws={"s":50, "alpha":0.7},
           line_kws={"color":"red"})

plt.suptitle("Anscombe's Quartet: Same Correlation, Very Different Data", y=1.05)
plt.show()
```

Correlation also does not imply causation. Outliers can distort
correlation dramatically. In very large datasets, even tiny effects can
appear statistically significant. Always visualize your data and reason
about the mechanism behind the relationship.

## Check for Understanding

**Q1. A dataset shows a correlation of +0.85 between advertising spend and sales. What does this tell us, and what does it *not* tell us?**  
<details>
<summary>Show Answer</summary>  
It tells us that higher advertising spend is strongly associated with higher sales. It does not prove that advertising causes the increase in sales, since correlation does not imply causation.  
</details>

**Q2. Two variables have a correlation of 0. Why should you still look at a scatterplot before concluding there is no relationship?**  
<details>
<summary>Show Answer</summary>  
A correlation of 0 means there is no *linear* relationship, but there could still be a nonlinear pattern. A scatterplot may reveal structure that correlation alone does not capture.  
</details>

**Q3. If we measure height in centimeters instead of meters, will the correlation between height and weight change? Why or why not?**  
<details>
<summary>Show Answer</summary>  
No. Correlation does not depend on the units of measurement because it adjusts for the variability of each variable. Changing the scale changes covariance, but not correlation.  
</details>

**Q4. A small dataset shows a correlation of 0.6 with a large p-value. What might this suggest?**  
<details>
<summary>Show Answer</summary>  
It suggests that although the relationship appears moderately strong, the sample size may be too small to confidently rule out random chance.  
</details>

**Q5. Why can a single extreme outlier dramatically change a correlation value?**  
<details>
<summary>Show Answer</summary>  
Because correlation depends on how data points vary around the mean, one extreme value can heavily influence the overall pattern and artificially inflate or reduce the measured relationship.  
</details>

## Congratulations!
With this we have completed the **Correlation** section, let’s keep learning.
