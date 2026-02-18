# Hypothesis Testing

Before diving in, here are some interesting references and resources on statistics that will help you follow this lesson more effectively:

- [Introduction to Hypothesis Testing (YouTube)](https://www.youtube.com/watch?v=0oc49DyA3hU)
- [Null vs. Alternative Hypotheses — Scribbr](https://www.scribbr.com/statistics/null-and-alternative-hypotheses/)
- [Statistics Crash Course (YouTube)](https://www.youtube.com/watch?v=kyjlxsLW1Is)
- [Simplilearn: Hypothesis Testing in Statistics](https://www.simplilearn.com/tutorials/statistics-tutorial/hypothesis-testing-in-statistics)

## The big idea: hypothesis testing

In business, science, or product design, you can't just rely on gut feelings. A manager might say, "Version B of our website feels better," or a teacher might say, "My new teaching method works better." But without evidence, these are just opinions. Hypothesis testing is the statistical way to check: are the differences we see real, or could they just be random chance?

Hypothesis testing is used anytime you have to make a decision based on data — in medicine, business, education, or even sports. At the end of a hypothesis test, we don't prove something 100%; we just decide if we have enough evidence to support one side. Hypothesis testing doesn't give us certainty: it helps us decide if there's enough evidence to support a claim, or if the difference we see could just be random chance.

![Hypothesis Testing Visualization](resources/04_hypothesis_testing_1.png)

The figure above illustrates how this works in practice:

- The *x-axis* shows the range of possible outcomes (for example, different possible average scores).
- The *y-axis* shows how likely each outcome is — this is called *probability density* (how common each value is if the null hypothesis is true).
- The *blue curve* shows what we would expect to see if the *null hypothesis (H₀)* were true (i.e., no real difference).
- The *shaded areas on the sides* represent rare outcomes — results that are unlikely if H₀ is true.
- If our observed result falls into those shaded regions, we *reject the null hypothesis*, because the outcome is too unlikely to be explained by random chance alone.

*Source: Adapted from www.analyticssteps.com/blogs/what-hypothesis-testing-types-and-methods*

## Null hypothesis (H₀): no effect, no difference

The null hypothesis is our starting assumption. It says: "Nothing special is happening." In medicine, H₀ might mean the new drug has no effect compared to the old one; in business, that the new website design performs the same as the old one; in education, that a new teaching method doesn't change student scores. In plain words: the null hypothesis assumes that any difference we see is just random chance, not a real effect.

Think of H₀ like the presumption of innocence in a courtroom — we assume innocence by default until strong evidence shows otherwise. The null hypothesis gives us a baseline to test against. Without it, we'd jump to conclusions every time we saw a small difference in the data, even if it was just random noise.

## Alternative hypothesis (H₁): there is an effect

The alternative hypothesis is the opposite of the null. It says: "Something is happening." In medicine, that the new drug does improve recovery; in business, that the new website gets more clicks than the old one; in education, that the new teaching method raises test scores. In simple terms, the alternative hypothesis assumes that the difference is real, not just random chance.

Extending the courtroom analogy: H₁ is the case the prosecution is making. We only accept it if the evidence is strong enough to reject H₀. The alternative hypothesis is what we're actually hoping to support with data — it represents the claim, effect, or improvement we're testing for.

## p-value: probability if the null hypothesis were true

The p-value tells us: "If the null hypothesis (H₀) is true, how likely is it to see results like ours?"

A coin toss helps make this concrete. Imagine you believe a coin is fair (H₀). You flip it 10 times and get 9 heads. Could this happen by chance? Yes, but the probability is very small. That tiny probability is like the p-value — and if it's small enough, it suggests our assumption (fair coin / no effect) may not be true.

Here is a more practical example. Suppose an old ad gave a 5% click rate and a new ad gave a 7% click rate — a difference of 2%. The p-value asks: "If ads really perform the same (H₀), how likely is it to see a difference of 2% or bigger just by random chance?"

- If p = 0.02: only a 2% chance it's just luck → reasonable evidence the new ad is actually better.
- If p = 0.40: a 40% chance it's just luck → too common to rule out chance → we can't say the new ad is better.

## Significance level (α)

The significance level is the threshold we choose *before* testing — most commonly 0.05, meaning we're okay with being wrong 5% of the time when we say there's a real effect.

If p < α, the result is statistically significant and we support H₁. If p ≥ α, the result is not strong enough and we stick with H₀.

This discussion has been somewhat abstract. Let's look at a concrete example to see how all these pieces fit together in practice.

## The t-test: Comparing Means 📊

Before moving on, you might want to watch this short introduction to
t-tests: https://www.youtube.com/watch?v=VekJxtk4BYM

### A practical question

Suppose Class A uses an old textbook and Class B uses a new one. Both
classes take the same test. We want to know: did the new textbook
actually improve scores, or could the difference just be random chance?

This is exactly the kind of question a t-test helps answer.

### Example: Two separate groups

The students in Class A are not the same students as in Class B. These
are independent groups, so we use what's called an *independent samples t-test*.

``` python
from scipy import stats

class_a = [65, 70, 68, 72, 66, 64, 69, 71]
class_b = [75, 78, 74, 80, 76, 79, 77, 81]

t_stat, p_val = stats.ttest_ind(class_a, class_b)

print("t-statistic:", t_stat)
print("p-value:", p_val)

if p_val < 0.05:
    print("The difference is statistically significant.")
else:
    print("No statistically significant difference detected.")
```

If the p-value is small, typically less than 0.05, we conclude that the
difference between the group averages is unlikely to be due to random
variation alone.

In plain language:

"The difference in average scores is unlikely to be due to chance, so
the new textbook likely improved performance."

### Visualizing the groups

Numbers are important, but pictures build intuition.

``` python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class_a = [65, 70, 68, 72, 66, 64, 69, 71]
class_b = [75, 78, 74, 80, 76, 79, 77, 81]

t_stat, p_val = stats.ttest_ind(class_a, class_b, equal_var=True)

plt.hist(class_a, bins=5, alpha=0.5, label="Class A (Old Book)", color="blue")
plt.hist(class_b, bins=5, alpha=0.5, label="Class B (New Book)", color="orange")

mean_a = np.mean(class_a)
mean_b = np.mean(class_b)

plt.axvline(mean_a, color="blue", linestyle="dashed", linewidth=1)
plt.axvline(mean_b, color="orange", linestyle="dashed", linewidth=1)

text = (
    f"Mean A: {mean_a:.1f}\n"
    f"Mean B: {mean_b:.1f}\n"
    f"t = {t_stat:.2f}\n"
    f"p = {p_val:.2e}"
)

plt.text(0.02, 0.98, text,
         transform=plt.gca().transAxes,
         verticalalignment="top",
         bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

plt.title("Class Test Scores Comparison")
plt.xlabel("Scores")
plt.ylabel("Number of Students")
plt.legend()
plt.show()
```
Output: 

![Hypothesis Testing
Visualization](resources/04_hypothesis_testing_2.png)

The image makes the result intuitive. Class B's scores sit noticeably
higher overall, with no overlap.

### What a t-test actually does

A t-test compares two averages while taking into account how much
variation there is within each group and how many observations we have.
It produces a *t-statistic*, which measures how large the difference is
relative to variability, and a *p-value*, which tells us how surprising
that difference would be if there were really no effect.

Once you understand that core idea, the rest is about choosing the
correct version for your data structure.

### Choosing the right type of t-test

There are three common versions of the t-test. The correct one depends
 on how your data is organized.

#### Independent samples t-test

Use this when comparing two separate, unrelated groups, like the two
classes above.

Function:

``` python
stats.ttest_ind(group1, group2)
```

#### Paired t-test

Use this when the same subjects are measured twice, for example before
and after a tutoring program. Each before score is linked to a specific
after score.

``` python
before = [62, 68, 71, 65, 70, 67, 63, 69]
after  = [70, 75, 74, 72, 78, 71, 69, 74]

t_stat, p_val = stats.ttest_rel(before, after)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.6f}")
```

Because we account for individual differences, paired tests are often
more sensitive than independent tests.

#### One-sample t-test

This version compares a group's mean to a known reference value. For
example, is a class average different from a national benchmark of 70?

``` python
scores = [68, 72, 74, 69, 71, 73, 70, 75]

t_stat, p_val = stats.ttest_1samp(scores, 70)
```

You will use this less often, but it is helpful to recognize.

### One-tailed vs two-tailed tests

By default, t-tests are two-tailed. That means we are checking for any
difference, whether one group is higher or lower than the other.

If you have strong theoretical justification to test only one direction,
you can run a one-tailed test. In SciPy, this is controlled with the
`alternative` parameter.

``` python
# Two-tailed (default)
stats.ttest_ind(class_a, class_b, alternative="two-sided")

# One-tailed: is class_a less than class_b?
stats.ttest_ind(class_a, class_b, alternative="less")
```

In most practical situations, two-tailed tests are the safer default
unless you have a strong reason to commit to one direction in advance.

### Writing conclusions clearly

Instead of saying "reject the null hypothesis," explain the result in
plain language.

For example:

"The difference in average scores is unlikely to be due to random chance
(p <= 0.05), so we conclude the new textbook likely improved
performance."

Being able to translate statistical output into clear language is an
important professional skill.

## A/B testing in the real world 🔬

What we just learned about hypothesis testing is exactly what powers *A/B testing*, one of the most common industry practices. A/B testing is when you compare two versions of something — a website button, an email subject line, a product feature — to see which one performs better. Version A is often the "control" (the current version) and Version B is the "treatment" (the new variation). You then collect data and use hypothesis testing to check whether there's a real difference in outcomes (e.g., clicks, purchases), or whether the difference could be due to random chance.

This is exactly the same statistical framework we've been learning: a null hypothesis that there's no difference between A and B, an alternative that there is, and a test statistic and p-value to help you decide. You've now seen the statistical foundation behind one of the most widely used tools in data science and industry.

## Important notes and limitations

### Assumptions of the t-test

The t-test relies on a few assumptions: the data should be roughly normally distributed, and observations should be independent of each other. If these assumptions are badly violated — highly skewed data, very small samples — other tests may be more appropriate. Non-parametric alternatives like the Mann-Whitney U test don't assume any particular distribution and can be a useful fallback.

### Just scratching the surface
Hypothesis testing is a core operation in many data pipelines. From comparing test scores to running large-scale A/B experiments, it helps us move from hunches to evidence-based decisions.

What we covered here are the fundamentals. There are many statistical tests for many different situations — not just comparing means, but also testing medians, standard deviations, and correlations (which we'll look at in the next lesson). Our goal here was to give a sense for what hypothesis testing is and how it works in practice, not to cover every possible test. As you encounter new problems, you'll learn which tests are appropriate for different data structures and research questions.


