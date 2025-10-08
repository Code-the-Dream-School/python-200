
# Hypothesis Testing

Let’s start a new chapter on Hypothesis testing. Evidence over intuition.

# Here are a few essential resources that will you follow the lessons efficiently:
1. https://www.youtube.com/watch?v=0oc49DyA3hU&t=5s
2. https://www.youtube.com/watch?v=2fgQ_8AKhJY
3. https://www.simplilearn.com/tutorials/statistics-tutorial/hypothesis-testing-in-statistics

---

## Important Notes & Limitations ⚠️

Before we move on, it’s important to highlight a few things about hypothesis testing:

### 1. Assumptions of the t-test
The common **t-test** relies on some assumptions:
- Data should be **roughly normally distributed**.  
- Observations should be **independent** from each other.  

If these assumptions are violated, other tests (e.g., **non-parametric tests**) may be more appropriate.

---

### 2. One-Sided vs Two-Sided Tests
- A **two-sided test** checks for **any difference** between two groups (values could be higher or lower).  
- A **one-sided test** checks for a difference in **one direction only** (e.g., “Group A mean is greater than Group B mean”).  

In practice, two-sided tests are more common unless you have a strong reason to test only one direction.

---

### 3. Just Scratching the Surface
What we covered here is just the beginning.  
There are many statistical tests for many measures, not just the mean:  
- Median  
- Standard deviation  
- Correlation (we’ll dive deeper into this in a later lesson)  

As you move forward, you’ll learn when and how to choose the right test depending on the type of data and research question.

---

📺 As wee begin with the lesson here are some interesting references & resource on Statistics which will help you with follow the lesson more effectively:  

- 📺 [Introduction to Hypothesis Testing (YouTube)](https://www.youtube.com/watch?v=0oc49DyA3hU)  
- 📖 [Null vs. Alternative Hypotheses — Scribbr](https://www.scribbr.com/statistics/null-and-alternative-hypotheses/)  
- 📺 [Statistics Crash Course (YouTube)](https://www.youtube.com/watch?v=kyjlxsLW1Is)  
- 📖 [Simplilearn: Hypothesis Testing in Statistics](https://www.simplilearn.com/tutorials/statistics-tutorial/hypothesis-testing-in-statistics)  

---


Let's learn Hypothesis Testing:

Let’s think of it this way: In business, science, or product design, you can’t just rely on gut feelings:  
- A manager might say, “Version B of our website feels better.”  
- A teacher might say, “My new teaching method works better.”  

But without evidence, these are just opinions.  
Hypothesis testing is the statistical way to check: Are the differences we see real, or could they just be random chance?

---

## The Big Idea: Hypothesis Testing

A structured way to test an assumption using data.

**Decision-making connection:**  
“Hypothesis testing is used anytime you have to make a decision based on data, in medicine, business, education, or even sports.”

**Outcome reminder:**  
At the end of a hypothesis test, we don’t prove something 100%, we just decide if we have enough evidence to support one side.

⚠️ Hypothesis testing ≠ absolute truth.  
Lastly, hypothesis testing doesn’t give us certainty, it helps us decide if there’s enough evidence to support a claim, or if the difference we see could just be random chance.

<img width="648" height="306" alt="Screenshot 2025-09-09 at 3 31 25 PM" src="https://github.com/user-attachments/assets/c7a158ef-6530-45c3-9319-3041e3cfe2a9" />

---

## Null Hypothesis (H₀): No Effect / No Difference

The null hypothesis is our starting assumption:  
It says: “Nothing special is happening.”  
- In medicine: The new drug has no effect compared to the old one.  
- In business: The new website design performs the same as the old one.  
- In school: A new teaching method doesn’t change student scores.  

In plain words: The null hypothesis assumes that any difference we see is just random chance, not a real effect.

**Courtroom Example:**  
H₀ (Null) = “The accused is innocent.”  
We assume innocence by default until strong evidence shows otherwise.

Why is this important?  
The null hypothesis gives us a baseline to test against.  
Without it, we’d jump to conclusions every time we saw a small difference, even if it was just random noise.

---

## Alternative Hypothesis (H₁): There is an Effect / Difference

The alternative hypothesis is the opposite of the null.  
It says: “Something is happening.”  
- In medicine: The new drug does improve recovery.  
- In business: The new website gets more clicks than the old one.  
- In school: The new teaching method raises test scores.  

In simple terms: The alternative hypothesis assumes that the difference is real, not just random chance.

**Courtroom Analogy:**  
H₁ (Alternative) = “The accused is guilty.”  
We only accept this if the evidence is strong enough to reject H₀ (innocence).

Why is this important?  
The alternative hypothesis is what we’re actually hoping to prove with data.  
It represents the claim, effect, or improvement we’re testing for.

---

## p-value: Probability Under the Null

The p-value tells us: “If the null hypothesis (H₀) is true, how likely is it to see results as extreme as ours, just by random chance?”

**Coin Toss Analogy:**  
Imagine you believe a coin is fair (H₀).  
You flip it 10 times and get 9 heads.  
Could this happen by chance? Yes, but the probability is very small.  
That tiny probability is like the p-value.

If the p-value is very small, it suggests our assumption (fair coin / no effect) may not be true.

**Advertisement Example:**  
- Old ad gave a 5% click rate.  
- New ad gave a 7% click rate.  
- Difference = 2%.  

Now, the p-value asks:  
“If ads really perform the same (H₀), how likely is it to see a difference of 2% (or bigger) just by random chance?”  

- If p = 0.02 → Only a 2% chance it’s just luck → evidence the new ad is actually better.  
- If p = 0.40 → A 40% chance it’s just luck → too common to rule out chance → we can’t say the new ad is better.  

---

## Significance Level (α)

The significance level is the threshold we choose before testing (often 0.05 = 5%).  

Meaning: “I’m okay with being wrong 5% of the time when I say there’s a real effect.”

**Decision rule:**  
- If p < α → result is meaningful, support H₁.  
- If p > α → result not strong enough, stick with H₀.  

---

## t-Test for Comparing Means

A t-test is a tool we use when we want to check if two groups really have different averages, or if the difference might just be luck.

**Example: Two Classes' Test Scores**  
- Class A uses the old textbook.  
- Class B uses a new textbook.  
- Both classes take the same test.  

We want to know: “Did the new textbook actually help, or are the score differences just random?”  

This is exactly what a t-test does: it compares the average scores of the two groups to see if the difference is real.

**Steps:**  
1. Collect data from two groups (old vs new).  
2. Use the t-test to compare their averages.  
3. Look at the p-value:  
   - If it’s small → difference is real.  
   - If it’s big → difference could just be random chance.  

**Example Code:**  
```python
from scipy import stats

# Step 1: Collect data
class_a = [65, 70, 68, 72, 66, 64, 69, 71]
class_b = [75, 78, 74, 80, 76, 79, 77, 81]

# Step 2: Perform independent t-test
t_stat, p_val = stats.ttest_ind(class_a, class_b)

print("t-statistic:", t_stat)
print("p-value:", p_val)

# Step 3: Interpret results
if p_val < 0.05:
    print("The difference is significant: the new textbook likely improved scores.")
else:
    print("No significant difference: the new textbook may not have made a real impact.")
```

**Output:**  
```
t-statistic: -6.985114284057083
p-value: 6.3969825252044474e-06
The difference is significant: the new textbook likely improved scores.
```

📌 Note: This kind of experiment is often called **A/B testing** in business.

---

## Visualizing the Groups

So far, we ran a t-test on the two classes’ scores. That gave us numbers (t-statistic and p-value).  
But numbers alone can feel abstract. To make it intuitive, we can visualize the same raw data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Same data as before ---
class_a = [65, 70, 68, 72, 66, 64, 69, 71]
class_b = [75, 78, 74, 80, 76, 79, 77, 81]

# 1) Run the t-test
t_stat, p_val = stats.ttest_ind(class_a, class_b, equal_var=True)

# 2) Visualization
plt.hist(class_a, bins=5, alpha=0.5, label="Class A (Old Book)", color="blue")
plt.hist(class_b, bins=5, alpha=0.5, label="Class B (New Book)", color="orange")

# Show group means
mean_a = np.mean(class_a)
mean_b = np.mean(class_b)
plt.axvline(mean_a, color='blue', linestyle='dashed', linewidth=1)
plt.axvline(mean_b, color='orange', linestyle='dashed', linewidth=1)

# Annotate results
text = (
    f"Mean A: {mean_a:.1f}\n"
    f"Mean B: {mean_b:.1f}\n"
    f"t = {t_stat:.2f}\n"
    f"p = {p_val:.2e}"
)
plt.text(0.02, 0.98, text, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.title("Class Test Scores Comparison")
plt.xlabel("Scores")
plt.ylabel("Number of Students")
plt.legend()
plt.show()
```
<img width="557" height="397" alt="Screenshot 2025-09-09 at 3 32 15 PM" src="https://github.com/user-attachments/assets/8ad241bf-89b0-4390-a9ee-66a02670e376" />

**Explanation:**  
- Step 1 (t-test): Calculated t-statistic & p-value.  
- Step 2 (plot): Drew histograms & marked averages.  
- Together: Numbers tell us if the difference is real; the picture helps us see that Class B’s scores sit higher overall.  

---

## Sharing Outcomes: Writing Conclusions in Plain Language

Statistical results are often written in technical terms like:  
“p < 0.05, reject null hypothesis.”  

But in real projects, you need to explain results so anyone can understand:  
👉 “We are 95% confident that students using the new textbook (Class B) scored higher than those using the old textbook (Class A).”  

---

## A/B Testing in the Real World 🅰️/🅱️

You’ve actually just learned the foundation of **A/B testing**, one of the most common applications of hypothesis testing in the real world.  

A/B testing is used everywhere — from tech companies like Google and Netflix to marketing teams and educators. It’s how we make **data-driven decisions** instead of relying on intuition.

### What is A/B Testing?

**A/B testing** means comparing **two versions** of something to see which one performs better:  
- Version **A** → the “control” (the current or original version)  
- Version **B** → the “treatment” (the new or experimental version)  

You then measure performance (like clicks, conversions, test scores, or engagement) and use **hypothesis testing** to decide if version B truly performs better — or if the observed difference could just be due to random chance.

### Real-World Examples

- **Tech & Product:**  
  A company tests two button colors — blue vs. green — to see which one gets more clicks.  
- **Marketing:**  
  Two email subject lines are tested to find which one has a higher open rate.  
- **Education:**  
  A teacher tries a new learning app and compares student test scores.  

Each of these examples uses the exact same logic you’ve already learned:
- **Null hypothesis (H₀):** There’s no difference between A and B.  
- **Alternative hypothesis (H₁):** There *is* a difference — the new version performs better.  
- **p-value:** Tells us how likely it is that the observed difference is due to random chance.  

If the p-value is low enough (e.g., < 0.05), we have evidence that **B truly outperforms A**.

### Why It Matters

A/B testing is everywhere in data science and analytics:
- Websites and apps run **hundreds of A/B tests daily**.  
- Marketing teams use it to refine campaigns.  
- Product teams use it to improve features based on user behavior.  

Every time you see a “new design,” “recommendation,” or “experiment” online, it likely started as an **A/B test** powered by the same hypothesis testing process you’ve just learned.

---

✅ So, congratulations, by understanding hypothesis testing, you now understand the **core logic behind A/B testing**, one of the most practical, powerful tools in modern data science!

---

✅ Lastly, hypothesis testing is a core skill in data science.  
From comparing test scores to running large-scale A/B experiments, it helps us move from **hunches** to **evidence-based decisions**.  
We’ll keep building on this foundation in upcoming lessons.


