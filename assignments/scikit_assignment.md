# Lesson 1 Assignment: Introduction to scikit-learn and the ML Ecosystem

In this assignment, you’ll practice the core scikit-learn workflow you learned in Lesson 1:

**create → fit → predict → evaluate**

You’ll do a few short warmups to build muscle memory, then complete one mini-project that feels closer to real ML work.

## How to Submit

Submit **one** of the following (your instructor will tell you which one your cohort is using):

- **GitHub:** Push your notebook/script to your repo and share the link (and/or open a PR).
- **Kaggle/Colab:** Share a link with outputs saved (plots visible + cells run).

At the top of your notebook, include a short write-up answering:
1) What did you build?  
2) What did you learn?  
3) What would you try next if you had more time?

---

## Part 1: Warmup Exercises (No AI)

These are designed to be quick and approachable. Please **turn off Copilot/AI help** for Part 1.

### Task 1 — Quick API Check: “Create → Fit → Predict”
Create a tiny dataset in NumPy:

- `X = [[1], [2], [3], [4], [5]]`
- `y = [2, 4, 6, 8, 10]`

Train a `LinearRegression` model and predict the output for `X = [[6]]`.

**Deliverable:** Print the prediction and write 1–2 sentences explaining what the model learned.

---

### Task 2 — Train/Test Split Practice
Using a built-in scikit-learn dataset like **Wine**:

1. Load with `load_wine(as_frame=True)`
2. Create `X` and `y`
3. Split into train/test with:
   - `test_size=0.2`
   - `random_state=0`
   - `stratify=y`

**Deliverable:** Print the shapes of `X_train`, `X_test`, `y_train`, `y_test`, and write 1 sentence explaining what `stratify=y` does.

---

### Task 3 — Your First Classifier (Not KNN): Decision Tree
Train a `DecisionTreeClassifier(random_state=0)` on the Wine dataset.

1. Fit on `X_train, y_train`
2. Predict on `X_test`
3. Print accuracy and a `classification_report`

**Deliverable:** Print the metrics and write 3–5 sentences:
- What seems “good” about the results?
- What makes this a **classifier**?
- What is one reason the Wine dataset might be easier/harder than real-world data?

---

### Task 4 — Unsupervised Learning Warmup: K-Means
Use `make_blobs` to create synthetic data:

- `n_samples=200`
- `centers=4`
- `cluster_std=0.8`
- `random_state=42`

Run `KMeans(n_clusters=4, random_state=42)` and plot the points colored by predicted cluster label.

**Deliverable:** One scatter plot + 2–3 sentences explaining what the colors mean.

---

### Task 5 — Reflection (Short)
Answer in 4–6 sentences:

- What is the difference between **supervised** and **unsupervised** learning?
- Which one uses labels (`y`)?
- In this assignment, which tasks were supervised and which were unsupervised?

---

## Part 2: Mini-Project (Pick ONE)

Choose **one** project below. Each should take ~2–6 hours.

---

# Project Option A — Customer Segmentation with K-Means (Classic Real-World Task)

## Goal
You work for a store and want to group customers into segments based on behavior.

## Dataset
Use one of these options:
- **Mall Customers dataset** (common CSV used for clustering exercises), OR
- Create your own “customer” dataset (at least 200 rows) with columns like:
  - `annual_income`
  - `spending_score`
  - `visits_per_month` (optional)

## Requirements
1. Load the data into a DataFrame and show:
   - `.head()`
   - `.info()`
   - `.describe()`

2. Do **light EDA** (2–3 plots total):
   - Scatter plot of two features (like income vs spending)
   - Histogram of at least one feature
   - (Optional) Pairplot if you want

3. Fit K-Means for **k = 2 through 8** and compute **inertia** for each k.
   - Make an elbow plot (k vs inertia)

4. Pick a k and justify it in 2–4 sentences.

5. Fit your final K-Means model and add a `cluster` column to the DataFrame.

6. Visualize clusters:
   - Scatter plot with points colored by cluster

7. Interpret clusters:
   - Write 2–5 sentences describing what each cluster “looks like” (e.g., “high income, high spending”)

## Deliverables
- Notebook with code + plots
- A short written summary at the end:
  - chosen k
  - what the clusters mean
  - one business action you’d take

## Stretch Goals (Optional)
- Standardize features before clustering and compare results.
- Try a second feature pair and see how clusters look.

---

# Project Option B — The scikit-learn Workflow on a Real Dataset (Decision Tree + Evaluation)

## Goal
Practice the full workflow on a real classification dataset using a model that is **not KNN**.

## Dataset
Use **Wine** from scikit-learn:
- `from sklearn.datasets import load_wine`

## Requirements
1. Load the dataset and create a DataFrame.
2. Do quick EDA:
   - Print `df.head()`
   - Plot **one histogram** of a feature you choose
   - Plot **one scatter plot** comparing two features, colored by class
3. Train/test split (`stratify=y`).
4. Train a `DecisionTreeClassifier`.
5. Evaluate using:
   - `accuracy_score`
   - `classification_report`
   - a confusion matrix plot
6. Interpretation (short writing):
   - Which class is easiest to predict? Which is hardest?
   - Name one feature you think might help the tree and why (use your EDA).

## Deliverables
- Notebook with code + outputs + plots
- A short “Results & Interpretation” section (8–12 sentences total)

## Stretch Goals (Optional)
- Try `max_depth=3` and compare results to the default tree.
- Explain: did restricting depth make performance better or worse?

---

## Grading Checklist (What “Complete” Looks Like)

A complete submission has:

- Code runs without errors  
- Uses scikit-learn workflow (create → fit → predict → evaluate)  
- Includes at least 2 plots in the project  
- Includes a short written interpretation (not just numbers)  
- Clear headings + comments so it’s easy to read  

---

## Help (If You Get Stuck)

- Re-read Lesson 1 and the scikit-learn API pattern.
- Ask in Slack with:
  - the error message
  - the cell/code you ran
  - what you expected to happen
- Remember: getting stuck is normal — that’s part of learning.

