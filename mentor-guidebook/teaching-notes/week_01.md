# Week 1: Introduction to Analysis

## Overview

Students revisited Python fundamentals through the lens of data analysis, working with three core libraries: Pandas (tabular data), NumPy (numerical computation), and Matplotlib (visualization). They also covered descriptive statistics, hypothesis testing, and correlation — and finished the week by building their first automated data pipeline using Prefect.

## Key Concepts

**Pandas and NumPy** — Pandas DataFrames are the workhorse of data analysis in Python: think of them as spreadsheets you can program. NumPy provides the fast math underneath — it's why Pandas and scikit-learn are so much faster than plain Python loops.

**Descriptive statistics** — Mean, median, variance, and standard deviation describe the shape of data. Visualizations (histograms, boxplots) are usually more informative than numbers alone when looking for skew or outliers.

**Hypothesis testing** — A t-test answers: "Is this difference real, or could it be random noise?" The p-value is the probability you'd see that result by chance. The conventional threshold is p < 0.05, but students should understand that this is a convention, not a law.

**Correlation** — Pearson correlation (r) measures linear association between two variables. Key takeaway: correlation does not imply causation. Anscombe's Quartet (four datasets with identical statistics but totally different shapes) is a useful demonstration.

**Prefect pipelines** — Prefect lets you wrap Python functions as `@task`s and combine them into a `@flow`. The main benefits are automatic logging, retry logic, and a dashboard to monitor runs. This is students' first exposure to the idea that code can be *orchestrated*, not just run once.

## Common Questions

- **"When do I use a DataFrame vs. a NumPy array?"** — DataFrames when you have labeled, mixed-type tabular data (like a spreadsheet). NumPy arrays when you're doing pure math on a grid of numbers.
- **"What does a p-value actually mean?"** — It's the probability of seeing your result (or something more extreme) *if the null hypothesis were true*. It does not tell you the probability that your hypothesis is correct.
- **"What's the point of Prefect if I can just run a script?"** — For one-off analysis, you probably don't need it. Prefect is for production pipelines that run repeatedly, need retries when things fail, and need logs so you can debug when something breaks at 3am.

## Watch Out For

- **Environment setup** — Students may have dependency issues getting everything installed. Make sure they're using the `uv` environment from the course setup instructions, not a system Python.
- **Confusing p < 0.05 with "proof"** — A significant p-value is evidence against the null hypothesis, not proof of anything. This misconception is extremely common.
- **Prefect server not running** — The Prefect dashboard (`prefect server start`) needs to be running in a separate terminal when students run their flows. This trips people up in the assignment.

## Suggested Activities

1. **Statistics intuition check:** Show a histogram of a skewed dataset. Ask: "Would the mean or median be a better summary here? Why?" Then reveal both values and discuss.

2. **Correlation vs. causation:** Share a fun spurious correlation (e.g., ice cream sales and drowning deaths both rise in summer). Ask students to explain why the correlation exists without causation, and how they'd design a study to separate the two.

3. **Pipeline walkthrough:** Have a student screen-share their Prefect flow running. Ask the group: "What would happen if the API call in the extract step failed? How does the pipeline handle that?" Walk through retries and logging together.
