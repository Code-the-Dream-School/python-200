


# Introduction to Data Analysis

> ### Contributor Note 
> This is a working outline for Python 200, Week 1. Each section below needs to be expanded into a full lesson. Use the code ideas and goals as a starting point — feel free to add examples, exercises, and links to visualizations or datasets. 

Welcome to the first lesson in Python 200, Introduction to Data analysis! 

To fill in later: brief motivational preview here. Briefly explain why this lesson matters, what students will be able to do by the end, and what topics will be covered. Keep it tight and motivating.

> For an introduction to the course, and a discussion of how to set up your environment, please see the [Welcome](00_Welcome.md) lesson. 

# Table of Contents
1. [Python 100 Refresher](#1-python-100-refresher)
2. [Descriptive statistics and distributions](#2-descriptive-statsistics-and-distributions)
3. [Hypothesis testing](#3-hypothesis-testing)
4. [Correlation](#4-correlation)
5. [Pipelines](#5-pipelines)
6. [Wrap-up](#6-wrap-up)

## 1. Python 100 Refresher
Reactivate core [Python 100](https://github.com/Code-the-Dream-School/python-essentials) skills, in particular:

- How to use Pandas to load, clean, and analyze data 
- Using Numpy for numerical operations
- Using Matplotlib to visualize. 

This is a useful time to get students brains back in the game! 

### Code ideas
- Pandas: `read_csv`, `info`, `head`, `tail`, `groupby`
- NumPy: basic array creation and operations (do not go too deeply into NumPy)
- Matplotlib: histograms, scatterplots, line plots


## 2. Descriptive statistics and distributions
Build intuition for distributions and probability, as well as measures of central tendency (mean/median) and spread (variance and standard deviation). Use visuals like histograms and boxplots.

### Code ideas
- Compute summary stats for 2–3 subgroups of a sample dataset.
- Explore robustness to outliers in normal vs skewed distributions.
- Simulate a normal vs biased die and discuss.

## 3. Hypothesis testing
Play up how important this is: it's about going beyond intuition. When you want to show business value, you need evidence, not just guesses. Explain traditional statistical hypothesis testing, the meaning of p-values, and how to interpret them.

Focus on t-tests for comparing means, defining null vs alternative hypotheses, and writing conclusions in plain language. A/B testing is a great example.

### Code ideas
- Find or generate a simple A/B dataset.
- Perform a t-test using `scipy.stats.ttest_ind`.
- Plot group distributions (KDE), show means and standard deviations.
- Work on interpreting p-values.

## 4. Correlation 
Here the focus shifts to relationships *between* variables, especially linear correlation. What does Pearson correlation measure, exactly?

Discuss common pitfalls: how correlation strength differs from significance. For instance, with huge datasets, even weak correlations can be statistically significant. Show why: *always plot your data*.

### Code ideas
- Show correlation and scatterplots for values from -1 to +1 in 0.2 increments.
- Use `np.corrcoef`, `seaborn.heatmap`, and scatterplots.
- Show effect of sample size on p-value (e.g., n=20 vs n=10,000).
- Demonstrate that high sample size can make a tiny correlation look significant.


## 5. Pipelines
Motivate why pipelines matter: we don’t want to re-run analysis manually every time. We want reproducible, modular workflows, especially in the context of cloud computing (tease later P200 weeks).

Prefect enables orchestration of analysis steps using decorators and tasks.

> You can mention Apache Airflow here as a more production-grade tool, to be introduced later in the cloud module.

### Code ideas
- Build a Prefect pipeline that integrates:
  - `load_data()`
  - `clean_data()`
  - `describe_and_plot()`
  - `run_ttest()`
  - `report_results()`
- Wrap steps in a `@flow()` function and run end-to-end.

## 6. Wrap-up 
Summarize key takeaways:
- Working with data requires descriptive statistics, hypothesis testing, and understanding relationships.
- Significance ≠ importance -- context matters.
- Pipelines are crucial in modern data engineering and cloud computing. 


