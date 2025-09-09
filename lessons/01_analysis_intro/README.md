# Week 1: Introduction to Analysis

This week introduces the foundations of analysis you’ll use throughout the course. We’ll review core concepts from Python 100, then explore basic ideas from probability and statistics, and end with producing reproducible analysis utilities using pipelines. 

> For an introduction to the course as a whole, and a discussion of how to set up your environment, please see the [Welcome](../00_Welcome.md) lesson. 

## Topics
1. [Python 100 Review](01_python100_review.md)  
Reactivate core [Python 100](https://github.com/Code-the-Dream-School/python-essentials) skills, in particular how to use Pandas to load and analyze data, NumPy for numerical operations, and Matplotlib for visualization. This is a quick review to kick out the rust and get our heads back in the game. 

2. [Descriptive statistics and distributions](02_distributions.md)  
Build intuition for distributions and probability, as well as measures of central tendency (mean/median) and spread (variance and standard deviation). Use visuals like histograms and boxplots.

3. [Hypothesis testing](03_hypothesis_testing.md)  
Basics of hypothesis testing. What is a p value? Play up how important this is: it's about going beyond intuition. When you want to show business value, you need evidence, not just guesses. Explain traditional statistical hypothesis testing, the meaning of p-values, and how to interpret them.

4. [Correlation](04_correlation.md)  
Here the focus shifts to relationships *between* variables, especially linear correlation. What does Pearson correlation measure, exactly? Discuss common pitfalls: how correlation strength differs from significance. For instance, with huge datasets, even weak correlations can be statistically significant. Show why: *always plot your data*.

5. [Pipelines](05_pipelines.md)  
Motivate why pipelines matter: we don’t want to re-run analysis manually every time. We want reproducible, modular workflows, especially in the context of cloud computing (tease later P200 weeks). Prefect enables orchestration of analysis steps using tasks and flows.

