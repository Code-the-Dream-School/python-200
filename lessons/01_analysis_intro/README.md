# Week 1: Introduction to Analysis
Welcome to the first week of Python 200! This week we will lay the groundwork for the rest of the course. We'll review core concepts from Python 100, explore basic ideas from probability and statistics, and end with reproducible analysis utilities using pipelines.

> For an introduction to the course as a whole, and a discussion of how to set up your environment, please see the [Welcome](../README.md) page. 

## Topics

### Python 100 Review
We are covering some of the main Python packages from Python 100, but in a new context. If you are already comfortable with these libraries, this will be a quick refresher. If it has been a while since you used them, this will be a good opportunity to get back up to speed. Note we are assuming you remember many basics of the Python standard library. 

Also, we will not be reviewing SQL this week: we are holding off on reviewing SQL until we use it more in week 6 when we build AI pipelines that interact with databases. 

1. [Pandas](01_pandas.md)
A review of the core Pandas skills from Python 100: loading datasets, exploring DataFrames, filtering, grouping, and summarizing. If it has been a while, this lesson will bring everything back quickly.

2. [NumPy](02_numpy.md)
NumPy is the numerical foundation underlying nearly every data science library you'll use in this course. We'll revisit arrays and vectorized operations, and develop a clearer picture of how NumPy relates to pandas, scikit-learn, and PyTorch.

3. [Matplotlib](03_matplotlib.md)
Visualization is not just a presentation tool — it's an analytical one. We'll review core Matplotlib patterns and reinforce the habit of looking at your data before drawing conclusions from it.

### New material
4. [Describing data](04_describing_data.md)
Before you can reason about patterns in data, you need tools to describe it: measures of central tendency, spread, and shape. We'll work through these ideas visually, using histograms and boxplots to build intuition.

5. [Hypothesis Testing](05_hypothesis_testing.md)
Intuition is a good starting point, but it's not evidence. This lesson covers the foundations of statistical hypothesis testing — what a p-value actually measures, and how to interpret it when you need to make a defensible claim from data.

6. [Correlation](06_correlation.md)
What does it mean for two variables to be related? We'll look at Pearson correlation in detail.

7. [Pipelines](07_pipelines.md)
Once your analysis works, you'll want to run it again reliably — on new data, on a schedule, or in the cloud. This lesson introduces data pipelines with Prefect.

## Week 1 Assignments
Once you finish the lessons, head on over to the [assignments](../../assignments/README.md) to get more hands-on practice with the material. 

