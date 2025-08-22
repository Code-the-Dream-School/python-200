# Pipelines with Prefect

Imagine running your analysis once, then being asked to run it again tomorrow, next week, or on a larger dataset. Manually re-running each step—loading, cleaning, analyzing, and reporting—would be painful, error-prone, and time-consuming.

**Pipelines solve this**. They give us reproducible, modular workflows where each step is defined and orchestrated. This becomes especially powerful in **cloud computing** and production data environments (This is a topic we'll explore in more detail in future lessons of Python 200).

### Learning objective: 
In this lesson, you’ll learn:

- Why pipelines matter for reproducibility and modularity.
- How to use Prefect to build and orchestrate pipelines.
- How to wrap multiple analysis steps into one end-to-end workflow.
By the end, you'll be able to design a complete data analysis workflow that loads data, cleans it, explores it, performs statistical tests, and reports results—all with a single command.

### Table of Contents
1. Understanding Data Pipelines
2. Prefect Basics: Flows and Tasks
3. Building Your First Prefect Pipeline
4. Wrap-up

## 1. Understanding Data Pipelines

What is a **Data Pipeline**?
A **data pipeline** is a series of connected data processing steps where the output of one step becomes the input of the next. Think of it like a factory assembly line for data-raw materials (data) enter at one end, go through various transformation stations, and emerge as a finished product (insights, reports, or processed datasets).

```python
Raw Data → Load → Clean → Analyze → Visualize → Report
```
- Without pipelines → manual re-runs, messy scripts, hard to reproduce results.
- With pipelines → clear modular steps, easy to run end-to-end, scalable to cloud systems.

2. ## Prefect

Now that we understand why pipelines are essential, let's look at a tool that helps us build them: **Prefect**.  

Prefect is an open-source workflow orchestration tool that helps you define, run, and monitor your data pipelines. It's designed to make data workflows more **robust** and **observable**.

---

### Core Concepts in Prefect

At its core, Prefect uses two main concepts to define your pipeline:

1. **`@task()` decorator**  
   - Transforms a regular Python function into a Prefect **task**.  
   - A task is a fundamental unit of work within a pipeline.  
   - When a task runs, Prefect:  
     - Tracks its state (*running, successful, failed*)  
     - Logs its output  
     - Handles retries if something goes wrong  

2. **`@flow()` decorator**  
   - Transforms a Python function into a Prefect **flow**.  
   - A flow is a collection of tasks and defines the overall workflow      logic. It orchestrates the execution order of your tasks and handles dependencies.  
   - Provides a single entry point to run your entire pipeline. 

---

### Simple Example

Let's see a super simple example to get the idea:

```python
from prefect import task, flow

# Define a task
@task
def say_hello(name):
    print(f"Hello, {name}!")

# Define a flow that uses the task
@flow
def my_pipeline():
    say_hello("Niharika")
    say_hello("Students")

# To run the flow
# This condition checks if the current script is the one being executed directly.
if __name__ == "__main__":  
    my_pipeline()
```
- Each function marked with @task becomes a pipeline step.
- The @flow function orchestrates those steps.
- Running my_pipeline() executes the whole workflow. 

## 3. Building Your First Prefect Pipeline
Let's build a data analysis pipeline using Prefect step by step.

0) Setup
```python
# First, install Prefect:
pip install prefect pandas matplotlib scipy
```
Step 1: Imports & Data Loader `(load_data)`:

Imports: We import necessary libraries: pandas for data manipulation, prefect for our pipeline magic, matplotlib.pyplot for plotting, scipy.stats for the t-test, and os for file path checks.

```python

from prefect import task, flow
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

@task
def load_data() -> pd.DataFrame:
    """
    Create a tiny, readable dataset:
      - Class: which class the student is in ('A' or 'B')
      - Score: exam score (numeric)
    """
    data = {
        "Class": ["A","A","A","A","A",  "B","B","B","B","B"],
        "Score": [65, 70, 68, 72, 66,    78, 82, 80, 79, 81]
    }
    df = pd.DataFrame(data)
    print("Exam scores loaded")
    return df

```

- load_data returns a DataFrame, which later tasks will receive as input.

Note: This is just a shortcut to build repeated labels:
```python
["A"]*5 + ["B"]*5
# → ["A","A","A","A","A","B","B","B","B","B"]
```

Step 2: Clean the Data `(clean_data)`

```python
@task
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      1) Ensure Score is numeric.
      2) Drop missing values.
    (This is minimal on purpose for clarity.)
    """
    df = df.copy()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df.dropna(subset=["Score", "Class"])
    print("Data cleaned")
    return df
```
Real data is messy. Converting to numeric and dropping NAs prevents errors later (plotting, stats). 
