# Introduction to Data Pipelines

Imagine running your analysis once, then being asked to run it again tomorrow, next week, or on a larger dataset. Manually re-running each step - loading, cleaning, analyzing, and reporting - would be painful, error-prone, and time-consuming.

Pipelines solve this. They give us reproducible, modular workflows where each step is defined and orchestrated. This approach becomes especially powerful in *cloud computing*, a topic we’ll explore in more detail in future lessons of Python 200.

### Learning objectives: 
By the end of this lesson, you’ll be able to:

- See why pipelines make analysis reproducible, modular, and scalable.
- Build and orchestrate workflows using Prefect tasks and flows.
- Make pipelines resilient and efficient with retries and logging.
- Monitor and debug workflows in real time using the Orion dashboard.
- Run a full data analysis workflow - load, clean, explore, analyze, and report - with a single command.

### Table of Contents
1. Understanding Data Pipelines
2. Prefect
3. Building Your First Prefect Pipeline
4. Wrap-up

## 1. Understanding Data Pipelines

Watch this video for an overview of data pipeline basics: https://www.youtube.com/watch?v=DHO2PuR3jrs 

A *data pipeline* is a series of connected data processing steps where the output of one step becomes the input of the next. Think of it like a factory assembly line for data - raw materials (data) enter at one end, go through various transformation stations, and emerge as a finished product (insights, reports, or processed datasets).

```text
Raw Data → Load → Clean → Analyze → Visualize → Report
```

If you’ve ever written a script that loads data, cleans it, analyzes it, and saves results — you’ve already built a pipeline! You don’t need a special framework to have a pipeline.

Specialized software packages like *Prefect*, *Airflow*, or *Luigi* simply make it easier to manage, monitor, and automate complex workflows - especially when they run on large datasets, across multiple systems, or in the cloud.

The most important thing for data engineering and analysis is the pipeline *mindset*: organizing your work into clear, ordered steps. Start with whatever tooling matches your project (a simple script is often fine), and introduce orchestration tools only when you need scheduling, retries, observability, or cross-system coordination.

- Without pipelines : manual re-runs, messy scripts, hard to reproduce results.
- With pipelines : clear modular steps, easy to run end-to-end, scalable to cloud systems.

## Types of Data Pipelines

Not all pipelines work the same way - their structure depends on how and when data moves. Here are three common styles:

### Batch Processing Pipelines
Batch pipelines process data in groups at scheduled intervals, such as hourly, nightly, or weekly. They’re ideal for large datasets that don’t require real-time processing. For example, generating daily sales reports or refreshing dashboards overnight.

### Streaming Data Pipelines
Streaming pipelines handle continuous data flow in real time. They’re useful for monitoring live or fast-moving sources, such as social media feeds, IoT sensors, or stock tickers, where new data keeps arriving every second.

### Data Integration Pipelines
Integration pipelines focus on combining data from multiple systems or file formats into a single, unified format. For example, in biology labs you might have data from imaging systems, electrical sensors, and manual lags that all need to be integrated into a single dataset for analysiys. 

## ETL Pipeline

An **ETL pipeline** (Extract, Transform, Load) is a specific type of data pipeline designed to move and prepare structured data for analysis. These pipelines are the backbone of many analytics systems, ensuring that data flows cleanly from raw sources to organized storage.

### How ETL Pipelines Work
1. **Extract** – Data is pulled from one or more sources such as databases, web pages, APIs, or a file system.  
2. **Transform** – The extracted data is cleaned, reshaped, aggregated, or enriched in a staging area. This ensures the data is consistent and ready for analysis. For instance, you might remove duplicates, handle missing values, or standardize formats during this step. 
3. **Load** – The processed data is moved into its final destination, such as database server.  

### How ETL Pipelines Differ from General Data Pipelines

Every ETL pipeline is a data pipeline, but not every data pipeline follows the ETL structure. Some pipelines may move raw data directly from one location to another (Extract → Load) without any transformation, while others follow an ELT approach (Extract, Load, Transform), where transformations occur after loading. 

ETL pipelines are typically used when structured data needs to be cleaned and standardized before storage.

### Example

Imagine a company collecting sales data from multiple regional databases. An ETL pipeline would pull all sales records (Extract), clean and standardize currency formats (Transform), and store them in a central data warehouse (Load) for unified reporting.

## Pipeline Tools in Python

Python provides several powerful tools for building pipelines, such as Luigi, Airflow, Metaflow, Dagster, and Prefect. Each has its strengths and use-cases. 

In this lesson, we will focus on *Prefect*, a Python workflow orchestration tool with a gentle learning curve and allows us to build and run pipelines locally very quickly. We'll explain Prefect in more detail below. 


## 2. Prefect
Prefect is an open-source workflow orchestration tool that helps you define, run, and monitor your data pipelines. It’s designed to make workflows more *robust* (automatic retries, error handling) and more *observable* (logs, states, dashboards).

For a nice video introduction to Prefect, check out the following:
https://www.youtube.com/watch?v=Kt8GAZRpTcE


Prefect lets you break your work into tasks, connect them in flows, and then track everything in real-time through the Prefect UI. You should have Prefect installed in your virtual environment for this lesson. To check if Prefect is installed, run:

```bash
prefect version
```
You should see the installed version number, confirming everything is ready to go. We are assuming you have Prefect 2.x installed, which is the latest major version as of this writing.

### 2.1 Core Concepts in Prefect

At its core, Prefect is built around two simple yet powerful ideas: tasks and flows. Together, they form the foundation of every Prefect pipeline.

1. Tasks

A task is the smallest unit of work in a pipeline - for instance, loading data, cleaning a dataset, or sending a notification. You define a task in Python by putting the `@task()` decorator in front of a function. Prefect automatically tracks its state (such as running, successful, or failed), logs its output, and can retry the task if it encounters an error.

Ideally, each task should be simple and focused - since tasks are the atomic unit of work in Prefect flows.

2. Flows
   
A flow represents the higher-level workflow logic that connects multiple tasks together into a full pipeline. You create one using the `@flow()` decorator. A flow orchestrates how and when tasks run, and serves as the single entry point for running the entire pipeline.

For more details, you can also check out the [official Prefect docs on flows](https://docs.prefect.io/v3/concepts/flows).

### Simple Example

Let's see a super simple example to get the idea. In your IDE, create a file called `hello_prefect.py` and run from the command line:

```python
from prefect import task, flow

# Define tasks
@task
def say_hello(name):
    print(f"Hello, {name}!")

@task
def say_goodbye(name):
    print(f"Goodbye, {name}!")

# Define a flow that uses the tasks
@flow
def chatty_pipeline(name):
    say_hello(name)
    say_goodbye(name)

# To run the flow
if __name__ == "__main__":  
    chatty_pipeline("Code the Dream")
```
Each function decorated with `@task` becomes a distinct step in your workflow. The `@flow` function orchestrates those steps. Running `chatty_pipeline("Code the Dream")` executes the whole workflow. 

**What Happens When You Run It?**

You may notice a couple of things when you run the code. First, there is a delay. Second, you don’t just see:

```bash
Hello, Code The Dream!
Goodbye, Code The Dream!
```
Instead, you’ll also see a *whole bunch* of extra stuff in your terminal - something like this:

![Task&flow](resources/task&flow.png)

This is because Prefect isn't just an ordinary package. It's a *framework* that manages and tracks how your code is executed. When you decorate a function with `@flow`, Prefect *orchestrates* the workflow rather than just running it like plain Python. It spins up a small temporary local server in the background to track the execution of your flow (this explains the delay). 

That’s why you see lines like "Starting temporary server on http://127.0.0.1:8597" - this is Prefect’s way of managing the flow run. It also prints logs about task execution, which you can view in real time.

By handing control to Prefect through `@task` and `@flow`, you gain features like Centralized logging, automatic retries on failure, viewing run history in a UI, tracking task states. 

Later in this lesson, we’ll discuss the Orion UI to see how Prefect logs, retries, and statuses appear visually.

### 2.2 How Prefect enhances your workflows

Once you've built a basic pipeline using @task and @flow, Prefect gives you powerful tools that can be useful in some production environments. In this section, we’ll focus on three essential features: 

**1. Logging**

Instead of relying on `print()`, Prefect provides structured logging through `get_run_logger()`. This allows you to log messages with different levels (INFO, WARNING, ERROR) that are automatically timestamped and stored in the Prefect UI (Orion) for later review.

```python
from prefect import task, flow
from prefect.logging import get_run_logger

@task
def add_numbers(a, b):
    logger = get_run_logger()
    logger.info(f"Adding {a} + {b}")
    
    result = a + b
    
    logger.info(f"Result = {result}")
    return result

@flow
def math_flow(a, b):
    total = add_numbers(a, b)
    return total

if __name__ == "__main__":
    math_flow(3, 7)
```
Console Output:

![Logging](resources/logging.png)

This is how the logs look in your console output after using `get_run_logger()`.

The same logs also appear in Orion, where they’re stored for later runs. Unlike `print()`, Prefect’s logger (just like Python's logging system) supports log levels like **INFO**, **WARNING**, and **ERROR**.

- **INFO** – for general updates about what your flow is doing.  
- **WARNING** – for potential issues that don’t stop execution.  
- **ERROR** – for critical problems that cause a task or flow to fail.  

These levels help you filter and search logs efficiently, especially in production environments.

When running Prefect locally, you might see a message like:

> `Warning | EventsWorker - Still processing items...`

This simply means Prefect’s server is shutting down before all internal logs are fully written.  This is fine and won't affect your results. If you’d like to remove the warning, you can add this line at the end of your script, which will flush all logs before exiting:

```python
import time
time.sleep(0.2)
```

**2. Retries and Failure Handling**

Suppose a network call fails intermittently. Prefect allows you to retry it automatically without writing complex try/except logic. You can specify the number of retries and delay between attempts using a `retries` argument to the `@task` decorator:

```python
from prefect import flow, task
from prefect.runtime import task_run
from prefect.logging import get_run_logger
from random import random

@task(retries=4, retry_delay_seconds=2)  # 1 initial + 4 retries
def flaky_task():
    logger = get_run_logger()
    attempt = task_run.run_count  # 1-based counter
    logger.info(f"Attempt #{attempt}")
    
    # Always fail on first attempt
    if attempt == 1:
        logger.warning("Failed on first attempt.")
        raise ValueError("First attempt always fails.")
    
    # 75% chance of failure on subsequent attempts
    if random() < 0.75:
        logger.warning("Random failure occurred.")
        raise ValueError("Task failed! Retrying...")
    
    logger.info("Task succeeded!")

@flow
def retry_demo_flow():
    flaky_task()

if __name__ == "__main__":
    retry_demo_flow()
```

We built the above task to fail on the first attempt and then randomly fail 75% of the time on subsequent attempts. When you run this flow, you’ll see logs indicating each attempt, any warnings about failures, and a final success message when it eventually succeeds.

You may have noticed the `task_run` object, which gives you access to metadata about the current task execution, including how many times it has been attempted. This is a useful feature in general for tasks. For instance, if you are on the first run of a task, you may need to implement some setup logic. On retries, you might want to skip setup and go straight to the main logic. The `task_run.run_count` allows you to implement this kind of conditional behavior based on the attempt number.

This kind of built-in feature in Prefect makes your pipelines more robust and resilient to transient issues without needing to write complex error handling code yourself.

**3. Monitoring and Visualization with Prefect Dashboard**

Prefect provides a web-based UI - the *Prefect Server UI* (which used to be called *Orion*) - that lets you visualize and monitor your workflows in real time. In the web-based dashboard, you can see your pipeline’s execution flow, review detailed task logs and states (including retries), track scheduled runs, and even debug failed steps interactively, all from one convenient dashboard.

*To launch the Dashboard:*

**Step 1:** Launch
```bash
prefect server start
```
This command launches the Dashboard server, which you can access via your web browser at:

```
http://localhost:4200
```

**Step 2: Run Your Flow**
In a separate terminal, run any Prefect flow (like the `chatty_pipeline` example from earlier). As your flow runs, you can watch the execution in real time on the Dashboard. You’ll see when each task starts, finishes, and if any retries occur.

**Step 3: Explore in the Dashboard**

After running your flow, open the **Dashboard UI** in your browser. The **Dashboard** displays all your active and completed flow runs, giving you a quick overview of what’s happening. Clicking on a specific flow run opens detailed information about its individual tasks and logs.

Each log entry is structured with timestamps and log levels such as **INFO**, **WARNING**, and **ERROR**, and all logs are stored for future review.

![Orion](resources/orion.png)

As you can see, the Orion UI provides a clear overview of your flow runs, their statuses, and detailed logs - all in one place. This makes it much easier to monitor and debug your workflows compared to relying on terminal output alone.

### Advanced Features of Prefect

Prefect isn’t just about running tasks and viewing logs - it also comes with powerful features to handle real-world workflows. We won’t dive into these now, we are are just scratching the surface. But in a deeper dive, we would cover:

1. Caching: Skip re-running expensive tasks if their inputs haven’t changed. They can be saved for subsequent runs. This makes your workflows faster and more efficient.

2. Parallelism: Run tasks concurrently to speed up data processing and reduce bottlenecks.

3. Scheduling: Automate your flows to run on a set interval or a cron-like schedule, so they can operate hands-free in production.


### When to use Prefect versus building your own?

You might have wondered earlier: *“Couldn’t we just write Python functions and call them in order?”*  

The short answer is **yes** - you can build a simple pipeline by chaining plain Python functions. There is no simple answer to this question: it's a matter of trade-offs. For small, one-off scripts, plain Python functions are typically sufficient. They are quick to write and require no additional dependencies.  

But when things start to scale up, and you start to find yourself building in more tooling for monitoring, retries, scheduling, or want a dashboard to monitor your pipelines, then a framework might be a good option. You don't have to reinvent the wheel!

## 3. Building a Prefect Pipeline

Now that we’ve explored Prefect’s core ideas and enhancements, let’s bring everything together and build a tiny, end-to-end data analysis pipeline from scratch.

Imagine we’re analyzing exam scores from two student groups - say, Class A and Class B - to see if there’s a meaningful difference in their average performance.
We’ll go through the typical steps of a small but realistic data workflow: loading, cleaning, analyzing, visualizing, and reporting results. Along the way, you’ll see how Prefect makes each part more structured, observable, and reusable.

Here’s what we’ll do:

- Load exam scores for classes A and B.
- Clean and prepare the data (with retries for reliability)
- Perform a statistical test (t-test)
- Visualize the results
- Wrap everything in Prefect tasks and flows with logging

By the end, you’ll have a clear picture of how a simple Python script can evolve into a well-orchestrated Prefect pipeline.

### Step 1: Imports & Data Loader `(load_data)` with logging

We import necessary libraries:
- `pandas` → handle data as tables (`DataFrame`).  
- `prefect` → build workflows with tasks/flows.  
- `matplotlib.pyplot` → visualize data.  
- `scipy.stats` → run statistical tests. 

We also use Prefect’s logger so messages appear in Orion with timestamps and log levels.

```python
from prefect import task, flow
from prefect.logging import get_run_logger 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

@task
def load_data() -> pd.DataFrame:
    logger = get_run_logger()
    """
    Create a tiny, readable dataset:
      - Class: which class the student is in ('A' or 'B')
      - Score: exam score (numeric)
    """
    data = {
        "Class": ["A","A","A","A","A", "B","B","B","B","B"],
        "Score": [65, 70, 68, 72, 66,    78, 82, 80, 79, 81]
    }
    df = pd.DataFrame(data)
    logger.info("Exam scores loaded successfully")
    return df

```
Utility:

Instead of using plain print statements, Prefect’s logs are automatically searchable, timestamped, and stored in the UI, making them easy to review. If your pipeline runs daily, you can quickly filter logs for today’s runs without sifting through raw console output.

Once data is loaded, we need to clean it for reliability.

### Step 2: Clean the Data `(clean_data)` with Retries

Sometimes data is messy or missing. Prefect’s retries make the pipeline fault-tolerant.

```python
@task(retries=3, retry_delay_seconds=3)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    """
    Basic cleaning:
      1) Ensure Score is numeric.
      2) Drop missing values.
    (This is minimal on purpose for clarity.)
    """
    df = df.copy()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df.dropna(subset=["Score", "Class"])
    logger.info(" Data cleaned")
    return df

```
- df.copy() → makes a safe copy so we don’t overwrite the original.
- pd.to_numeric(..., errors="coerce") → converts values to numbers (invalid → NaN).
- df.dropna(...) → removes rows with missing values

 Once the data is clean, the next step is to explore and visualize it so we can understand class-wise performance.

### Step 3: Describe & Plot (describe_and_plot)
After cleaning, let’s summarize the dataset and visualize the score distribution.

```python
@task
def describe_and_plot(df: pd.DataFrame) -> None:
    logger = get_run_logger()
    """
    - Log summary stats per class.
    - Make a boxplot comparing Class X vs Class Y scores.
    - Save the plot.
    """
    # Generate descriptive statistics for each class
    summary = df.groupby("Class")["Score"].describe()
    logger.info("Summary statistics per class:\n%s", summary)

    # Plot
    ax = df.boxplot(by="Class", column="Score")
    plt.title("Exam Scores by Class")
    plt.suptitle("")  # removes automatic 'Score by Class' super-title
    plt.xlabel("Class")
    plt.ylabel("Score")

    # Save the plot
    plt.savefig("scores_boxplot.png")
    logger.info('Plot saved to "scores_boxplot.png"')
    plt.close()

```
When this task runs, it logs something like:

```
       count   mean   std   min   25%   50%   75%   max
Class
A        5.0  68.2   2.59  65.0  66.0  68.0  70.0  72.0
B        5.0  80.0   1.87  78.0  79.0  80.0  81.0  82.0
```
**Explanation**

Here, `.describe()` automatically gives us a mini summary for each group (Class A and Class B):

- **count** tells us how many scores are in the group.
- **mean** is the average score.
- **std** (standard deviation) shows how spread out the scores are.
- **min** and **max** are the lowest and highest scores.
- **25%, 50%, 75%** are the quartiles - with 50% being the median.

From the output, we can see that **Class B has higher average scores (80 vs 68.2)**, and both groups are fairly consistent since the standard deviation is small.

![Output_pipeline](resources/Output_pipeline.png)

With this overview, we now have both numerical summaries and visual patterns, making it easier to move into statistical testing in the next step.

### Step 4: Run a t-test (run_ttest)

Once we’ve explored the data, the next step is to formally test whether the difference between Class A and Class B is statistically significant. For this, we use a t-test.

```python
@task
def run_ttest(df: pd.DataFrame) -> tuple[float, float]:
    """
    Independent samples t-test:
      - Compares average Score between Class A and Class B.
      - Returns (t_statistic, p_value).
    """
    logger = get_run_logger()

    a = df[df["Class"] == "A"]["Score"]
    b = df[df["Class"] == "B"]["Score"]

    t_stat, p_val = ttest_ind(a, b)

    logger.info("T-test result: t=%.2f, p=%.4f", t_stat, p_val)
    return t_stat, p_val
```

With the t-test done, we now have statistical evidence. But numbers alone aren’t enough -  let’s translate them into clear conclusions in the next step.

### Step 5: Report Results (report_results)

Finally, we summarize the statistical test by showing the group means and stating whether the difference is significant.

```python
@task
def report_results(ttest_result: tuple[float, float], df: pd.DataFrame) -> None:
    """
    Report the results of the t-test with group means.
    """
    logger = get_run_logger()

    t_stat, p_val = ttest_result
    mean_a = df[df["Class"] == "A"]["Score"].mean()
    mean_b = df[df["Class"] == "B"]["Score"].mean()

    logger.info("Class A mean: %.1f, Class B mean: %.1f", mean_a, mean_b)

    if p_val < 0.05:
        logger.info("Conclusion: The difference is statistically significant (p < 0.05).")
    else:
        logger.info("Conclusion: No statistically significant difference (p ≥ 0.05).")
```

**Sample Output:**

```bash
Conclusion: The difference is statistically significant (p < 0.05)
```

With this final reporting step, your pipeline now summarizes group means and explains whether the difference is statistically meaningful.

### Step 6: Orchestrate Everything (@flow)

We bring it all together using the `@flow` decorator:

```python
@flow
def analysis_pipeline():
    """
    The *recipe* that runs all steps in order.
    """
    df = load_data()
    clean_df = clean_data(df)
    describe_and_plot(clean_df)
    result = run_ttest(clean_df)
    report_results(result, clean_df)

if __name__ == "__main__":
    analysis_pipeline()
```
Now you have a full mini pipeline - from data loading to cleaning, visualization, statistical testing, and reporting - all orchestrated with Prefect in a single flow.

This gives you a clear, reproducible, and automated workflow that’s easy to run, track, and extend as your project grows.

If you want to see it in the dashboard, you can run the flow and then open the Prefect UI to watch the tasks execute in real time, view logs, and see the final results.

You should incorporate this code into a single .py file and run it to see the full pipeline in action. 

## 4. Wrap-up

In this lesson, you’ve learned how pipelines automate complex workflows, making your data analysis **reproducible**, **modular**, and **scalable**. Prefect enhances this process by providing decorators for tasks and flows, built-in centralized **logging**, **retries**, caching, and seamless **monitoring** through the *Orion* dashboard. By building your pipeline from clear, modular steps, you can run the entire analysis with one command, handle failures automatically, and monitor your workflow in real time, all while keeping your code maintainable.

For your upcoming assignment, you’ll build your own data pipeline following this structure. Define each analysis step as a Prefect task, orchestrate them in a flow, and consider how you can use retries and logging to make your pipeline robust and observable. 

**👏 Well done!**
You just walked through your very first Prefect pipeline. Keep this momentum for your assignment. Congratulations on finishing your final lesson for Python 200. You are ready to do some more hands-on work. 