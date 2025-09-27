# Pipelines with Prefect

Imagine running your analysis once, then being asked to run it again tomorrow, next week, or on a larger dataset. Manually re-running each step - loading, cleaning, analyzing, and reporting - would be painful, error-prone, and time-consuming.

**Pipelines solve this**. They give us reproducible, modular workflows where each step is defined and orchestrated. This approach becomes especially powerful in **cloud computing**, a topic we’ll explore in more detail in future lessons of Python 200.

### Learning objectives: 
By the end of this lesson, you’ll be able to:

- See why pipelines make analysis reproducible, modular, and scalable.
- Build and orchestrate workflows using Prefect tasks and flows.
- Make pipelines resilient and efficient with retries, caching, logging and scheduling.
- Monitor and debug workflows in real time using the Orion dashboard.
- Run a full data analysis workflow - load, clean, explore, analyze, and report - with a single command.

### Table of Contents
1. Understanding Data Pipelines
2. Prefect
3. Building Your First Prefect Pipeline
4. Wrap-up

## 1. Understanding Data Pipelines

What is a **Data Pipeline**?

A **data pipeline** is a series of connected data processing steps where the output of one step becomes the input of the next. Think of it like a factory assembly line for data - raw materials (data) enter at one end, go through various transformation stations, and emerge as a finished product (insights, reports, or processed datasets).

```python
Raw Data → Load → Clean → Analyze → Visualize → Report
```
- Without pipelines → manual re-runs, messy scripts, hard to reproduce results.
- With pipelines → clear modular steps, easy to run end-to-end, scalable to cloud systems.

## 2. Prefect

Now that we understand why pipelines are essential, let's look at a tool that helps us build them: **Prefect**.  

Prefect is an open-source workflow orchestration tool that helps you define, run, and monitor your data pipelines. It's designed to make data workflows more **robust** and **observable**.

### 2.1 Why Prefect?

You might wonder: **“Why can’t we just write Python functions for each step and run them in order?”**

Good question! Prefect lets you keep writing plain Python while adding production-grade workflow behavior that’s painful to build yourself.

Here’s what Prefect gives you out of the box:

- Retries for flaky steps → if one step fails (e.g., network timeout), Prefect can automatically retry.

- Caching → skip recomputation of expensive tasks if inputs haven’t changed.

- Visual UI (Orion) → monitor your workflows with run history, task states, and logs.

- Parallel execution → run independent tasks side-by-side with map().

- Logging → keeps all logs in one place, making debugging and tracking easier.

- Scheduling → run your pipelines daily/weekly.


In other words, Prefect handles the complexities of production workflows so you can focus on your analysis logic.

---

### 2.2 Why use Prefect instead of just functions?

Here’s what Prefect adds on top of plain Python:

| **Feature**               | **Plain Python**        | **Prefect**                          |
|---------------------------|-------------------------|--------------------------------------|
| Retry failed steps        | Manual try/except       | Declarative retries (`retries=3`)    |
| Cache expensive results   | Manual caching          | Built-in caching                     |
| Parallel execution        | Manual threading        | Easy parallel mapping                |
| Monitoring & debugging    | Print statements & logs | UI with history & logs *(Orion)*     |
| Resilience to failures    | Manual error handling   | Automatic retries & recovery         |
| Scheduling workflows      | Manual script runs      | Built-in scheduling & orchestration  |


### 2.3 Core Concepts in Prefect

At its core, Prefect uses two main concepts to define your pipeline:

1. **`@task()` decorator**  
    - Transforms a Python function into a Prefect **task**.
    - Prefect tracks its state (running, successful, failed), logs its output, and can retry on failure.

2. **`@flow()` decorator**  
   - Transforms a Python function into a Prefect **flow**.  
   - A flow is a collection of tasks and defines the overall workflow logic. It orchestrates the execution order of your tasks and handles dependencies.  
   - Provides a single entry point to run your entire pipeline. 

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
def output():
    say_hello("Code The Dream")
    say_hello("Students")

# To run the flow
# This condition checks if the current script is the one being executed directly.
if __name__ == "__main__":  
    output()
```
- Each function decorated with `@task` becomes a distinct step in your workflow.
- The `@flow` function orchestrates those steps.
- Running `output()` executes the whole workflow. 

### 2.4 How Prefect enhances your workflows

**1. Retries and Failure Handling**

Suppose a network call fails intermittently. Prefect allows you to declare retries.

Here's a very simple example showing retries in action with a flaky function using Prefect:

```python
from prefect import flow, task
from random import random
import time

# Define a flaky task
@task(retries=3, retry_delay_seconds=2)
def flaky_task():
    print("Trying to run task...")
    if random() < 0.7:  # 70% chance to fail
        raise ValueError("Task failed! Retrying...")
    print("Task succeeded!")

# Define a flow
@flow
def retry_demo_flow():
    flaky_task()

if __name__ == "__main__":
    retry_demo_flow()
```

- The task flaky_task has a 70% chance to fail each run. Prefect automatically retries it up to 3 times, waiting 2 seconds between attempts. If it eventually succeeds within 3 retries, the flow continues normally. If it still fails after 3 retries, the flow marks the task as failed.

**Sample Output (your console or Orion logs):**

```bash
Trying to run task...
Task failed! Retrying...
Trying to run task...
Task failed! Retrying...
Trying to run task...
Task succeeded!

```

This makes your pipeline robust without manual `try/except` blocks.

**2. Caching for Expensive Operations**

Tasks can cache their results so repeated runs don’t redo expensive work. If a step takes hours, Prefect can cache its result:

**Simple Example**

Suppose you have a task that downloads data from the internet:

```python
from prefect import task, flow
from prefect.tasks import task_input_hash
from datetime import timedelta

# Task to download data
@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def download_data(url):
    print("Downloading data from", url)
    # Imagine this takes time
    return f"Data from {url}"

# Flow that uses the task
@flow
def data_flow():
    result1 = download_data("https://example.com/data")
    result2 = download_data("https://example.com/data")
    print(result1, result2)
if __name__ == "__main__":
    data_flow()

```

- In the second call, input hasn't changed, Prefect skips re-execution and reuses the cached result, saving time and resources. 
- Cache is task-specific and you can control how long the cache is valid with `cache_expiration`.

**3. Parallelism with map()**

When you have many independent tasks, Prefect makes parallel execution simple:

```python
from prefect import task, flow

@task
def process_item(item: int) -> int:
    return item * 2

@flow
def parallel_flow():
    items = [1, 2, 3, 4]
    results = process_item.map(items)  # runs concurrently
    return results

```
This runs all *process_item* tasks concurrently, dramatically speeding up processing.

**4. Monitoring and Visualization with Orion UI**

Prefect provides a web-based UI (**the Prefect Server UI**, often called ***Orion***), where you can:

- Visualize your pipeline's execution flow
- See task logs, states, retries
- Monitor scheduled runs
- Debug failed steps interactively

*To launch Orion:*

**Step 1**: Install Prefect
```bash
pip install prefect
```

**Step 2:** Start Orion
```bash
prefect server start
```
This command launches the Orion server, which you can access via your web browser at:

```
http://localhost:4200
```

```
Note:
To launch the Prefect UI (version-dependent).
- For many Prefect 2.x installs run: `prefect orion start`.
- For some later releases the equivalent command is: `prefect server start`.
- If unsure, run `prefect --help` (or check `pip show prefect` / your installed version) and follow the CLI shown by your version.

```
**Step 3:** Run your pipeline script

Ensure your pipeline script uses the `@flow decorator` as shown earlier. When you execute the script, Prefect will register the run with Orion.

**Step 4:** View the dashboard

- Open the Orion UI in your browser.
- You’ll see your pipeline run listed.

This is invaluable in production environments.

**5. Logging**

- Prefect automatically captures and centralizes logs.
- Instead of using only print, you can use `get_run_logger()` for structured logging.

```python
from prefect.logging import get_run_logger

@task
def log_example():
    logger = get_run_logger()
    logger.info("This is tracked in Orion UI!")
```

Logs are captured and shown in Orion, aiding debugging at scale.

**6. ⏰ Scheduling Flows** 

Until now, we’ve run flows manually by calling them. But what if you want your analysis to run every morning at 9 AM, or every 5 minutes? Prefect makes this easy with ***scheduling***.

**Example: Interval Scheduling**

Here’s the simplest way to schedule a flow to run every 5 minutes:

```python
from prefect import flow

# Step 1: Define your flow
@flow
def my_flow():
    print("Hello from Prefect!")

# Step 2: Serve the flow with a schedule
if __name__ == "__main__":
    my_flow.serve(
        name="my-scheduled-flow",  # Deployment name (identify this flow)
        interval=300                # Run every 300 seconds (5 minutes)
    )

```
**How it works:**

 - *.serve()* turns the flow into a “running service.” 
 - Assign a name like "my-scheduled-flow" to identify this deployment in logs or the Orion dashboard.

👉 When you schedule a flow with `.serve()`, Prefect automatically creates something called a deployment. It uses your flow’s name and the deployment name you gave inside `.serve()` to identify it. For example: `my-flow/my-scheduled-flow`. 

And, Prefect keeps the flow actively running and polling for scheduled executions.

#### Quick Note About Interval & Cron:

- **Interval**: run the flow every fixed amount of time (e.g., every 5 minutes, every 2 hours).

```python
interval=300  # 300 seconds = 5 minutes
```

- **Cron**: run the flow at specific times/days using cron syntax (e.g., "0 9 * * *" → every day at 9 AM).

👉 Use `interval` for simple repetitive runs, `cron` for precise timing schedules.

---

## 3. Building Your First Prefect Pipeline
Let's build a data analysis pipeline using Prefect step by step.

### Step 0: Setup
```python
# First, install Prefect:
pip install prefect pandas matplotlib scipy
```

### Step 1: Imports & Data Loader `(load_data)` with logging

We import necessary libraries: `pandas` for data manipulation, `prefect` for our pipeline magic, `matplotlib.pyplot` for plotting, `scipy.stats` for the t-test.

We’re adding logging to the data loading step using **Prefect’s logger**, so the logs show up in the Orion UI with timestamps and levels. 

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
        "Class": ["A","A","A","A","A",  "B","B","B","B","B"],
        "Score": [65, 70, 68, 72, 66,    78, 82, 80, 79, 81]
    }
    df = pd.DataFrame(data)
    logger.info("Exam scores loaded successfully")
    return df

```
🔎 Utility:

- Instead of plain prints, logs are searchable, timestamped, and stored in Prefect’s UI.
- If this pipeline ran every day, you could filter logs for today’s runs without digging through raw output.

#### 📌Function Reference:

- pd.DataFrame(data) → creates a table (DataFrame) from a dictionary.
- **load_data** returns a DataFrame, which later tasks will receive as input.

Note: This is just a shortcut to build repeated labels:
```python
["A"]*5 + ["B"]*5
# → ["A","A","A","A","A","B","B","B","B","B"]
```

### Step 2: Clean the Data `(clean_data)` with Retries
 
Sometimes data loading/cleaning fails (e.g., file not found, temporary glitch).
By adding `retries=3`, Prefect will automatically retry the task if it errors.

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
🔎 Utility:

- If the first attempt fails, Prefect waits 3 seconds and tries again (up to 3 retries).
- No manual restart needed → pipelines are more resilient.

#### 📌Function Reference:
- df.copy() → makes a safe copy so we don’t overwrite the original.
- pd.to_numeric(..., errors="coerce") → converts values to numbers (invalid → NaN).
- df.dropna(...) → removes rows with missing values

### Step 3: Describe & Plot (describe_and_plot)

```python
@task
def describe_and_plot(df: pd.DataFrame) -> None:
    """
    - Print summary stats per class.
    - Make a boxplot comparing Class A vs Class B scores.
    - Save the plot. 
    """
    # Generate descriptive statistics for each class
    summary = df.groupby("Class")["Score"].describe()
    print(summary)

    # Plot
    ax = df.boxplot(by="Class", column="Score")
    plt.title("Exam Scores by Class")
    plt.suptitle("")  # removes automatic 'Score by Class' super-title
    plt.xlabel("Class")
    plt.ylabel("Score")

    # Save the plot
    plt.savefig("scores_boxplot.png")
    print('Plot saved to "scores_boxplot.png"')
    plt.close()
```

When this task runs, it prints something like:

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
- From the output, we can see that **Class B has higher average scores (80 vs 68.2)**, and both groups are fairly consistent since the standard deviation is small.

![Output_pipeline](resources/Output_pipeline.png)

#### 📌Function Reference: 

- df.groupby("Class") → splits data into Class A and Class B. 
- .describe() → gives stats: count, mean, std, min, max, quartiles. 
- df.boxplot(by="Class", column="Score") → makes side-by-side boxplots.
- plt.savefig("file.png") → saves plot as an image.

### Step 4: Run a t-test (run_ttest)

```python
@task
def run_ttest(df: pd.DataFrame) -> tuple[float, float]:
    """
    Independent samples t-test:
      - Compares average Score between Class A and Class B.
      - Returns (t_statistic, p_value).
    """
    a = df[df["Class"] == "A"]["Score"]
    b = df[df["Class"] == "B"]["Score"]
    
    #Welch’s t-test is more robust when groups have unequal variance.
    t_stat, p_val = ttest_ind(a, b, equal_var=False)

    print(f"T-test result: t={t_stat:.2f}, p={p_val:.4f}")
    return t_stat, p_val
```
#### 📌 Function Reference:

- df[df["Class"] == "A"]["Score"] → filter rows where Class = A, get scores.
- ttest_ind(a, b, equal_var=False) → compares mean of group A vs group B.
- t-statistic = size of the difference relative to variation.
- Null hypothesis (H0): both classes have the same average score.
- p-value: probability we’d see a difference this large if H0 were true.

**T-test output**
```python
T-test result: t = -8.07, p = 0.0002
```

**1. t = -8.07**

- This is the t-statistic. It measures how many standard errors the difference between the group means is away from zero.
- Negative sign just means the mean of the first group (Class A) is less than the mean of the second group (Class B).
- The magnitude (8.07) is large, which indicates a strong difference between the groups.

**2. p = 0.0002**

- This is the p-value, which tells you the probability of observing such a difference if the groups were actually the same (null hypothesis).
- A very small p-value (< 0.05) means the difference is statistically significant.


### Step 5: Report Results (report_results)

```python
@task
def report_results(ttest_result, df):
    t_stat, p_val = ttest_result
    mean_a = df[df["Class"] == "A"]["Score"].mean()
    mean_b = df[df["Class"] == "B"]["Score"].mean()
    print(f"Class A mean: {mean_a:.1f}, Class B mean: {mean_b:.1f}")
    
    if p_val < 0.05:
        print("Conclusion: The difference is statistically significant (p < 0.05).")
    else:
        print("Conclusion: No statistically significant difference (p ≥ 0.05).")
```
Prints `Conclusion: The difference is statistically significant (p < 0.05)`

📌 Function Reference:
- .mean() → calculates average of values.
- if p_val < 0.05: → 5% threshold is a common cutoff for significance.

### Step 6: Orchestrate Everything (@flow)

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
- The **@flow** decorator turns `analysis_pipeline` into a complete workflow.
- When called, it executes all steps in order, passing data between tasks.

### Step 7: Monitor Your Pipeline

Now that your pipeline is running, you can use the ***Orion*** dashboard (which you learned how to launch earlier) to:

- View all pipeline runs and their task states.
- Check logs for debugging and tracking.
- Observe retries, caching, and execution order in real-time.
- You can also **schedule pipeline** runs to run automatically at specific intervals (e.g., daily, hourly) or trigger them based on events.

Orion gives you full visibility into your workflow, so you can track progress and ensure reproducibility without manually inspecting each step.

## 4. Wrap-up

In this lesson, you’ve learned how pipelines automate complex workflows, making your data analysis **reproducible**, **modular**, and **scalable**. Prefect enhances this process by providing decorators for tasks and flows, built-in **retries**, **caching**, **scheduling**, centralized **logging**, and seamless **monitoring** through the *Orion* dashboard. By building your pipeline from clear, modular steps, you can run the entire analysis with one command, handle failures automatically, schedule runs, and monitor your workflow in real time, all while keeping your code maintainable.

For your upcoming assignment, you’ll build your own data pipeline following this structure. Define each analysis step as a Prefect task, orchestrate them in a flow, and consider how you can use retries, caching, and logging to make your pipeline robust and observable. Think about how the Orion dashboard can help you monitor and debug your workflow efficiently as you run it on different datasets.

**👏 Well done!**
You just walked through your very first Prefect pipeline. 🎉Keep this momentum for your assignment.