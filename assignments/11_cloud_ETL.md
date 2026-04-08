# Week 11 Assignment: Cloud ETL Capstone

This is the capstone project for Python 200. You have spent ten weeks building toward this: a cloud ETL pipeline that extracts data from a live API, transforms it with a language model, loads the results to Azure Blob Storage, and runs the whole thing as an orchestrated, observable Prefect flow. This week you build it yourself.

The warmup checks your understanding of Prefect and production patterns. The project is the pipeline.

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_11/`. Inside that folder, create:

1. `warmup_11.py` -- warmup exercises (conceptual answers as comments, code as requested)
2. `etl_pipeline.py` -- the complete ETL pipeline
3. `outputs/pipeline_run.md` -- a short written reflection on your run

When finished, commit and open a PR as described in the [assignments README](README.md).

**Setup reminder:** You will need `az login` active for Blob Storage, an OpenAI API key in `.env`, and the Prefect server running in a separate terminal when you test your pipeline.

```bash
uv pip install prefect requests openai python-dotenv azure-storage-blob azure-identity pandas
```

# Part 1: Warmup

Put all warmup answers in `warmup_11.py`. Use comments to mark each section and question. For conceptual questions, write your answer as a comment block. For code questions, write working Python.

## Prefect Orchestration

### Prefect Question 1

In a comment block, answer: what is the difference between a `@task` and a `@flow` in Prefect? You have a helper function that converts a temperature from Celsius to Fahrenheit -- a pure, in-memory calculation with no I/O. Would you decorate it with `@task`? Why or why not?

### Prefect Question 2

Write the decorator (just the decorator line, not the full function) for a task named `call_api` that retries up to 3 times with a 30-second delay between attempts.

### Prefect Question 3

You run your pipeline and the Prefect UI shows: `extract` is *Completed*, `transform` is *Failed*, `load` never ran. In a comment block, describe: where in the UI do you look to understand what went wrong, and what specific information would you expect to find there?

## Production Patterns

### Production Question 1

In a comment block, explain what `raise_for_status()` does and why it is better than writing `if response.status_code != 200: print("error")` in a pipeline task. What happens to downstream tasks in each case when the API returns a 500 error?

### Production Question 2

Your pipeline uploads results to `final/{today}/weather_etl.json` with `overwrite=True`. The pipeline crashes halfway through the transform step. You fix the bug and re-run it from the beginning. In a comment block, explain: what does `overwrite=True` protect you from in this scenario, and what would happen without it?

### Production Question 3

Write a task stub -- just the function signature, decorator, and a single log line -- that uses `get_run_logger()` to log an INFO message saying how many records were loaded. The function should accept `records` (a list) and `blob_path` (a string) as arguments.

# Part 2: Project -- Full ETL Pipeline

Build `etl_pipeline.py`: a complete Prefect flow with three tasks that orchestrate the full Extract, Transform, Load pipeline. This is the capstone -- write it yourself using the lessons as a guide, not as code to copy.

## Requirements

### Extract task

- Decorated with `@task(retries=2, retry_delay_seconds=10)`
- Calls the Open-Meteo API for 7 days of hourly `temperature_2m` and `precipitation` data for a city of your choosing
- Uses `raise_for_status()`
- Returns the raw JSON response as a dict
- Prints a confirmation message

### Transform task

- Decorated with `@task`
- Reshapes the `"hourly"` parallel lists into individual per-hour records
- Classifies the first 24 records (one day) using the OpenAI API with this system prompt:

```text
You are classifying hourly weather conditions for outdoor running.
Given a temperature in Celsius and a precipitation amount in mm,
classify the conditions as exactly one of: good, marginal, or bad.
Reply with that one word only -- no punctuation, no explanation.
```

- Falls back to `"unknown"` if the model returns an unexpected response
- Prints a progress message every 6 records
- Returns the list of enriched records

### Load task

- Decorated with `@task`
- Uploads the enriched records as JSON to `final/<today>/weather_etl.json` in your `pipeline-data` container
- Uses `overwrite=True`
- Prints a confirmation with the blob path and byte count

### Flow

- Decorated with `@flow(log_prints=True)`
- Calls the three tasks in order
- Prints a completion message with the final blob path

## Running and Verifying

1. Start the Prefect server: `prefect server start`
2. Run your pipeline: `python etl_pipeline.py`
3. Open `http://localhost:4200` and verify all three tasks show *Completed*
4. Navigate to your storage account in the Azure Portal and confirm the blob exists at `pipeline-data/final/<today>/weather_etl.json`

## Reflection

Write `outputs/pipeline_run.md` with a short reflection (4-6 sentences) covering:

- Did the pipeline run cleanly on the first try? If not, what failed and how did you fix it?
- What did the Prefect UI show? Were there any retries?
- What is one thing you would change or add if you were deploying this pipeline to run on a daily schedule?

## Video

Record a short video (target: 3-4 minutes, max: 5). Show:

1. The pipeline running in your terminal to completion
2. The Prefect UI with all three tasks in *Completed* state, and the logs from at least one task
3. The final blob in the Azure Portal under `pipeline-data/final/<date>/`

Paste the video link in a comment at the top of `etl_pipeline.py`.

---

Congratulations! With this step, you've finished Python for Cloud & AI.
