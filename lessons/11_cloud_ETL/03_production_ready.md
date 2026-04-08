# Making Pipelines Production-Ready

The pipeline you built in the previous lesson works. If everything goes right, it extracts, transforms, and loads without issue. But "everything going right" is an optimistic assumption for a pipeline that depends on two external APIs and a cloud storage service. This lesson covers four patterns that make your pipeline reliable -- not just correct.

For reference:
- [Prefect retries documentation](https://docs.prefect.io/v3/concepts/tasks#retries)
- [Prefect logging documentation](https://docs.prefect.io/v3/develop/logging)
- [What is idempotency?](https://blog.postman.com/what-is-idempotency/)

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why transient failures are inevitable in cloud pipelines and how retries address them
- Use `raise_for_status()` to propagate errors cleanly through a Prefect flow
- Apply `log_prints=True` and `get_run_logger()` to produce useful logs
- Explain idempotency and how `overwrite=True` and date-based blob paths support it

## Retries

External API calls fail sometimes. Rate limit responses, brief network timeouts, momentary service unavailability -- none of these are bugs in your code, but they will crash your pipeline if you do not account for them.

Prefect's retry mechanism handles this cleanly. You already added it to the extract task in the previous lesson:

```python
@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float) -> dict:
    ...
```

When the task raises any exception, Prefect waits 10 seconds and tries again, up to 2 more times. If all attempts fail, the task (and the flow) is marked as Failed. You see this in the UI as a task that cycled through multiple attempts before failing -- each attempt is logged separately, so you can see what happened on each try.

A few practical guidelines: use retries on tasks that make external network calls (API requests, Blob Storage operations). Do not use them on tasks where a failure indicates a genuine bug in your code -- retrying a programming error just wastes time and obscures the real problem. Two or three retries with a 10-30 second delay is a reasonable default for most API-dependent tasks.

## Error Handling

`raise_for_status()` is a small habit with a large payoff. Compare these two patterns:

```python
# Silent failure -- the pipeline continues with bad data
response = requests.get(url)
if response.status_code != 200:
    print("Something went wrong")

# Loud failure -- Prefect catches the exception and marks the task Failed
response = requests.get(url)
response.raise_for_status()
```

In the first version, if the API returns a 404 or 500, the pipeline continues with an empty or malformed response, potentially writing corrupt data to Blob Storage and showing a false *Completed* status in the UI. In the second version, the exception propagates immediately: Prefect marks the task as Failed, logs the exception, and stops the flow. The error is obvious and the data is clean.

The general principle: in a pipeline, fail loudly and early. A visible failure is easier to diagnose and fix than a silent one that corrupts downstream data.

## Logging

You already have `log_prints=True` on your flow, which captures all `print()` output as Prefect log entries. For most pipeline work, this is enough -- progress messages like `"Classified 6/24 records"` appear in the Prefect UI alongside task states.

For more structured log entries -- with explicit severity levels -- use `get_run_logger()` inside a task:

```python
from prefect.logging import get_run_logger

@task
def load(records: list, blob_path: str) -> None:
    logger = get_run_logger()
    ...
    logger.info(f"Loaded {len(records)} records to {blob_path}")
```

The difference: `logger.info()` entries are tagged with a severity level (INFO, WARNING, ERROR) and are filterable in the Prefect UI. `print()` output is captured as plain INFO logs. For a pipeline that runs regularly, structured logging makes it easier to search for warnings and errors across many runs.

Use `logger.warning()` for anything worth flagging but not worth stopping the pipeline -- for example, recording `"unknown"` label counts after the transform step. Use `logger.error()` for conditions that should not happen and may indicate a real problem. Reserve `logger.info()` for routine progress.

## Idempotency

A pipeline is *idempotent* if running it twice produces the same result as running it once. This matters because pipelines fail and get re-run -- due to a retry, a manual re-run after a bug fix, or a scheduled re-run that overlapped with a previous run.

The load task already handles this with `overwrite=True`:

```python
container.upload_blob(blob_path, payload, overwrite=True)
```

Without `overwrite=True`, a second run would raise an error because the blob already exists. With it, the second run simply replaces the first output with an identical result. The pipeline is safe to re-run.

The blob path pattern -- `final/{today}/weather_etl.json` -- reinforces this. Each day's run writes to its own path, so re-running today's pipeline does not affect yesterday's data. This is a simple form of *partitioning by date*, a standard practice in data engineering.

## What Comes Next

The pipeline you have built runs on your laptop, triggered manually by `python etl_pipeline.py`. In a real production environment, you would want it to run automatically on a schedule, perhaps every morning before the work day starts.

Prefect supports this through *deployments* -- a configuration file that tells Prefect how to run your flow, with what parameters, and on what schedule. The `prefect deploy` command creates a deployment from your flow, and a Prefect *worker* executes it. [Prefect Cloud](https://www.prefect.io/cloud) provides a hosted version of this infrastructure with a managed UI, team features, and SLA monitoring.

These topics are beyond the scope of this course, but you will almost certainly encounter them if you work in data engineering. The pipeline you built here -- a `@flow` with three `@task` functions, retries, logging, and idempotent writes -- is the same pattern that production Prefect deployments use. The jump from local to scheduled is smaller than it looks.

---

You built a cloud ETL pipeline from scratch. Extract from a live API, transform with a language model, load to cloud storage, orchestrated with Prefect, observable in a UI, resilient to transient failures. That is Week 1 pipelines + Week 9 Blob Storage + Week 10 LLM transform, all working together. Well done.
