# Making Pipelines Production-Ready

The pipeline you built in the previous lesson works. If everything goes right, it extracts, transforms, and loads without issue. But "everything going right" is an optimistic assumption for a pipeline that depends on two external APIs and a cloud storage service. This lesson covers four patterns that make your pipeline reliable -- not just correct.

These patterns are extremely common in real-world data engineering. Most production pipeline work is not about writing complicated algorithms -- it is about making systems dependable, observable, and safe to re-run.

For reference:

* [Prefect retries documentation](https://docs.prefect.io/v3/concepts/tasks#retries)
* [Prefect logging documentation](https://docs.prefect.io/v3/develop/logging)
* [What is idempotency?](https://blog.postman.com/what-is-idempotency/)

## Learning Objectives

By the end of this lesson, you will be able to:

* Explain why transient failures are inevitable in cloud pipelines and how retries address them
* Use `raise_for_status()` to propagate errors cleanly through a Prefect flow
* Apply `log_prints=True` and `get_run_logger()` to produce useful logs
* Explain idempotency and how `overwrite=True` and date-based blob paths support it

## Retries

External API calls fail sometimes. Rate limit responses, brief network timeouts, momentary service unavailability -- none of these are bugs in your code, but they will crash your pipeline if you do not account for them.

Prefect's retry mechanism handles this cleanly. You already added it to the extract task in the previous lesson:

```python id="1e4pqa"
@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float) -> dict:
    ...
```

When the task raises any exception, Prefect waits 10 seconds and tries again, up to 2 more times. If all attempts fail, the task (and the flow) is marked as Failed.

In the Prefect UI, you will see the task cycle through multiple attempts before ultimately succeeding or failing. Each attempt is logged separately, which makes debugging much easier.

Retries are especially useful for operations that depend on external systems:

* API calls
* Blob Storage uploads/downloads
* database connections
* cloud authentication requests

These systems occasionally fail for reasons outside your control, and a retry often fixes the problem automatically.

A useful rule of thumb:

* use retries for transient external failures
* do **not** use retries for genuine programming bugs

If your code has a syntax error or a broken loop, retrying will not help -- it only delays the inevitable failure and makes debugging noisier.

Two or three retries with a 10-30 second delay is a very common production default.

## Error Handling

`raise_for_status()` is a small habit with a large payoff. Compare these two patterns:

```python id="t1k65j"
# Silent failure -- the pipeline continues with bad data
response = requests.get(url)

if response.status_code != 200:
    print("Something went wrong")
```

```python id="p8ye6s"
# Loud failure -- Prefect catches the exception and marks the task Failed
response = requests.get(url)

response.raise_for_status()
```

In the first version, if the API returns a 404 or 500, the pipeline continues with an empty or malformed response. That can lead to corrupted downstream data, misleading results, or a pipeline that appears successful even though the data is wrong.

In the second version, the exception propagates immediately:

* Prefect marks the task as *Failed*
* the error appears clearly in the logs
* downstream tasks do not run
* the bad data never gets written

The general principle is simple:

> In data pipelines, visible failures are usually safer than silent corruption.

A failed run is inconvenient, but incorrect data silently flowing through a system is often much harder to detect and fix later.

## Logging

You already have `log_prints=True` on your flow, which captures all `print()` output as Prefect log entries. For many small pipelines, this is enough.

For example:

```python id="dr8j7f"
print("Classified 6/24 records")
```

will appear automatically in the Prefect UI logs.

For more structured logging -- with severity levels -- use `get_run_logger()` inside a task:

```python id="g0lfm5"
from prefect.logging import get_run_logger

@task
def load(records: list, blob_path: str) -> None:
    logger = get_run_logger()

    logger.info(
        f"Loaded {len(records)} records to {blob_path}"
    )
```

Structured logs are useful because they can be filtered and searched more easily across many runs.

Common logging levels:

* `logger.info()` → routine progress updates
* `logger.warning()` → something unusual but non-fatal
* `logger.error()` → a serious problem or failure

For example, `"unknown"` model classifications might deserve a warning log but probably should not crash the entire pipeline.

Good logging becomes more important as pipelines grow larger and run automatically on schedules. When something breaks overnight, logs are often the first place engineers look.

## Idempotency

A pipeline is *idempotent* if running it twice produces the same result as running it once.

This matters because pipelines frequently get re-run:

* after a retry
* after fixing a bug
* after a deployment failure
* after an infrastructure outage

The load task already handles this with `overwrite=True`:

```python id="5j3hjr"
container.upload_blob(
    blob_path,
    payload,
    overwrite=True
)
```

Without `overwrite=True`, a second run would fail because the blob already exists.

With it, the pipeline safely replaces the previous output with the updated result.

This makes the pipeline much safer to re-run during debugging or recovery.

The blob path pattern also helps:

```python id="5aqz0j"
final/{today}/weather_etl.json
```

Each day's data is stored in its own folder, so rerunning today's pipeline does not overwrite yesterday's results.

This pattern -- partitioning data by date -- is extremely common in production data systems because it keeps historical runs isolated and easier to manage.

## What Comes Next

The pipeline you built runs locally when you execute:

```bash id="2tnwth"
python etl_pipeline.py
```

In a real production environment, pipelines are usually scheduled automatically -- for example:

* every morning at 6am
* every hour
* every night after business hours

Prefect supports this through *deployments*, which package your flow configuration and scheduling rules. A Prefect *worker* then executes the flow automatically according to that schedule.

[Prefect Cloud](https://www.prefect.io/cloud) provides a hosted version of this infrastructure with additional monitoring, alerting, and collaboration features.

These topics are beyond the scope of this course, but the important thing is that the pipeline structure you built here is already very close to what real production systems use.

The jump from:

```bash id="6b6gk4"
python etl_pipeline.py
```

to a scheduled production deployment is smaller than it looks.

---

You built a cloud ETL pipeline from scratch:

* extract from a live API
* transform with a language model
* load to cloud storage
* orchestrate with Prefect
* monitor through a UI
* recover gracefully from transient failures

That is Week 1 pipelines + Week 9 Blob Storage + Week 10 LLM transforms, all working together in a realistic workflow.