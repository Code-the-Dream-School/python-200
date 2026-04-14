# Week 11: Cloud ETL Capstone

## Overview

This is the capstone week. Students take everything from the course — Prefect orchestration (Week 1), Azure Blob Storage (Week 9), and LLM transforms (Week 10) — and wire it into a single, production-aware pipeline. The lessons revisit Prefect in a cloud context, walk through building the pipeline step by step, and cover four production patterns: retries, error handling, logging, and idempotency. The assignment is the full pipeline, written from scratch.

## Key Concepts

**Prefect in the cloud** — The core Prefect concepts (tasks, flows, retries, logging) from Week 1 apply here, but now the tasks do real I/O: hitting an external API, calling an LLM, writing to Blob Storage. This is where observability matters — when a cloud pipeline fails, the Prefect UI is how you find out what went wrong and where.

**Retries** — Cloud operations fail transiently (network timeouts, rate limits, brief service outages). The `retries` and `retry_delay_seconds` parameters on `@task` handle this automatically. The lesson's guidance: retry network/API calls, don't retry logic errors.

**`raise_for_status()`** — Calling this after an HTTP request means a failed request raises an exception immediately, which Prefect catches and records. Without it, bad responses (4xx, 5xx) can silently produce malformed data that propagates downstream.

**Structured logging** — `get_run_logger()` inside a Prefect task writes logs that appear in the Prefect UI, not just the terminal. This is what allows you to debug a pipeline that ran while you weren't watching.

**Idempotency** — A pipeline is idempotent if running it multiple times produces the same result. Date-partitioned blob paths (`final/2025-06-01/weather_etl.json`) plus `overwrite=True` achieve this: re-running the pipeline on the same day overwrites the previous result cleanly, with no duplicates or conflicts.

## Common Questions

- **"What's the difference between `@task` and `@flow`?"** — A `@flow` is the entry point that Prefect tracks as a single run. `@task`s are the individual units of work within it. Flows can call tasks; flows can also call other flows. Pure in-memory helper functions don't need to be tasks.
- **"Why does the Prefect UI show my task as Failed even though my script printed success?"** — The task likely raised an exception after the print. Check the logs in the UI for the full traceback.
- **"What does `overwrite=True` protect me from?"** — If the pipeline crashes mid-run and you re-run it, `overwrite=True` on the blob upload means the partial output from the first run gets replaced cleanly. Without it, you'd either get an error (if the blob already exists and overwrite isn't set) or stale data from the previous run.
- **"How would I run this pipeline on a schedule?"** — Prefect deployments (briefly mentioned at the end of the lesson) let you trigger flows on a cron schedule. Prefect Cloud (a hosted version) makes this easier. This is the natural next step beyond the course.

## Watch Out For

- **Three things must be running or ready** — Before testing the pipeline: (1) `az login` must be active, (2) the OpenAI API key must be in `.env`, (3) the Prefect server (`prefect server start`) must be running in a separate terminal. If any of these are missing, the pipeline will fail in a specific and diagnosable way — but students may not know which to check first.
- **Prefect server on a different port** — Prefect's default dashboard is at `http://localhost:4200`. If something else is using that port, the server may fail to start or open on a different port. This is rare but worth knowing.
- **"Write it yourself" is intentional** — The assignment explicitly says not to copy-paste from the lessons. This is the capstone — students should be synthesizing, not transcribing. If a student shows up with code that closely mirrors the lesson examples but doesn't understand it, gently ask them to walk through what each part does.
- **The reflection matters** — `outputs/pipeline_run.md` asks students to describe what failed, what the Prefect UI showed, and what they'd change for production. This is often the most informative part of the submission — read it.

## Suggested Activities

1. **Prefect UI walkthrough:** Ask a student to share their Prefect UI showing all three tasks in Completed state. Click into one task together and look at the logs. Ask: "If the transform task had failed instead, what would you look at first to diagnose why?"

2. **Production scenario drill:** Present a failure scenario: "Your pipeline ran last night, but the Prefect UI shows the transform task failed after 12 out of 24 records were processed. The extract task completed. What do you do?" Walk through: check the logs, identify the error, re-run (idempotency means it's safe), verify the output.

3. **Capstone reflection:** Go around the group and ask each student to share one thing from their `pipeline_run.md`: something that broke, something that surprised them, or something they'd add if this were a real production pipeline. This surfaces both debugging stories and genuine design thinking.
