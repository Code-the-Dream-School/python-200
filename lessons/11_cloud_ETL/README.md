# Week 11: Cloud ETL

You have built all the pieces. In Week 1 you wrote a Prefect pipeline that ran locally. In Week 9 you learned to extract data from a REST API and load it into Azure Blob Storage. In Week 10 you added a Transform step, using an LLM to classify each record and write enriched results back to the cloud. The only thing missing is a single orchestrated pipeline that runs all three steps in sequence, reports what happened, and handles failures gracefully.

That is what this week is about. The three lessons take you from a brief Prefect refresher, through filling in the complete pipeline with real cloud tasks, to the production patterns that separate a working prototype from something reliable enough to run on a schedule. By the end, you will have a cloud ETL pipeline you built from scratch -- extract, transform, load, orchestrated, and observable. 🏁

## Topics

1. [Prefect, Revisited](01_prefect_revisited.md)
A quick bridge from the Week 1 Prefect introduction to the cloud context. Reviews the core concepts (@task, @flow, retries, logging, the dashboard) without re-explaining them from scratch, then shows what changes when your tasks do real cloud I/O instead of local file operations.

2. [Building the Pipeline](02_build_pipeline.md)
Fills in the ETL skeleton with the real tasks from Weeks 9 and 10 -- extract from Open-Meteo, transform with OpenAI, load to Blob Storage -- wired together in a single @flow. Covers how to run it, and how to read a successful (and a failed) run in the Prefect UI.

3. [Making Pipelines Production-Ready](03_production_ready.md)
Introduces four patterns that distinguish a working pipeline from a reliable one: retries for transient failures, raise_for_status() for clean error propagation, structured logging, and idempotent blob writes. Closes with a brief look at what comes next -- scheduling, Prefect Cloud, and parameterized deployments.

## Week 11 Assignment

Once you finish the lessons, head on over to the [assignments](../../assignments/README.md) for the capstone project.
