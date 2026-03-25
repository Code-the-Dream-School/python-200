# Making Pipelines Production-Ready

*Introduces the patterns that distinguish a working pipeline from a reliable one. Keeps scope tight -- one concept per pattern.*

Sections:

* Retries: Prefect's retries and retry_delay_seconds parameters -- when API calls fail
raise_for_status() and explicit error handling in tasks
* Logging with log_prints=True and Prefect's built-in task logging
* Idempotency: why overwriting blobs with overwrite=True matters, blob naming strategies
* Brief note on what comes next (not required for capstone): scheduling, Prefect Cloud, parameterization

No new concepts after this -- the lesson should feel like a finish line, not a cliff.

**External resources**: Prefect retries docs; Prefect logging guide
