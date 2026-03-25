# Prefect, Revisited

*Quick bridge from the Week 1 Prefect introduction to the cloud context. Should feel like reconnecting with something familiar, not learning something new.*

**Sections**:
  * Quick review: @flow, @task, running a flow locally (should take 5 minutes -- students saw this in Week 1)
  * What changes in a cloud ETL context: tasks do real I/O (API calls, Blob operations), failures matter, observability matters
  * The Prefect UI: prefect server start, viewing runs, task states (brief tour)
  * Structuring a pipeline: one task per logical step, flows compose tasks
  * A minimal end-to-end skeleton (no Azure yet -- just the structure)

**Representative code sketch**

```python
from prefect import flow, task

@task
def extract(url: str) -> list:
    ...  # requests.get(url)

@task
def transform(records: list) -> list:
    ...  # Azure OpenAI calls

@task
def load(records: list, path: str) -> None:
    ...  # Blob Storage upload

@flow
def etl_pipeline(url: str, blob_path: str):
    raw = extract(url)
    enriched = transform(raw)
    load(enriched, blob_path)
```

**External resources**: Prefect 3.x docs: flows and tasks; Week 1 Prefect lesson (internal link)
