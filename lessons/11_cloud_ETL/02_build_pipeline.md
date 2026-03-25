# Building the Pipeline

*Fills in the skeleton with the real Azure tasks from Weeks 9 and 10. The completed pipeline is the capstone prototype.*

**Sections:**

  * Implementing the extract task: REST API call, return structured data
  * Implementing the transform task: Azure OpenAI enrichment from Lesson 10-3
  * Implementing the load task: Blob Storage upload from Lesson 9-3
  * Wiring them together in a @flow
  * Running the full pipeline: etl_pipeline(url=..., blob_path=...)
  * Reading the Prefect UI: what a successful run looks like, what a failed run shows

**Representative code sketch:**

```python
@task(retries=2, retry_delay_seconds=10)
def extract(url: str) -> list:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@task
def transform(records: list, client: AzureOpenAI) -> list:
    results = []
    for record in records:
        response = client.chat.completions.create(...)
        results.append({**record, "label": response.choices[0].message.content})
    return results

@task
def load(records: list, container: ContainerClient, path: str) -> None:
    container.upload_blob(path, json.dumps(records).encode(), overwrite=True)

@flow(log_prints=True)
def etl_pipeline(url: str, blob_path: str):
    raw = extract(url)
    client = AzureOpenAI(...)
    enriched = transform(raw, client)
    container = ContainerClient(...)
    load(enriched, container, blob_path)
```

## External Resources
TBD
