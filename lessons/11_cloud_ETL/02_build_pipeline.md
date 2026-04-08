# Building the Pipeline

You have already written all three pieces of this pipeline. In Week 9 you extracted data from the Open-Meteo API and loaded it to Blob Storage. In Week 10 you read that data back, classified it with OpenAI, and wrote enriched results to a processed path. This lesson wires those pieces into a single Prefect flow.

For reference:
- [Prefect tasks documentation](https://docs.prefect.io/v3/concepts/tasks)
- [Week 9: Loading Pipeline Data to Blob Storage](../09_cloud_data/03_loading_pipeline.md)
- [Week 10: Transforming Blob Data with OpenAI](../10_llm_pipelines/02_blob_data.md)

## Learning Objectives

By the end of this lesson, you will be able to:

- Implement extract, transform, and load as Prefect tasks using real cloud operations
- Wire the three tasks into a @flow and run the complete pipeline
- Read a Prefect UI run to confirm success or identify a failure

## Setup

This pipeline requires all the packages from Weeks 9 and 10:

```bash
uv pip install prefect requests openai python-dotenv azure-storage-blob azure-identity pandas
```

Load your `.env` file and define your constants at the top of the script:

```python
import os
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_URL = "https://<your-account>.blob.core.windows.net"
CONTAINER = "pipeline-data"
MAX_RECORDS = 24  # process one day of hourly data

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)
VALID_LABELS = {"good", "marginal", "bad"}
```

## The Extract Task

The extract task calls the Open-Meteo API and returns the raw JSON response. Adding `retries=2` means Prefect will automatically retry the task up to twice if it raises an exception -- useful since API calls can fail transiently:

```python
import requests
from prefect import task

@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,precipitation"
        f"&forecast_days=7"
    )
    response = requests.get(url)
    response.raise_for_status()
    print(f"Extracted forecast data for ({latitude}, {longitude})")
    return response.json()
```

## The Transform Task

The transform task reads the raw API response, reshapes the parallel lists into individual records, and classifies each one with the OpenAI API. Initializing the OpenAI client inside the task keeps the task self-contained:

```python
from openai import OpenAI
from prefect import task

@task
def transform(data: dict, max_records: int) -> list:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    hourly = data["hourly"]
    records = []
    for i in range(min(max_records, len(hourly["time"]))):
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        })

    enriched = []
    for i, record in enumerate(records):
        user_msg = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
        raw_label = response.choices[0].message.content.strip().lower()
        label = raw_label if raw_label in VALID_LABELS else "unknown"
        enriched.append({**record, "conditions": label})
        if (i + 1) % 6 == 0:
            print(f"  Classified {i + 1}/{len(records)} records")

    print(f"Transform complete: {len(enriched)} records enriched")
    return enriched
```

In a production environment with Azure OpenAI, swap `OpenAI` for `AzureOpenAI` using the two-line change from [Week 10, Lesson 3](../10_llm_pipelines/03_Azure_OpenAI.md).

## The Load Task

The load task uploads the enriched records to Blob Storage. The `ContainerClient` is initialized inside the task for the same reason as the OpenAI client -- self-contained tasks are easier to test and reason about:

```python
import json
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from prefect import task

@task
def load(records: list, blob_path: str) -> None:
    credential = DefaultAzureCredential()
    container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

    payload = json.dumps(records).encode("utf-8")
    container.upload_blob(blob_path, payload, overwrite=True)
    print(f"Loaded {len(payload)} bytes to {blob_path}")
```

## Wiring the Flow

The flow calls the three tasks in order and constructs the blob path from today's date:

```python
from datetime import date
from prefect import flow

@flow(log_prints=True)
def etl_pipeline(latitude: float = 35.2271, longitude: float = -80.8431):
    today = date.today().isoformat()
    blob_path = f"final/{today}/weather_etl.json"

    data = extract(latitude, longitude)
    enriched = transform(data, max_records=MAX_RECORDS)
    load(enriched, blob_path)

    print(f"Pipeline complete. Results at {blob_path}")

if __name__ == "__main__":
    etl_pipeline()
```

`log_prints=True` on the flow means every `print()` call in the flow and its tasks is captured as a Prefect log entry. You will see these in the Prefect UI alongside the task state transitions.

## Running the Pipeline

Make sure the Prefect server is running in one terminal:

```bash
prefect server start
```

Then run the pipeline in a second terminal:

```bash
python etl_pipeline.py
```

You should see Prefect's output in the terminal as each task starts and completes. The full run typically takes 1-3 minutes, depending on OpenAI response times.

## Reading the Prefect UI

Open `http://localhost:4200` and click into the `etl-pipeline` run. A successful run shows all three tasks in *Completed* state. Click any task to see its logs -- you should see the print statements from each task captured there.

If a task fails, it shows in *Failed* state (red). Click the failed task and open the *Logs* tab to find the exception traceback. The other tasks that completed successfully remain green, so you can see exactly where in the pipeline the failure occurred without re-running the whole thing.

Try deliberately breaking the pipeline -- pass an invalid latitude, or temporarily remove your `.env` API key -- and observe what the UI shows. Reading failed runs is a skill, and the best way to develop it is to cause a few failures intentionally.
