# Loading Pipeline Data to Blob Storage

In Week 1, you used Prefect to build a data pipeline that ran entirely on your local machine -- it read files from disk, processed them, and wrote results back to disk. That pipeline did real work, but when the script finished, the output sat in a local folder on your laptop. If you deleted the folder, ran the script on a different machine, or wanted another script to pick up where it left off, you had a problem.

In production, pipelines need durable, accessible storage -- data that persists after the job finishes, is readable by the next step in the pipeline, and is not tied to any single machine. That is exactly what Blob Storage provides. This lesson builds the Extract + Load pattern you will use as the foundation of the capstone pipeline in Week 11.

**[This video is a concise explanation of Extract, Load, Transform (ETL) pipelines.](https://youtu.be/OW5OgsLpDCQ?si=HgAnExKKWgEk_um9)**

For reference:
- [Requests library documentation](https://requests.readthedocs.io/en/latest/)
- [Open-Meteo API documentation](https://open-meteo.com/en/docs)

## Learning Objectives

By the end of this lesson, you will be able to:

- Design a structured folder layout for pipeline data in Blob Storage
- Extract data from a public REST API and serialize it as JSON
- Upload serialized data to a structured path in Blob Storage
- Download a blob and read it into a pandas DataFrame

## Designing a Folder Structure

Blob names are just strings, but you can use forward slashes in them to organize data logically. A common convention for pipeline data is:

```text
<layer>/<date>/<filename>
```

For example:
- `raw/2024-01-15/weather.json` -- the original API response, unchanged
- `processed/2024-01-15/weather_classified.json` -- transformed by a later pipeline step
- `final/2024-01-15/weather_etl.json` -- the finished pipeline output

This keeps pipeline layers clearly separated. It also makes it straightforward to reprocess data for a specific date (look up the `raw/` path) and to verify that each stage produced output. You will use this exact structure in Weeks 10 and 11.

## Extracting from a REST API

You worked with the `requests` library in Week 1. This is a quick review -- the only new part is what you do with the data afterward.

We will use the [Open-Meteo API](https://open-meteo.com/), a free weather API that requires no API key. The URL below requests 7 days of hourly temperature and precipitation data for Charlotte, NC:

```python
import requests

url = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=35.2271&longitude=-80.8431"
    "&hourly=temperature_2m,precipitation"
    "&forecast_days=7"
)

response = requests.get(url)
response.raise_for_status()
data = response.json()
```

`raise_for_status()` is a one-line error check: if the server returned an error status code (4xx or 5xx), it raises an exception immediately rather than silently continuing with bad data. This is a habit worth building in all pipeline code. You can swap in any latitude and longitude you like -- Open-Meteo's website has a map tool for looking up coordinates.

## Serializing for Storage

Blob Storage stores bytes. Before uploading, you need to convert your data to a byte string. The two most common formats in pipelines are JSON and CSV:

- *JSON* is the natural choice for nested or irregular data -- API responses, records with optional fields, anything with a non-flat structure.
- *CSV* fits flat, tabular data that you plan to load into pandas or a database.

The Open-Meteo response is nested (it returns parallel lists for each field under an `"hourly"` key), so JSON is the right choice here:

```python
import json

payload = json.dumps(data).encode("utf-8")
```

`json.dumps()` converts the Python dictionary to a JSON string. `.encode("utf-8")` converts that string to bytes.

## Uploading to Blob Storage

With the data serialized, upload it to a structured path. Using today's date as part of the path ensures that each daily run produces a new file rather than overwriting a previous one:

```python
from datetime import date
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

today = date.today().isoformat()  # e.g., '2024-01-15'
blob_path = f"raw/{today}/weather.json"

credential = DefaultAzureCredential()
container = ContainerClient(
    account_url="https://<account>.blob.core.windows.net",
    container_name="pipeline-data",
    credential=credential
)

container.upload_blob(blob_path, payload, overwrite=True)
print(f"Uploaded to {blob_path}")
```

## Reading Back into Pandas

Uploading to the cloud is only half the picture. Let's confirm the data landed correctly by reading it back:

```python
import io
import pandas as pd

raw = container.download_blob(blob_path).readall()
data_back = json.loads(raw.decode("utf-8"))

df = pd.DataFrame(data_back["hourly"])
print(df.head())
```

The Open-Meteo `"hourly"` field contains a dictionary of parallel lists -- one list per field (`"time"`, `"temperature_2m"`, `"precipitation"`). Passing that dictionary directly to `pd.DataFrame()` creates a column for each field.

## Putting It Together

Here is the complete Extract + Load pipeline as a single script:

```python
import requests
import json
from datetime import date
import pandas as pd
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

ACCOUNT_URL = "https://<account>.blob.core.windows.net"
CONTAINER = "pipeline-data"
LATITUDE = 35.2271
LONGITUDE = -80.8431

# Extract
url = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&hourly=temperature_2m,precipitation"
    f"&forecast_days=7"
)
response = requests.get(url)
response.raise_for_status()
data = response.json()

# Serialize
payload = json.dumps(data).encode("utf-8")

# Load
today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

credential = DefaultAzureCredential()
container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)
container.upload_blob(blob_path, payload, overwrite=True)
print(f"Uploaded {len(payload)} bytes to {blob_path}")

# Verify: list blobs in the container
print("\nBlobs in container:")
for blob in container.list_blobs():
    print(f"  {blob.name}  ({blob.size} bytes)")

# Read back and confirm
raw = container.download_blob(blob_path).readall()
df = pd.DataFrame(json.loads(raw.decode("utf-8"))["hourly"])
print(f"\nFirst 5 rows:")
print(df.head())
```

In Week 10 you will pick up from the `raw/` blob and add the Transform step: reading those records back out, enriching them with an LLM, and writing the results to a `processed/` path.
