# Transforming Blob Data with OpenAI

In Week 9, you built an Extract + Load pipeline that pulled weather data from the Open-Meteo API and stored it in Blob Storage at `raw/<date>/weather.json`. Today you add the missing middle step: Transform. You will read that raw data back out, send each hourly record through an LLM to classify the conditions, and write the enriched results to `processed/<date>/weather_classified.json`.

This is the pattern you will use as the Transform step in the Week 11 capstone. By the end of this lesson, the full ETL skeleton will be in place.

For reference:
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)

## Learning Objectives

By the end of this lesson, you will be able to:

- Read raw data from Blob Storage and reshape it into individual records
- Design a constrained prompt for a pipeline classification task
- Iterate over records, call the OpenAI API for each, and collect results
- Handle unexpected model responses gracefully with a fallback value
- Write enriched records back to Blob Storage and spot-check the results

## Setup

Make sure your `.env` file contains your OpenAI API key (same setup as Weeks 5-7), and install the packages if needed:

```bash
uv pip install openai python-dotenv azure-storage-blob azure-identity
```

## Reading Raw Data from Blob Storage

The Week 9 pipeline stored the full Open-Meteo response as JSON. That response has a nested structure: under the `"hourly"` key you will find parallel lists for `"time"`, `"temperature_2m"`, and `"precipitation"`. For the transform step, it is more convenient to work with individual records -- one dictionary per hour -- so you will reshape the data after reading it:

```python
import json
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from datetime import date

ACCOUNT_URL = "https://<account>.blob.core.windows.net"
CONTAINER = "pipeline-data"

today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

credential = DefaultAzureCredential()
container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

raw = container.download_blob(blob_path).readall()
data = json.loads(raw.decode("utf-8"))

# Reshape from parallel lists into a list of records
hourly = data["hourly"]
records = []
for i in range(len(hourly["time"])):
    record = {
        "time": hourly["time"][i],
        "temperature_2m": hourly["temperature_2m"][i],
        "precipitation": hourly["precipitation"][i],
    }
    records.append(record)

print(f"Loaded {len(records)} hourly records")
```

`records` is now a list of 168 dictionaries (7 days * 24 hours), one per hour.

## Designing the Prompt

The task: classify each hourly record as `good`, `marginal`, or `bad` for outdoor running, based on temperature and precipitation. This is a reasonable LLM task because the classification requires judgment -- there is no exact formula for what makes conditions "good" for running. People weigh temperature, rain, and personal preference differently.

Following the principle from the previous lesson, we want a constrained system prompt that returns exactly one word:

```python
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)
```

The user message for each record will be a brief description of the conditions:

```python
def make_user_message(record):
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )
```

## Iterating Over Records

With the prompt in place, iterate over the records and call the API for each one. Load your API key from `.env` first:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

enriched = []
for record in records:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(record)},
        ]
    )
    label = response.choices[0].message.content.strip().lower()
    enriched_record = {**record, "conditions": label}
    enriched.append(enriched_record)
```

`{**record, "conditions": label}` creates a new dictionary with all the original fields plus the new `"conditions"` key. The original `record` is not modified.

## Handling Unexpected Responses

The constrained prompt works well in practice, but models occasionally produce output that does not match the expected format -- an extra word, a sentence, a capitalization variant. In a pipeline, you do not want one unexpected response to crash the entire run. Add a validation step:

```python
VALID_LABELS = {"good", "marginal", "bad"}

for record in records:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(record)},
        ]
    )
    raw_label = response.choices[0].message.content.strip().lower()
    label = raw_label if raw_label in VALID_LABELS else "unknown"
    enriched.append({**record, "conditions": label})
```

Any response outside the expected set is recorded as `"unknown"` rather than raising an error. You can review `"unknown"` records later to see whether the prompt needs adjustment.

## Writing Enriched Results Back to Blob Storage

Upload the enriched records to a `processed/` path, keeping the same date folder:

```python
processed_path = f"processed/{today}/weather_classified.json"
payload = json.dumps(enriched).encode("utf-8")
container.upload_blob(processed_path, payload, overwrite=True)
print(f"Uploaded {len(payload)} bytes to {processed_path}")
```

## Spot-Checking the Results

Before trusting the output, take a quick look at the distribution of labels:

```python
import pandas as pd

df = pd.DataFrame(enriched)
print(df["conditions"].value_counts())
print(df.head(10))
```

If the distribution looks wildly off -- for example, everything classified as "bad" on a mild sunny day -- that is a signal the prompt needs tuning. A quick sanity check like this is cheap and often catches problems early.

## Putting It Together

Here is the complete Transform script, from reading the raw blob to writing the processed output:

```python
import json
import os
from datetime import date
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

load_dotenv()

ACCOUNT_URL = "https://<account>.blob.core.windows.net"
CONTAINER = "pipeline-data"
VALID_LABELS = {"good", "marginal", "bad"}
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

def make_user_message(record):
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

# Read
today = date.today().isoformat()
credential = DefaultAzureCredential()
container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

raw = container.download_blob(f"raw/{today}/weather.json").readall()
data = json.loads(raw.decode("utf-8"))

hourly = data["hourly"]
records = []
for i in range(len(hourly["time"])):
    records.append({
        "time": hourly["time"][i],
        "temperature_2m": hourly["temperature_2m"][i],
        "precipitation": hourly["precipitation"][i],
    })

print(f"Loaded {len(records)} records")

# Transform
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enriched = []
for i, record in enumerate(records):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(record)},
        ]
    )
    raw_label = response.choices[0].message.content.strip().lower()
    label = raw_label if raw_label in VALID_LABELS else "unknown"
    enriched.append({**record, "conditions": label})
    if (i + 1) % 24 == 0:
        print(f"  Processed {i + 1} records...")

# Load
processed_path = f"processed/{today}/weather_classified.json"
container.upload_blob(processed_path, json.dumps(enriched).encode("utf-8"), overwrite=True)
print(f"Uploaded to {processed_path}")

# Spot-check
df = pd.DataFrame(enriched)
print("\nLabel distribution:")
print(df["conditions"].value_counts())
```
In Week 11, you will take this Transform script and the Week 9 Extract + Load script and wire them together into a single orchestrated Prefect flow.
