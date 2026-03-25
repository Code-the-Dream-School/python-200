# Transforming Blob Data with OpenAI

*The hands-on lesson. Reads data from Blob Storage (Week 9), applies an LLM transformation, writes results back.*

Sections:

* Reading a dataset from Blob Storage into Python (brief review of Week 9 pattern)
* Designing a prompt for a structured pipeline task: classification or field extraction
* Iterating over records and collecting LLM responses
* Handling failures gracefully: what to do when the API returns an unexpected format
* Writing enriched results back to Blob Storage under a processed/ path
* Spot-checking results: reading back a sample and inspecting what the LLM produced

**Representative code sketch**:

```python
import json, io
import pandas as pd
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

# read raw data from blob (Week 9 pattern)
credential = DefaultAzureCredential()
container = ContainerClient(
    "https://<account>.blob.core.windows.net",
    "pipeline-data",
    credential=credential
)
raw = container.download_blob("raw/2024-01-01/data.json").readall()
records = json.loads(raw)

# transform with OpenAI
client = OpenAI()
results = []
for record in records:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Classify the sentiment of the text. "
                           "Reply with exactly one word: positive, negative, or neutral."
            },
            {"role": "user", "content": record["text"]}
        ]
    )
    label = response.choices[0].message.content.strip().lower()
    results.append({**record, "sentiment": label})

# write enriched data back
container.upload_blob(
    "processed/2024-01-01/data.json",
    json.dumps(results).encode(),
    overwrite=True
)
```
**External resources**: OpenAI structured outputs guide; OpenAI rate limits and best practices
