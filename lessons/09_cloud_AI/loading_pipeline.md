# Loading Pipeline Data to Blob Storage

*Applies Blob Storage to a realistic data task: pull data from an API, write results to the cloud. This is the Extract + Load skeleton for the capstone.*

**Sections:**

  * Designing a simple cloud-friendly folder structure in a container (e.g., raw/date/filename.json)
  * Reading from a public REST API with requests (brief review -- they know this from Week 1)
  * Serializing data (JSON, CSV) for storage
  * Uploading to a structured path in Blob Storage
  * Reading it back into a pandas DataFrame (closing the loop)

**Representative Code Sketch**:

```python
import requests, json
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

# extract
response = requests.get("https://api.example.com/data")
data = response.json()

# serialize
payload = json.dumps(data).encode()

# load to blob
credential = DefaultAzureCredential()
container = ContainerClient(
    "https://<account>.blob.core.windows.net",
    "pipeline-data",
    credential=credential
)
container.upload_blob("raw/2024-01-01/data.json", payload, overwrite=True)

# read back
import pandas as pd, io
raw = container.download_blob("raw/2024-01-01/data.json").readall()
df = pd.read_json(io.BytesIO(raw))
```

**External resources**: Azure Blob Storage Python samples (GitHub); requests library docs
