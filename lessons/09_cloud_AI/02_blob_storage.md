# Azure Blob Storage

*Covers the core concepts and CRUD operations for Blob Storage.*

**Sections:**
* What Blob Storage is: containers, blobs, the hierarchy (account → container → blob)
* When to use Blob Storage vs. a database (unstructured data, files, pipeline artifacts)
* The azure-storage-blob SDK: BlobServiceClient, ContainerClient, BlobClient
* Uploading a file
* Downloading a file
* Listing blobs in a container
* Deleting a blob
* Working with text and binary data (upload a CSV, upload an image)

**Representative Code Sketch**:

```python
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = BlobServiceClient(
    account_url="https://<account>.blob.core.windows.net",
    credential=credential
)

container = client.get_container_client("my-container")

# upload
container.upload_blob("hello.txt", b"hello world", overwrite=True)

# list
for blob in container.list_blobs():
    print(blob.name)

# download
data = container.download_blob("hello.txt").readall()
```

**External Resources:** Azure Blob Storage Python quickstart (Microsoft Learn); Blob Storage concepts overview
