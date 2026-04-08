# Azure Blob Storage

Now that Python can authenticate to Azure, you need somewhere to put data. Azure Blob Storage is that place.

Blob Storage is Azure's service for storing unstructured data in the cloud -- files of any kind: JSON exports from APIs, CSVs, images, model artifacts, log files, pipeline outputs. It is the default landing zone for data in Azure pipelines, and you will use it throughout the rest of this course.

> TODO: add short video overview of Azure Blob Storage (~5 min)

For more on what Blob Storage is and when to use it:
- [Introduction to Azure Blob Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction)
- [Azure Blob Storage quickstart for Python](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python)

## Learning Objectives

By the end of this lesson, you will be able to:

- Describe the account, container, and blob hierarchy in Blob Storage
- Explain when to use Blob Storage vs. a relational database
- Use the azure-storage-blob SDK to upload, list, download, and delete blobs

## The Hierarchy: Account, Container, Blob

Blob Storage is organized into three levels.

A *storage account* is the top-level resource. Each account has a unique name in Azure and a URL of the form `https://<account-name>.blob.core.windows.net`. Your CTD environment already has a storage account provisioned in your resource group -- you can find it by navigating to your resource group in the Azure Portal.

A *container* is a grouping of blobs within the storage account -- analogous to a top-level folder. You can have many containers in a single account. A common pattern is one container per project or pipeline.

A *blob* is an individual file. Blobs are stored by name, and names can include forward slashes, which gives the appearance of a folder structure (e.g., `raw/2024-01-15/weather.json`). This is just a naming convention, not a true hierarchy -- more on this in the next lesson.

Think of it like a cloud-hosted filesystem: the storage account is the drive, containers are top-level directories, and blob names are paths within those directories.

## Blob Storage vs. a Database

Blob Storage is designed for *files*, not rows and columns. The rule of thumb: if you need to query data by value -- filter by date, join tables, aggregate by category -- use a relational database like Azure SQL or a warehouse like Snowflake. If you just need to store and retrieve files as-is, Blob Storage is the right choice. It is especially well-suited for large files, irregular data, and the raw and processed layers of a pipeline where you are writing entire datasets at once rather than querying individual records.

## The azure-storage-blob SDK

Install the package:

```bash
uv pip install azure-storage-blob
```

The SDK provides three client objects at different levels of the hierarchy:

- `BlobServiceClient` operates at the storage account level and is used to list or create containers.
- `ContainerClient` operates within a specific container -- use it to upload, list, download, and delete blobs. This is the one you will use most often.
- `BlobClient` gives you fine-grained control over a single blob.

For most pipeline work, `ContainerClient` is the right level of abstraction. You can create one directly:

```python
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
container = ContainerClient(
    account_url="https://<account>.blob.core.windows.net",
    container_name="my-container",
    credential=credential
)
```

Replace `<account>` with your storage account name (visible in the Azure Portal under your resource group) and `"my-container"` with the name of a container you have created.

## CRUD Operations

### Uploading a Blob

```python
container.upload_blob("hello.txt", b"hello world", overwrite=True)
```

The first argument is the blob name (its path within the container), the second is the content as bytes. The `overwrite=True` parameter means that if a blob with the same name already exists, it will be replaced. Without it, you will get an error if the blob already exists -- something to keep in mind for pipelines that may run more than once.

### Listing Blobs

```python
for blob in container.list_blobs():
    print(blob.name, blob.size)
```

`list_blobs()` returns an iterable of `BlobProperties` objects. The `name` attribute is the full blob path; `size` is the file size in bytes.

### Downloading a Blob

```python
data = container.download_blob("hello.txt").readall()
print(data)  # b'hello world'
```

`download_blob()` returns a `StorageStreamDownloader`. Call `.readall()` to get the full content as bytes.

### Deleting a Blob

```python
container.delete_blob("hello.txt")
```

## Working with Text

Pipeline data is usually text -- JSON, CSV, plain text. Since blobs are stored as bytes, you need to encode strings before uploading and decode after downloading:

```python
# Encoding before upload
text = "time,temp\n2024-01-15T00:00,12.3\n2024-01-15T01:00,11.8"
container.upload_blob("data.csv", text.encode("utf-8"), overwrite=True)

# Decoding after download
raw = container.download_blob("data.csv").readall()
text = raw.decode("utf-8")
print(text)
```

UTF-8 is the standard encoding for text data in pipelines. You will almost never need to use anything else.
