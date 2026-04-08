# Week 9 Assignments

This week you moved from the Azure Portal into Python. You learned how scripts authenticate to Azure, how to read and write files in Blob Storage, and how to build an Extract + Load pipeline that pulls data from a REST API and stores it in the cloud. The warmup checks your understanding of those concepts and gives you some practice with the SDK. The project has you build the pipeline end-to-end.

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_09/`. Inside that folder, create:

1. `warmup_09.py` -- warmup exercises (conceptual answers as comments, code as functions)
2. `project_09.py` -- the Extract + Load pipeline
3. `outputs/` -- for any files you download from Blob Storage

When finished, commit and open a PR as described in the [assignments README](README.md).

**Setup reminder:** Make sure you have run `az login` in your terminal before running any of this week's scripts. All scripts in this assignment use `DefaultAzureCredential`, which relies on that session.

```bash
uv pip install azure-identity azure-mgmt-resource azure-storage-blob requests pandas
```

# Part 1: Warmup

Put all warmup answers in `warmup_09.py`. Use comments to mark each section and question (e.g., `# --- Azure Authentication ---` and `# Q1`). For conceptual questions, write your answer as a comment block. For code questions, write working Python functions.

## Azure Authentication

### Azure Authentication Question 1

In a comment block, answer: when you run a Python script locally that uses `DefaultAzureCredential`, what does it rely on to authenticate? What command must you have run first, and how does `DefaultAzureCredential` know to use it?

### Azure Authentication Question 2

In a comment block, answer: why can't a deployed pipeline (running on an Azure VM or container) use `az login` for authentication? What does it use instead, and why does the same Python code work without changes?

### Azure Authentication Question 3

You run a script that creates a `DefaultAzureCredential` and immediately gets an `AuthenticationError`. In a comment block, describe the two most likely causes and how you would diagnose each.

## Blob Storage

### Blob Storage Question 1

In a comment block, describe the three-level hierarchy of Azure Blob Storage in your own words. Give a concrete analogy that maps each level to something familiar (a filesystem, a filing cabinet, etc.).

### Blob Storage Question 2

For each scenario below, write one sentence in a comment block saying whether you would use Blob Storage or a relational database (like Azure SQL), and why.

- A REST API returns a JSON payload each hour. You need to store the raw responses for reprocessing later.
- Your pipeline produces a table of 50 million customer transactions that your analytics team queries by date range and customer ID every day.
- A computer vision model produces image embeddings as NumPy arrays. You need to save them between pipeline runs.

### Blob Storage Question 3

Write a function `list_container(container_client)` that prints the name and size (in bytes) of every blob in the container, one per line. The function should take a `ContainerClient` object as its only argument and return nothing.

### Blob Storage Question 4

Write a function `upload_text(container_client, blob_name, text)` that encodes a Python string as UTF-8 and uploads it as a blob, overwriting any existing blob with the same name. The function should take a `ContainerClient`, a blob name string, and a text string, and return nothing.

# Part 2: Project -- Extract + Load Pipeline

Build `project_09.py`, a script that implements a complete Extract + Load pipeline using the Open-Meteo weather API and Azure Blob Storage.

## Setup

At the top of your script, define these constants (fill in your own values):

```python
ACCOUNT_URL = "https://<your-account>.blob.core.windows.net"
CONTAINER = "pipeline-data"
```

You will need to create the `pipeline-data` container in your storage account before running the script. You can do this in the Azure Portal: navigate to your storage account, click "Containers," and create a new container named `pipeline-data` with private access.

## Step 1: Extract

Call the Open-Meteo API to retrieve 7 days of hourly weather data. Use the URL pattern below -- pick any city you like by substituting its latitude and longitude:

```text
https://api.open-meteo.com/v1/forecast?latitude=<lat>&longitude=<lon>&hourly=temperature_2m,precipitation&forecast_days=7
```

Charlotte, NC is `latitude=35.2271&longitude=-80.8431` if you want a default. Use `response.raise_for_status()` to catch errors early.

## Step 2: Serialize

Convert the API response to JSON bytes using `json.dumps()` and `.encode("utf-8")`.

## Step 3: Load

Upload the serialized data to your container at the path `raw/<today>/weather.json`, where `<today>` is today's date in ISO format (use `date.today().isoformat()`). Use `overwrite=True`.

Print a confirmation message showing the blob path and the number of bytes uploaded.

## Step 4: Verify

List all blobs in the container and print each one's name and size.

## Step 5: Read Back

Download the blob you just uploaded. Parse the JSON and load the `"hourly"` field into a pandas DataFrame. Print the first 5 rows.

Save the downloaded JSON to `outputs/weather_raw.json` so your mentor can inspect it without running the script.

## Video

Record a short video (target: 3 minutes, max: 5). Show:

1. The script running in your terminal with no errors
2. The blob appearing in the Azure Portal (navigate to your storage account, then Containers, then `pipeline-data`)
3. The DataFrame output printed to your terminal

Paste the video link in a comment at the top of `project_09.py`.
