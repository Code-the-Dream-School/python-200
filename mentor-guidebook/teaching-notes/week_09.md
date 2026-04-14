# Week 9: Data in the Cloud

## Overview

Students moved from the Azure portal into Python, writing scripts that interact with Azure services programmatically. The week covers identity-based authentication with `DefaultAzureCredential`, Azure Blob Storage (creating, reading, listing, and deleting files), and an end-to-end Extract+Load pipeline that pulls weather data from a public API and stores it in the cloud. This is the first week where students are writing real cloud code.

## Key Concepts

**`DefaultAzureCredential`** — Azure's recommended way to authenticate Python scripts. It tries a chain of credential sources in order: environment variables, managed identity, Azure CLI session, etc. Locally, it uses the `az login` session. In production (on a VM or container), it uses a managed identity — the same code works in both environments without changes.

**Azure Blob Storage hierarchy** — Three levels: storage account (top-level resource in Azure) → container (like a folder or bucket) → blob (the actual file). The analogy: account = the filing cabinet, container = a drawer, blob = a document inside.

**ETL pattern** — Extract (pull data from a source), Transform (reshape or clean it), Load (store it somewhere). This week is Extract + Load only; the Transform step comes in Week 10.

**Blob paths as structure** — Blobs have flat names, but you can use `/` in the name to create a logical folder structure (e.g., `raw/2025-06-01/weather.json`). This date-partitioned layout is a standard pattern for data pipelines — it makes it easy to find data from a specific day and avoid overwriting previous runs.

**`raise_for_status()`** — After an HTTP request, this method raises an exception if the status code indicates failure (4xx or 5xx). This is the right way to catch API errors early rather than continuing with bad data.

## Common Questions

- **"Why `DefaultAzureCredential` instead of just using an API key?"** — API keys are secrets that can be accidentally committed to GitHub or leaked. `DefaultAzureCredential` uses your existing authenticated identity (the `az login` session) — no secret to manage locally. In production, managed identities eliminate secrets entirely.
- **"What's the difference between Blob Storage and a database?"** — Blob Storage is for files (JSON, CSV, images, videos — any binary or text file). A database is for structured, queryable data where you need to filter by column values, join tables, etc. Blob Storage is much cheaper and simpler for raw data storage.
- **"Why is the blob path `raw/<today>/weather.json` and not just `weather.json`?"** — Date partitioning. If you run the pipeline every day, you get a separate file per day and can look back at historical data. If you just used `weather.json`, each run would overwrite the previous one (unless you use a different name or `overwrite=False`).

## Watch Out For

- **`az login` must be active** — This is the most common failure mode. If a student runs the script without a valid `az login` session, `DefaultAzureCredential` will fail with an `AuthenticationError`. The fix is `az login` in the terminal, then re-run the script.
- **Container must exist before uploading** — The `pipeline-data` container must be created in the Azure portal before the script runs. Students who skip this step will get a container-not-found error. Walk them through creating it if needed.
- **The date dependency between Week 9 and Week 10** — Week 9 uploads to `raw/<today>/weather.json`. Week 10 downloads from the same path using `date.today()`. If a student runs Week 9 on one day and Week 10 on another day, the path won't match. The fix: use the fallback dataset in `assignments/resources/weather_raw.json` for the Week 10 project.
- **Account URL** — Students need to fill in their specific storage account URL (the `ACCOUNT_URL` constant). This is in the Azure portal on their storage account's overview page. If they leave the placeholder, the script fails immediately.

## Suggested Activities

1. **Authentication chain walkthrough:** Ask: "If your script is running on an Azure VM (not your laptop), and you can't run `az login` there, how does it authenticate?" Walk through the `DefaultAzureCredential` chain — specifically the managed identity option. This is what makes the same code work locally and in production.

2. **Blob Storage vs. database decision drill:** Give students three scenarios and ask which storage type to use: (1) storing hourly API responses as JSON files, (2) storing a 50 million-row transaction table that needs daily queries by date and customer, (3) storing model training artifacts. Discuss the reasoning.

3. **Pipeline verification:** Ask a student to screen-share their Azure portal with the `pipeline-data` container open. Walk through what they see: the blob name, size, last-modified date, folder structure. Then show it in the terminal via `az storage blob list` or the Python SDK. Connecting the code to the visual is useful.
