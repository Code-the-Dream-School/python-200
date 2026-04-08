# Week 10 Assignments

This week you learned to use an LLM as a data transformation step -- reading from Blob Storage, enriching records with a model call, and writing results back. You also saw how two lines of code separate the OpenAI API from Azure OpenAI, and why that distinction matters in enterprise environments.

The warmup checks your judgment about when LLMs belong in a pipeline and what makes Azure OpenAI different. The project has you build the Transform step end-to-end, connecting it to the Week 9 data you already have in Blob Storage.

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_10/`. Inside that folder, create:

1. `warmup_10.py` -- warmup exercises (conceptual answers as comments, code as functions)
2. `project_10.py` -- the LLM Transform pipeline
3. `outputs/` -- save the first 10 enriched records here (see project details below)

When finished, commit and open a PR as described in the [assignments README](README.md).

**Setup reminder:** You will need your OpenAI API key in a `.env` file (same setup as Weeks 5-7) and an active `az login` session for Blob Storage access.

```bash
uv pip install openai python-dotenv azure-storage-blob azure-identity pandas
```

# Part 1: Warmup

Put all warmup answers in `warmup_10.py`. Use comments to mark each section and question (e.g., `# --- LLMs as Transform ---` and `# Q1`). For conceptual questions, write your answer as a comment block. For code questions, write working Python.

## LLMs as Transform

### LLMs as Transform Question 1

For each task below, write a one-sentence comment saying whether you would use an LLM or deterministic code, and why.

- Parse the string `"Jan 5th, 2024"` into an ISO date format like `"2024-01-05"`.
- Classify a customer support ticket -- "my card was charged twice" -- into one of: billing, technical, or general.
- Calculate the average of a list of numbers.
- Extract the company name from a freeform job title like `"Sr. Data Eng @ Acme Corp (contract)"`.
- Determine whether a product review is more than 100 words long.

### LLMs as Transform Question 2

Your colleague has written the following pipeline prompt:

```python
system = "Summarize this product review in a few sentences."
```

In a comment block, explain what problem this creates downstream in a pipeline, and rewrite the prompt so it produces output that is easy to parse and store reliably.

### LLMs as Transform Question 3

Your dataset has 50,000 records and you need to run a classification call for each one using `gpt-4o-mini`. In a comment block, answer:

1. If each call takes 1 second on average, how long would sequential processing take?
2. What is one practical strategy to handle this more efficiently at scale, without changing models?

## Azure OpenAI

### Azure OpenAI Question 1

In a comment block, name two reasons an organization might use Azure OpenAI instead of calling the OpenAI API directly. Be specific -- "it's better" is not an answer.

### Azure OpenAI Question 2

When you switch from `OpenAI` to `AzureOpenAI`, the client initialization takes three Azure-specific parameters. In a comment block, name them and describe what each one is. (Do not include the standard `api_key` -- describe the Azure-specific ones.)

### Azure OpenAI Question 3

In a comment block, answer: when using `AzureOpenAI`, the `model` parameter in `chat.completions.create()` does not take a value like `"gpt-4o-mini"`. What does it take instead, and where do you find the right value to use?

# Part 2: Project -- LLM Transform Pipeline

Build `project_10.py`, a script that reads your Week 9 weather data from Blob Storage, classifies each hourly record with an LLM, and writes the enriched results back.

## Setup

At the top of your script, define your constants (fill in your own values):

```python
ACCOUNT_URL = "https://<your-account>.blob.core.windows.net"
CONTAINER = "pipeline-data"
```

## Step 1: Read

Download the raw weather data you uploaded in Week 9 from `raw/<today>/weather.json`. Parse the JSON and reshape the `"hourly"` parallel lists into a list of per-hour record dictionaries (each with `"time"`, `"temperature_2m"`, and `"precipitation"`).

If you did not complete Week 9, a fallback dataset is available at `assignments/resources/weather_raw.json` -- load it with `json.load()` and reshape it the same way.

## Step 2: Transform

For each record, call the OpenAI API to classify the conditions as `good`, `marginal`, or `bad` for outdoor running, based on temperature and precipitation. Use this system prompt exactly (so your mentor can compare results):

```python
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)
```

The user message for each record should be: `"Temperature: <value>C, Precipitation: <value>mm"`.

To keep costs and runtime manageable, process only the first 24 records (one day of hourly data). Add a fallback: if the model's response is not one of the three valid labels, store `"unknown"` instead.

Print a progress message every 6 records so you can see it running.

## Step 3: Write

Upload the enriched records (with the new `"conditions"` field) to `processed/<today>/weather_classified.json` in Blob Storage. Use `overwrite=True`.

## Step 4: Spot-Check

Download the processed blob, load it into a pandas DataFrame, and print:
- `df["conditions"].value_counts()`
- The first 5 rows of the DataFrame

## Step 5: Save Output

Save the first 10 enriched records to `outputs/first_10_records.json` so your mentor can inspect the results without running the script.

## Step 6: Reflect

Add a comment block at the top of `project_10.py` (3-5 sentences) answering: was classifying weather conditions for outdoor running actually a good use of an LLM? Could deterministic code have done this better? What would you lose or gain by switching to a rule-based approach (e.g., "temperature > 10 and precipitation < 1 → good")?

## Video

Record a short video (target: 3 minutes, max: 5). Show:

1. The script running in your terminal with no errors
2. The `value_counts()` output and first 5 rows printed to the terminal
3. The processed blob appearing in the Azure Portal under `pipeline-data/processed/<date>/`

Paste the video link in a comment at the top of `project_10.py`.
