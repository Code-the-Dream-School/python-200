# LLMs as a Transform Step

In Weeks 5 through 7, you used language models the way most people first encounter them: as a conversational interface. You sent a message, got a response, maybe kept a conversation going. The model was a partner you were talking to.

This week, the framing shifts. An LLM is no longer the end product -- it is a processing step inside a larger pipeline. A script reads a dataset from Blob Storage, sends each record through a model call, and writes the enriched results back. The model never sees a user. It just does work.

This is a genuinely useful pattern in data engineering, but it comes with a different set of constraints than interactive use. This lesson covers what those constraints are and how to work with them.

For reference:
- [OpenAI Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs)
- [OpenAI API rate limits and best practices](https://platform.openai.com/docs/guides/rate-limits)

## Learning Objectives

By the end of this lesson, you will be able to:

- Place LLMs correctly in an ETL pipeline (Transform, not Extract or Load)
- Identify which data tasks are good candidates for LLM processing and which are not
- Describe the cost and latency implications of LLM calls at scale
- Design prompts that return constrained, parseable output

## Where LLMs Fit in ETL

A data pipeline typically has three stages:

- *Extract*: pull data from a source (an API, a database, a file)
- *Transform*: clean, enrich, or reshape the data
- *Load*: write the result to a destination (a database, a storage service, a downstream system)

LLMs belong in the Transform step. They receive data, process it, and return something new. They are not well-suited for Extract or Load: those steps involve I/O operations -- making HTTP calls, reading files, writing to storage -- which are handled deterministically by the surrounding code. The LLM just handles the transformation logic.

## What LLMs Do Well

The tasks where LLMs add the most value in pipelines tend to involve language that is too variable or ambiguous for rule-based code to handle reliably.

*Text classification* is the most common use case. Given a customer support ticket, classify it as billing, technical, or general. Given a job posting, classify it as entry-level, mid-level, or senior. These require reading comprehension and judgment -- a regex cannot reliably do them.

*Field extraction* is another strong fit. Given a freeform address string like "Apt 4B, 123 Main St, New York NY 10001", extract the city and zip code. Given a job title like "Sr. Data Eng @ Acme Corp", extract the company name. The input is irregular; the output you need is structured.

*Text normalization* covers cases like resolving inconsistent values -- "New York", "NYC", "New York City", and "N.Y.C." should all map to "New York, NY". Code-based approaches require you to anticipate every variant in advance; an LLM handles novel variants naturally.

*Summarization* works well when you need to reduce long text to a shorter form -- collapsing a 500-word product review into a single sentence, for example.

## What LLMs Do Poorly

The flip side is equally important. LLMs are the wrong tool when the task has a correct answer that code can compute reliably.

Arithmetic belongs in code. Date parsing belongs in code. Sorting, filtering, and aggregating structured data belongs in pandas or SQL. If you can write a function that produces the right answer deterministically, you should. LLM calls add latency, cost, and the possibility of inconsistent output. Use them only when the task genuinely requires language understanding.

A useful heuristic: if you can write a unit test with a single expected output, use code. If the "correct" answer requires judgment or reading comprehension, consider an LLM.

## Cost and Latency at Scale

When you use an LLM interactively, one API call goes unnoticed. In a pipeline processing thousands of records, those calls add up.

As a rough reference using `gpt-4o-mini` (pricing as of 2024 -- check the [OpenAI pricing page](https://openai.com/api/pricing/) for current rates):

- A short classification call with a ~200-token input and a 5-token output costs about $0.00003.
- For 10,000 records: roughly $0.30 in API costs.
- For 100,000 records: roughly $3.00.

Cost is usually manageable with efficient, smaller models. Latency is the bigger concern. Each API call takes roughly 0.5 to 2 seconds. Calling an LLM sequentially for 10,000 records takes two to six hours of wall-clock time. For most pipeline use cases, that is too slow.

The practical solution for large datasets is to use OpenAI's [Batch API](https://platform.openai.com/docs/guides/batch), which processes requests asynchronously at reduced cost. For smaller datasets -- the kind you are working with in this course -- sequential calls are fine. The important thing is to be aware of the tradeoff before committing to an LLM step in a high-volume pipeline.

## Designing Prompts for Pipelines

Interactive LLM use rewards rich, open-ended prompts. Pipeline use rewards the opposite: precise, constrained prompts that return output your code can parse without ambiguity.

Consider the difference between these two prompts for sentiment analysis:

```text
Bad:  "Describe the sentiment of this customer review."
Good: "Classify the sentiment of this customer review.
       Reply with exactly one word: positive, negative, or neutral."
```

The first prompt might produce "This review expresses a mildly negative sentiment overall, though with some positive undertones." That is useful to a human reader but extremely difficult to parse in a pipeline. The second prompt constrains the output to a known set of values.

For more complex output, asking for JSON is even better:

```text
"Classify the sentiment and extract the main topic.
 Reply with valid JSON only, using this exact format:
 {\"sentiment\": \"positive\", \"topic\": \"shipping\"}"
```

JSON output is easy to parse with `json.loads()` and makes it straightforward to add multiple fields to a single call. The tradeoff is that it adds a small amount of prompt overhead and occasionally the model produces malformed JSON, which you need to handle.

For this week, we will use single-word constrained output. It is simpler to implement and robust enough for the task at hand. When you encounter a pipeline task that needs multiple structured fields per record, JSON output is worth the extra parsing code.
