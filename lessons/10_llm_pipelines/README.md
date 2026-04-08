# Week 10: LLMs in Pipelines

In Weeks 5 through 7, you used LLMs interactively -- building chatbots, augmenting them with retrieved knowledge, and wiring them into agents that could reason and act. In Week 9, you learned to store and retrieve data in Azure Blob Storage. This week you connect those two skills.

The core idea is a shift in how you think about language models: not as a conversational partner, but as a data processing step. LLMs are remarkably good at certain transformation tasks -- classifying freeform text, extracting structured fields from messy data, normalizing inconsistent values -- that would be tedious or impossible to handle with deterministic code. When one of those tasks sits in the middle of a data pipeline, an LLM call is a legitimate engineering choice.

> For an introduction to the course, and a discussion of how to set up your environment, please see the [Welcome](../README.md) page.

## Topics

1. [LLMs as a Transform Step](01_llms_transform.md)
Reframes the LLM skills from Weeks 5-7 for a pipeline context. Covers where LLMs belong in ETL, what kinds of tasks they handle well vs. poorly, the practical realities of cost and latency at scale, and how to design prompts that produce reliable, parseable output.

2. [Transforming Blob Data with OpenAI](02_blob_data.md)
The hands-on lesson. Reads the weather data you loaded in Week 9, applies an LLM classification step to each record, handles unexpected responses gracefully, writes enriched results back to a processed Blob Storage path, and spot-checks the output with pandas.

3. [Azure OpenAI: A Note for Production](03_Azure_OpenAI.md)
A short reading lesson. Explains what Azure OpenAI is, why most enterprise environments use it instead of the public OpenAI API, and what two lines of code change when you make the switch. No hands-on exercise required.

## Week 10 Assignments

Once you finish the lessons, head on over to the [assignments](../../assignments/README.md) to get more hands-on practice with the material.
