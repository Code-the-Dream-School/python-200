# Week 10: LLMs in Pipelines

## Overview

Students learned to use LLMs as a data transformation step in a pipeline — not as a chatbot, but as a processing component that takes structured input and produces structured output. The week reframes everything students know about LLMs in a new context: batch processing, cost management, output parsing, and reliability. They read weather data from Blob Storage, classify each record with an API call, and write the enriched results back. A final lesson covers Azure OpenAI for enterprise environments.

## Key Concepts

**LLMs as a Transform step** — In an ETL pipeline, the Transform step reshapes or enriches data. LLMs are good at tasks that are too ambiguous for deterministic code: classifying freeform text, extracting entities from unstructured input, summarizing, or normalizing inconsistent values. They're poor at arithmetic, exact lookups, or anything that requires perfect reliability.

**Constrained, parseable output** — In a pipeline, you can't just ask for "a summary." You need output you can reliably parse: a single word, a JSON object, a fixed set of labels. The system prompt in this week's assignment ("Reply with that one word only — no punctuation, no explanation") is the pattern. If the model deviates, the pipeline needs a fallback.

**Cost and latency at scale** — An API call per record gets expensive and slow at scale. 50,000 records at 1 second each is 14 hours of sequential processing. Students learn to think about batch size, rate limits, and when to consider async or parallel processing — even if they don't implement it this week.

**Azure OpenAI** — The same models (GPT-4o, GPT-4o-mini) available through Microsoft's Azure infrastructure instead of OpenAI directly. The practical difference is two lines of code (`AzureOpenAI` instead of `OpenAI`, plus an `azure_endpoint` and `api_version`). The business reasons: data stays in a specific region, training opt-out, unified billing, compliance requirements.

## Common Questions

- **"When should I use an LLM vs. just writing an if/else?"** — If you can write a deterministic rule that reliably covers all cases, write the rule. LLMs are for cases where the input is too varied or ambiguous for rules to work well. The lesson's warmup Q1 is designed to surface this judgment.
- **"What happens if the model returns something unexpected?"** — The pipeline should handle it with a fallback (the assignment uses `"unknown"` for any response that isn't one of the three valid labels). Never assume an LLM's output will match your schema exactly.
- **"Why do we only process 24 records instead of all 168?"** — Cost and time. Processing the full week of hourly data (168 records) is 7x more API calls for a teaching exercise. Students learn the pattern with 24 records; the code is the same at any scale.
- **"What's a deployment name?"** — In Azure OpenAI, you deploy a model instance and give it a name. That deployment name (not the model name like `"gpt-4o-mini"`) is what you pass to the `model` parameter. Students who read the `03_Azure_OpenAI.md` lesson will have seen this — it's a common gotcha when switching from the OpenAI API.

## Watch Out For

- **Both `az login` and the OpenAI API key are required** — This week needs both: Blob Storage access (via `az login`) and the LLM calls (via `OPENAI_API_KEY` in `.env`). Students who forget either will hit an error. The assignment's setup reminder covers this, but it's worth confirming at the start of the session.
- **Week 9 data dependency** — The project reads from `raw/<today>/weather.json` in Blob Storage. If a student's Week 9 run used a different date, that path doesn't exist. Remind them of the fallback: `assignments/resources/weather_raw.json`.
- **API call costs** — 24 API calls is very cheap (fractions of a cent), but students who accidentally loop over all 168 records or run the script many times may use more credits than expected.
- **"Unknown" fallback masking bugs** — If a student's `value_counts()` shows many `"unknown"` values, it's worth investigating whether the model is returning unexpected output or whether the prompt isn't working as intended.

## Suggested Activities

1. **LLM vs. deterministic code judgment:** Give the warmup Q1 scenarios to the group verbally and ask for quick show-of-hands: LLM or deterministic code? Then discuss the edge cases — what if the date format is inconsistent? What if the job titles are in multiple languages? When does the "just write a rule" answer break down?

2. **Reflect on the assignment:** Ask students to share their reflection comment (warmup Q1 and the project's Step 6 reflection): was classifying weather conditions for outdoor running actually a good use of an LLM? What would a rule-based approach look like? What would you gain or lose by switching? This metacognitive question often sparks the most interesting discussions.

3. **Cost back-of-envelope:** Ask: "If `gpt-4o-mini` costs $0.15 per million input tokens, and each of your weather classification calls uses roughly 60 tokens, what would it cost to classify 1 million records?" Work through it together. This connects abstract API pricing to real engineering decisions.
