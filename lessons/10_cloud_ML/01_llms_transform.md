# LLMs as a Transform Step

*Conceptual framing. Reorients the LLM skills from Weeks 5-7 toward pipeline thinking.*

**Sections:**

* Review: where LLMs fit in ETL -- they belong in the Transform step, not Extract or Load
* Tasks LLMs are well-suited for in pipelines: classification, extraction, summarization, normalization
* Tasks where LLMs are a poor fit: anything deterministic, math, tasks with a correct answer (use code instead)
* Cost and latency reality: API calls cost money and take time -- pipeline design needs to account for this
* Batching strategy: why you don't call an LLM row-by-row on a large dataset; simple batching patterns
* Structuring prompts for reliability in pipelines: constrained outputs (JSON, single words, yes/no) over open-ended generation

**External resources**: OpenAI cookbook: structured outputs; practical guide to LLM costs and rate limits
