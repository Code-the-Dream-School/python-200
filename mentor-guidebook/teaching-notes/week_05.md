# Week 5: Introduction to AI

## Overview

Students shifted from building models to working with large language models as a service. The week covers what LLMs actually are (architecture, tokenization, embeddings), how to use the OpenAI API programmatically (completions, parameters, chatbots), prompt engineering techniques, and a grounding lesson in AI ethics. By the end, students are building a functioning chatbot and a job application helper.

## Key Concepts

**Tokens and embeddings** — LLMs don't read words; they read tokens (roughly word-pieces). Embeddings are high-dimensional vectors that represent the meaning of text — similar meanings cluster together in this space. Students don't need to implement these, but understanding the concepts helps them make sense of why LLMs behave the way they do.

**The completions API** — The fundamental interaction: send a list of messages (with roles: `system`, `user`, `assistant`), get a completion back. Key parameters to understand: `temperature` (randomness — 0 is deterministic, 1+ is creative), `max_tokens` (output length limit), `n` (number of completions to generate).

**Stateless API, stateful chatbot** — The API itself has no memory. Every call is independent. A chatbot maintains state by accumulating the conversation history in a list and sending the full history with each new message. This is a conceptually important pattern that students often miss.

**Prompt engineering** — The lesson's "Golden Rules" are worth knowing: be specific, give the model a role, provide examples (few-shot), and constrain the output format. Chain-of-thought prompting (asking the model to "think step by step") reliably improves performance on reasoning tasks.

**AI ethics** — Four key issues: bias in training data, energy and water consumption, misinformation (deepfakes, hallucinations, scams), and labor displacement. The lesson distinguishes between jobs likely to be automated vs. augmented.

## Common Questions

- **"Why does changing the temperature change the output so much?"** — At temperature=0, the model always picks the most likely next token. Higher temperatures give lower-probability tokens a chance to be selected, leading to more varied (and sometimes more creative) outputs. This is why temperature=0 is good for data extraction and high temperature is good for creative writing.
- **"Why does the chatbot forget things if I start a new session?"** — Each API call is stateless — the model has no persistent memory. The chatbot works by replaying the full conversation history on each call. Once the conversation list is cleared, the context is gone.
- **"What's a hallucination?"** — When an LLM confidently produces information that is simply false. It's not "lying" — the model generates plausible-sounding text based on patterns, without any mechanism to verify facts.
- **"How much does the API cost?"** — Students are usually using free-tier credits. Costs scale with token count. `gpt-4o-mini` is very cheap; `gpt-4o` is significantly more expensive. The lesson focuses on `gpt-4o-mini`.

## Watch Out For

- **API key management** — Students must have a `.env` file with their `OPENAI_API_KEY`. The most common issue is the key not loading (wrong file location, wrong variable name, or the `load_dotenv()` call missing). Check this first if a student says the API isn't working.
- **Rate limits** — OpenAI's free tier has rate limits. Students running many API calls quickly may hit these. If they get a `RateLimitError`, they just need to wait a moment.
- **Using AI to do the AI assignment** — The prompt engineering warmup asks students to reflect on why prompts work. Using an LLM to write those answers defeats the purpose. Encourage genuine experimentation.

## Suggested Activities

1. **Live temperature demo:** Ask a student to call the same prompt with `temperature=0` three times, then with `temperature=1` three times. Compare the outputs. This makes the concept visceral in a way the lesson description doesn't.

2. **Prompt engineering challenge:** Give the group a bad prompt ("Write something about this company") and a target output format (a three-bullet executive summary). Have students iterate on the prompt live until they get consistent, well-formatted output. Discuss what changes made the biggest difference.

3. **Ethics discussion:** Ask: "Have any of you encountered an AI hallucination that affected you — in something you read, used, or submitted?" Then: "What would you do differently now that you know how these systems work?"
