# Week 7: AI Agents

## Overview

Students built their first AI agents — systems where an LLM doesn't just generate text, but actively decides which tools to call, observes the results, and reasons toward a goal. The week moves from theory (the ReAct framework) through a from-scratch implementation, a multi-tool data analysis agent, the smolagents framework, and GitHub Copilot as an agent-mode coding assistant. The assignment revisits the Week 1 World Happiness dataset through a conversational agent interface.

## Key Concepts

**The ReAct loop** — Agents operate in a cycle: Reason (decide what to do next), Act (call a tool), Observe (read the result), repeat until done. Each iteration involves an API call. Students should be able to trace this loop through the message history.

**Tool definitions** — Tools are Python functions paired with a JSON schema that describes their name, purpose, and parameters. The LLM reads the schema and decides when and how to call a tool. Writing good tool schemas (clear names, accurate descriptions) is as important as writing good prompts.

**Tool-based vs. code-based agents** — A tool-based agent (like `ToolCallingAgent` in smolagents) calls pre-defined functions. A code-based agent (like `CodeAgent`) generates and executes Python code on the fly. Code agents are more flexible but harder to constrain and carry security considerations.

**smolagents** — A lightweight framework from Hugging Face that simplifies agent creation. The `@tool` decorator auto-generates the JSON schema from the function's type hints and docstring. Students who understand the from-scratch implementation will appreciate how much boilerplate smolagents eliminates.

**GitHub Copilot in agent mode** — Students use Copilot to diagnose and fix intentional bugs in a small ETL script (`mini_etl.py`), guided by a failing test suite. The lesson emphasizes reviewing AI-generated code critically rather than accepting it blindly.

## Common Questions

- **"How does the model know when it's done?"** — The model generates a final response (instead of a tool call) when it believes it has enough information to answer. `max_steps` acts as a safety valve — the lesson sets this to prevent infinite loops.
- **"Why do we need a JSON schema? Can't the model just read the function?"** — The model doesn't have access to your code. The schema is how you communicate the function's interface in a format the model can reason about. smolagents automates this from docstrings.
- **"What's the security risk of CodeAgent?"** — The agent generates and runs arbitrary Python code on your machine. A malicious or poorly constrained agent could in principle read files, make network requests, or execute destructive commands. `additional_authorized_imports` limits this somewhat.
- **"Why did the agent fail the first time (before I added the correlation tool)?"** — The warmup recreates a scenario where the agent hits the tool-round limit because no tool exists for the requested operation. This is intentional — it teaches students that the tool set defines what an agent can do.

## Watch Out For

- **API key required throughout** — Every agent interaction calls the OpenAI API. Students without a working key will be blocked immediately. Check `.env` setup first.
- **smolagents version sensitivity** — smolagents is actively developed and its API changes between versions. If students hit import errors or unexpected behavior, check that they installed the version referenced in the lesson.
- **Agent verbosity** — smolagents logs every step of the ReAct loop. Students may be confused by the volume of output. This is normal and worth celebrating — it shows the agent "thinking."
- **mini_etl.py bugs are intentional** — The two bugs in the mini-ETL script are deliberate, designed to be caught by the test suite. If a student asks "is this broken?" the answer is yes, on purpose — that's the Copilot exercise.
- **`reset=False` in the assignment** — The project runs five queries in sequence with `reset=False` so the agent retains context. If a student uses `reset=True` (the default), the agent will lose context between queries and produce inconsistent results.

## Suggested Activities

1. **Trace the ReAct loop:** Ask a student to share the output from warmup Q6 (the full `messages` list printed as JSON). Walk through it together: identify each `system`, `user`, `assistant`, and `tool` message. Ask: "How many API calls did this agent make to answer one question?"

2. **Tool vs. code agent comparison:** Share the warmup Q8 results from a student who tried the scatter plot prompt with both agent types. Did the `ToolCallingAgent` change the dot color to green? (Probably not — it can only call the tools it has.) Did the `CodeAgent`? Discuss what this reveals about when each is more useful.

3. **Agent design exercise:** Ask the group: "You're building an agent for a hospital triage desk that helps classify patient urgency from their intake form. What tools would you give it? What would you *not* let it do? Would you use a tool-based or code-based agent?"
