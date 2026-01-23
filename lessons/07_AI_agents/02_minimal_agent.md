# Minimal Tool Agent

This notebook builds a minimal agent that can answer a simple question:

> "What time is it right now?"

We use this tiny agent to make the ReAct pattern clear:

- LLMs cannot run code.
- LLMs can only ask you to run code (a tool).
- Your Python code runs the tool and sends the result back to the model.
- Sometimes you need two API calls:
  - one where the model asks for a tool,
  - one where the model uses the tool result to answer.

We also show that the agent can choose not to use a tool, for example when answering a conceptual question.

## How ReAct Agent Works

Before we write any agent code, we need to understand what is happening.

Key points:

- The model cannot run Python.
- The model cannot read your files or your system clock.
- The model can only produce text.

So how can it use a tool like `get_current_time`?

1. You tell the model what tools are available (using the `tools` parameter in the API call).
2. You send a chat request (system + user messages + tools).
3. The model looks at the conversation and decides to either:
   - answer directly, or
   - return a special `tool_calls` field.
4. A `tool_call` is the model saying:
   - 'Please run this function and tell me the result.'
5. Your Python code sees the `tool_call`, runs the local Python function, and gets a result.
6. You send that result back to the model as a new message with `role: tool`.
7. You call the model again with the updated messages so it can see the tool result and give a final answer.

That is the ReAct loop:

> think -> ask for a tool -> Python acts -> model sees the result -> model finishes the answer.

When you set tool_choice="auto", the model is not running any special tool-selection algorithm. It simply reads the conversation and the tool descriptions you provide (the name, description, and parameters fields) and uses its normal language-model reasoning to decide whether a tool would help answer the user’s question. If it believes a tool is appropriate, it outputs a structured tool_calls block instead of a normal answer. This is just the model predicting text that follows the function-call schema it was trained to produce. The model does not run code or know the tool results by itself — it only requests the tool. Your Python code actually executes the function and sends the result back as a tool message, after which the model can finish the answer.

## Implementation

### Initial Setup

```python
from dotenv import load_dotenv
from openai import OpenAI
import os

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')
```

Output:

    Successfully loaded environment variables from .env
    OpenAI client created.

### Baseline query

First, we ask the model for the current time without giving it any tools.

The model has no real access to your system clock, so it will have to guess.
This is the kind of question where a tool would help.

```python
baseline_messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'What time is it right now?'},
]

baseline_response = client.chat.completions.create(
    model='gpt-4.1-mini',
    messages=baseline_messages,
)
print(baseline_response.choices[0].message.content)
```
Output:

    I’m not able to access real-time information. Please check the current time on your device or through an online time service.

### Tool Definition

Now we define a simple Python function that returns the current local time.

Important:

- This function is run by Python, not by the model.
- The model can only request that we run it.

```python
from datetime import datetime

def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

get_current_time()
```

Output:

    '2026-01-19 17:30:38'

### Passing the tool description to the model

The model does not see our Python code. It only sees tool definitions, a list of tool schemas. These are defined using json. 

We use a `tools` list to describe each tool:

- `name`: the name of the function
- `description`: natural language description to help the model decide when to use it
- `parameters`: a specifications of the function's arguments (in this case there are no arguments)

When we add more complex functions with parameters, we will see more complexity in the following tool list

```python
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]
print('Tools list defined with one tool: get_current_time')
```

Output:

    Tools list defined with one tool: get_current_time

A bit more on the tool specification:

Before the model can call any Python function, we have to describe that function in a structured way so the model knows it exists and knows what arguments it should provide. The important thing to understand is that the model does not see your actual Python code directly. Instead, it only sees a *description* of the function written using a format based on the JSON Schema (we feed it one json schema for each item in the list in tools -- here just one). 

In this example, the only tool we expose is the `get_current_time` function. If we later add more tools, the overall shape stays the same; only the details change. We will add more json-specified functions to tell the LLM what tools it has available. 

The main take-home message is that this dictionary is effectively a machine-readable description of your function. The LLM uses this description (not your Python code) to decide whether to call the tool and how to format the arguments.

Optionally, we can also provide a system prompt to the model. The system prompt tells the model how to behave.

We want the model to:

- Use the `get_current_time` tool only when needed.
- Answer directly when no tool is needed.
- Briefly state in the final answer whether it used a tool.

### Implementing the ReAct Agent

Now we write a `run_agent` function that:

1. Builds the initial `messages` list.
2. Makes the first API call with `tools=tools` and `tool_choice='auto'.`
3. Checks if there are any `tool_calls`.
4. If there are tool calls:
   - runs the tool(s) in Python,
   - appends the tool results to `messages` (as `role: tool`),
   - makes a second API call to get the final answer.
5. If there are no tool calls:
   - returns the first response as the final answer.

```python
import json

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''
```

#### Why do we sometimes call the API twice?

This is a common point of confusion.

##### First call

When we call `client.chat.completions.create` with `tools=tools` and `tool_choice='auto'`, the model:

- reads the conversation
- decides whether it needs a tool
- returns a message that may include `tool_calls`

If `tool_calls` is present, that first response is not a final answer. It is the model saying:

> 'I am not done yet. Please run this tool and then come back to me with the result.'

In that case, 
- looks at `tool_calls`
- runs the corresponding Python function (`get_current_time`)
- builds a new message with `role: tool` that contains the tool result

##### Second call

We then call the API again with the updated `messages` list, which now includes:

- the original system + user messages
- the model's first response (where it asked for the tool)
- the `tool` message with the tool result

Only now does the model have all the information it needs to give a final answer.

So the pattern is:

1. First call: model may request tools
2. Python: runs tools and adds tool outputs as messages
3. Second call: model sees tool outputs and gives final answer

### Agent Test 1: Query requiring tool use

Now we use `run_agent` on the same question as before:

> 'What time is it right now?'

This time, the model should:

- decide to call `get_current_time`,
- use the real tool result in its final answer,
- briefly mention that it used a tool.

```python
answer_with_agent = run_agent('What time is it right now?')
print(answer_with_agent)
```

Output:

    First response received from model...
    ChatCompletion(id='chatcmpl-CzsBdnd9VNr7XiA1RAW18iFmnqMra', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_y2pzEhksbc4YKDDIDW4iGPjx', function=Function(arguments='{}', name='get_current_time'), type='function')]))], created=1768862337, model='gpt-4.1-mini-2025-04-14', object='chat.completion', service_tier='default', system_fingerprint='fp_376a7ccef1', usage=CompletionUsage(completion_tokens=11, prompt_tokens=75, total_tokens=86, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    
    Agentic mode engaged...
    
    Tool called: get_current_time
    Tool result: 2026-01-19 17:38:58
    
    Second response received from model...
    ChatCompletion(id='chatcmpl-CzsBfDc7VB5g7WnzFbCnrahvFYN7q', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='The current time is 17:38:58.', refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None))], created=1768862339, model='gpt-4.1-mini-2025-04-14', object='chat.completion', service_tier='default', system_fingerprint='fp_376a7ccef1', usage=CompletionUsage(completion_tokens=11, prompt_tokens=77, total_tokens=88, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    
    The current time is 17:38:58.

### Agent Test 2: Query not requiring tool use

Now we ask a conceptual question that does not require the current time, such as:

> 'Explain what an LLM agent is (including the ReAct framework), in two sentences.'

The model should:

- answer directly,
- not call the tool,
- briefly mention that it did not use a tool.

```python
answer_conceptual = run_agent(
    'Explain what an LLM agent is (including the ReAct framework), in two sentences.'
)
print(answer_conceptual)
```

Output:

    First response received from model...
    ChatCompletion(id='chatcmpl-CzsBm10zUIBNDlRTScZUf7MnGyX0c', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='An LLM agent is an AI system that leverages a large language model to perform tasks by understanding and generating human-like text, often interacting with external tools or environments to complete complex goals. The ReAct framework enhances LLM agents by combining reasoning and acting in an interleaved manner, allowing the agent to reason about the problem, take actions based on that reasoning, observe outcomes, and iteratively improve its responses.', refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None))], created=1768862346, model='gpt-4.1-mini-2025-04-14', object='chat.completion', service_tier='default', system_fingerprint='fp_376a7ccef1', usage=CompletionUsage(completion_tokens=85, prompt_tokens=86, total_tokens=171, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    
    No tools needed....
    
    An LLM agent is an AI system that leverages a large language model to perform tasks by understanding and generating human-like text, often interacting with external tools or environments to complete complex goals. The ReAct framework enhances LLM agents by combining reasoning and acting in an interleaved manner, allowing the agent to reason about the problem, take actions based on that reasoning, observe outcomes, and iteratively improve its responses.

Congratulations!

This notebook gives you a true hello world for agentic AI:

- one simple tool (`get_current_time`)
- a clear two-step ReAct loop
- explicit code that shows how tool calls are handled
- an example where the tool is used
- an example where it is not used

In later lessons we can extend this pattern:

- Add a `load_csv(file_path)` tool to load a dataset into memory.
- Add a `summarize_data()` tool to report dtypes, missing values, and basic statistics.
- Add a `plot_data(...)` tool to create simple matplotlib visualizations.

The ReAct logic stays the same:

1. The model decides whether to use a tool.
2. Python runs the tool.
3. The tool result is added as a `tool` message.
4. The model reasons again with that new information.

This is the core idea of agentic AI that we will build on in the next notebooks.

## Check for Understanding

### Question 1



Choices:
- A. 
- B. 
- C. 
- D. 


<details>
<summary> View Answer </summary>
<strong>Answer:</strong>  <br>

</details>

### Question 2



Choices:
- A. 
- B. 
- C. 
- D. 


<details>
<summary> View Answer </summary>
<strong>Answer:</strong>  <br>

</details>
