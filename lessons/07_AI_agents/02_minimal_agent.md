# Minimal Tool Agent

As you've seen in the intro lesson, we will be looking at ReAct agents in our lessons. The following image illustrates the ReAct loop that these types of agents employ to generate responses to queries.
![React loop](./resources/react_loop.png)

The ReAct framework involves the reasoning step and the acting step. In the reasoning step, the LLM uses its internal reasoning capability to determine whether external tools are needed to answer the query. If it decides that an external tool is needed, the acting step is initiated wherein a tool call is output that requests the result of a given tool. The LLM can then use this result to generate the response. If a tool call is not necessary, it will directly generate the response. This loop can continue as long as needed till the agent determines the task to be completed.

To demonstrate how tool usage works in agentic frameworks, we will go through a simple example in this lesson. Before we get started, here are a few important things to note.
- LLMs cannot run code.
- LLMs can only ask you to run code (a tool).
- The tool is run externally and the result is sent back to the model.
- In case an external tool is needed, you need two API calls per query:
  - one where the model asks for a tool,
  - one where the model uses the tool result to answer.

We will go over these in greater detail as we go through our example implementation. We will also show, through another query, that the agent can choose not to use a tool, for example when answering a conceptual question.

## Implementation

We will implement an agent that is able to answer a simple question:
> "What time is it right now?" 

### Initial Setup

We will use OpenAI's agentic framework in this example. Therefore, we will first setup and connect to the OpenAI model server through the client. As with previous lessons, ensure your `.env` file contains the OpenAI API key.

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

Once the client is created successfully, the output should say the following:

    Successfully loaded environment variables from .env
    OpenAI client created.

### Baseline query

First, we ask the model for the current time without giving it any tools.

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
Recall that the model cannot run Python, has no real access to your system clock, it can only produce text. So it will have to guess or say outright that it cannot provide the answer. This is the kind of question where a tool would help.

The output should look something like this:

    I’m not able to access real-time information. Please check the current time on your device or through an online time service.

Now that we have established that the model cannot tell the current time without an external tool, let's look at the agentic framework that would enable the model to use a tool to tell the current time.

### Tool Definition

We will first define a simple Python function that returns the current local time. The agent will use this tool to answer our query. As mentioned earlier,this function is run by Python, not by the model. The model can only request that we run it.

```python
from datetime import datetime

def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

get_current_time()
```
The output should look like the following:

    '2026-01-19 17:30:38'

Next we will look at how to pass this tool to the model.

### Passing the tool description to the model

Before the model can call any Python function, we have to describe that function in a structured way so the model knows it exists and knows what arguments it should provide. Since the model can only request that a tool be run, it only sees tool definitions, a list of tool schema. We feed it one json schema for each item in the list in tools -- here there's just one. The model does not see your actual Python code directly. Instead, it only sees a *description* of the function written using a format based on the JSON Schema. The schema for each tool in the `tools` list consists of:
- `name`: the name of the function
- `description`: natural language description to help the model decide when to use it
- `parameters`: a specifications of the function's arguments (in this case there are no arguments)

When we add more complex functions with parameters, we will see more complexity in the tool list. The overall shape stays the same; only the details change. We will add more json-specified functions in later examples to tell the LLM what tools it has available. As of now, the tools list for our `get_current_time` tool looks like the following:

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

The output should look like the following:

    Tools list defined with one tool: get_current_time

The main takeaway here is that this dictionary is effectively a machine-readable description of your function. The LLM uses this description (not your Python code) to decide whether to call the tool and how to format the arguments. Hence, we should make sure that the description is as comprehensive and in-depth as possible to enable the LLM to reason in the most appropriate manner whether the tool will be useful for its task.

Optionally, we can also provide a system prompt to the model. The system prompt tells the model how to behave and use the tool. But this approach generally tends to use a lot of tokens, especially for multiple complex tools that will require extensive descriptions and usage instructions.

Note that we want the model to:
- Use the `get_current_time` tool only when needed.
- Answer directly when no tool is needed.
- Briefly state in the final answer whether it used a tool.
We will rely on the model's internal reasoning ability for this.

Now that we have the tool schema ready to pass to the model, we will now implement a method that allows the agent to use the ReAct loop from the image above to answer queries.

### Implementing the ReAct Agent

To emulate the ReAct loop for our agent, we develop the `run_agent` function shown below. It is helpful to view these steps in comparison with the image at the beginning of the lesson. The following steps are involved in the function:
1. You send a chat request (system + user messages + tools) to the API. You tell the model what tools are available using the `tools` parameter in the API call.
2. The model looks at the conversation and decides (the reasoning step in the image) to either:
   - answer directly, or
   - return a special `tool_calls` field. A `tool_call` is the model saying: "Please run this function and tell me the result."
3. Your Python code sees the `tool_call`, runs the local Python function, and gets a result. This is the acting step.
4. You send that result back to the model as a new message with `role: tool`.
5. You call the model again with the updated messages so it can see the tool result and give a final answer.

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

The `run_agent` method performs the following steps:
1. Builds the initial `messages` list.
2. Makes the first API call with `tools=tools` and `tool_choice='auto'.` When you set `tool_choice='auto'`, the model is not running any special tool-selection algorithm. It simply reads the conversation and the tool descriptions you provide (the name, description, and parameters fields) and uses its normal language-model reasoning to decide whether a tool would help answer the user’s question. This is why the tool description is so important!
3. Checks if there are any `tool_calls`. A structured `tool_calls` block is output instead of a normal answer if the model believes a tool is appropriate. This is just the model predicting text that follows the function-call schema it was trained to produce.
4. If there are tool calls:
   - runs the tool(s) in Python,
   - appends the tool results to `messages` (as `role: tool`),
   - makes a second API call to get the final answer.
5. If there are no tool calls:
   - returns the first response as the final answer.

In the case of tool usage, the API is called twice to respond to one query. Why does this happen? Let's investigate this further.

*First call*: When we call `client.chat.completions.create` with `tools=tools` and `tool_choice='auto'`, the model:
- reads the conversation
- decides whether it needs a tool
- returns a message that may include `tool_calls`

If `tool_calls` is present, that first response is not a final answer. It is the model saying:

> 'I am not done yet. Please run this tool and then come back to me with the result.'

In that case, the `run_agent` method:
- looks at `tool_calls`
- runs the corresponding Python function (`get_current_time`)
- builds a new message with `role: tool` that contains the tool result

*Second call*: We then call the API again with the updated `messages` list, which now includes:
- the original system + user messages
- the model's first response (where it asked for the tool)
- the `tool` message with the tool result

Only now does the model have all the information it needs to give a final answer.

So to summarize, when the model deems the usage of a tool necessary the API call pattern is:
1. First call: model may request tools
2. Python: runs tools and adds tool outputs as messages
3. Second call: model sees tool outputs and gives final answer

Another important thing to note is that you can also include `tools=tools` and `tool_choice='auto'` in the second API call and see the result to check whether the model would like further tool calls. Here we're merely emulating a single ReAct loop since the `get_current_time` function needs to be run just once to get the current time.

We will now test this agent one two queries, one that will require the use of the tool (the `get_current_time` function) and one that will not.

### Agent Test 1: Query requiring tool use

Now we use `run_agent` on the original question we wanted the agent to answer:

> 'What time is it right now?'

This time, the model should:
- decide to call `get_current_time`, and
- use the real tool result in its final answer
<!-- - briefly mention that it used a tool. -->

```python
answer_with_agent = run_agent('What time is it right now?')
print(answer_with_agent)
```
Since we have asked for the API responses to be printed, we can also observe the structure of the messages when the tool use is requested. The output should look like the following:

    First response received from model...
    ChatCompletion(id='chatcmpl-CzsBdnd9VNr7XiA1RAW18iFmnqMra', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_y2pzEhksbc4YKDDIDW4iGPjx', function=Function(arguments='{}', name='get_current_time'), type='function')]))], created=1768862337, model='gpt-4.1-mini-2025-04-14', object='chat.completion', service_tier='default', system_fingerprint='fp_376a7ccef1', usage=CompletionUsage(completion_tokens=11, prompt_tokens=75, total_tokens=86, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    
    Agentic mode engaged...
    
    Tool called: get_current_time
    Tool result: 2026-01-19 17:38:58
    
    Second response received from model...
    ChatCompletion(id='chatcmpl-CzsBfDc7VB5g7WnzFbCnrahvFYN7q', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='The current time is 17:38:58.', refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None))], created=1768862339, model='gpt-4.1-mini-2025-04-14', object='chat.completion', service_tier='default', system_fingerprint='fp_376a7ccef1', usage=CompletionUsage(completion_tokens=11, prompt_tokens=77, total_tokens=88, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    
    The current time is 17:38:58.

In the first message, you can see the `choices` parameter as a `Choice` object with `finish_reason='tool_calls'`, and `tool_calls=[ChatCompletionMessageFunctionToolCall(function=Function(arguments='{}', name='get_current_time'), type='function')]))` and `content=None`. This signifies that the model has stopped the response generation and wants to call the function `get_current_time`. This engages "Agentic mode" and the function is called. Once the previous messages and the function output is fed to the model, the second response has the `Choice` object with `finish_reason='stop'` and `content='The current time is 17:38:58.'`. This content is the output response from the agent. 

### Agent Test 2: Query not requiring tool use

Now that we've seen how the agentic framework responds to a query with tool use, we test the agent with a conceptual question that does not require the current time function, such as:

> 'Explain what an LLM agent is (including the ReAct framework), in two sentences.'

The model should:
- answer directly, and
- not call the tool
<!-- - briefly mention that it did not use a tool. -->

```python
answer_conceptual = run_agent(
    'Explain what an LLM agent is (including the ReAct framework), in two sentences.'
)
print(answer_conceptual)
```
The output should look like the following:

    First response received from model...
    ChatCompletion(id='chatcmpl-CzsBm10zUIBNDlRTScZUf7MnGyX0c', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='An LLM agent is an AI system that leverages a large language model to perform tasks by understanding and generating human-like text, often interacting with external tools or environments to complete complex goals. The ReAct framework enhances LLM agents by combining reasoning and acting in an interleaved manner, allowing the agent to reason about the problem, take actions based on that reasoning, observe outcomes, and iteratively improve its responses.', refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None))], created=1768862346, model='gpt-4.1-mini-2025-04-14', object='chat.completion', service_tier='default', system_fingerprint='fp_376a7ccef1', usage=CompletionUsage(completion_tokens=85, prompt_tokens=86, total_tokens=171, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    
    No tools needed....
    
    An LLM agent is an AI system that leverages a large language model to perform tasks by understanding and generating human-like text, often interacting with external tools or environments to complete complex goals. The ReAct framework enhances LLM agents by combining reasoning and acting in an interleaved manner, allowing the agent to reason about the problem, take actions based on that reasoning, observe outcomes, and iteratively improve its responses.

Now the first message directly outputs the `Choice` object with `finish_reason='stop'` and `content` that is the conceptual response to the query. Most importantly, there are no `tool_calls`. So the "Agentic mode" is not engaged and the first message content is directly output as the response. 

Congratulations! You've gotten your first taste of the agentic AI framework.
This lesson gives you a true hello world for agentic AI. We have:
- one simple tool (`get_current_time`)
- a clear two-step ReAct loop
- explicit code that shows how tool calls are handled
- an example where the tool is used
- an example where it is not used

It is important to note that even a simple task like telling the time is outside the capabilities of an LLM. It is only capable of generating a response to a query based on its training and internal parameters. This is why the ability to leverage an external tool (in addition to external data sources using RAG) represents such a huge leap in the efficacy of agents.

In the next lesson, we will extend this pattern. We will work with multiple tools with arguments that can load a csv file, and summarize and plot its data.

<!-- - Add a `load_csv(file_path)` tool to load a dataset into memory.
- Add a `summarize_data()` tool to report dtypes, missing values, and basic statistics.
- Add a `plot_data(...)` tool to create simple matplotlib visualizations. -->

The ReAct logic will stay the same:
1. The model decides whether to use a tool.
2. Python runs the tool.
3. The tool result is added as a `tool` message.
4. The model reasons again with that new information.

This is the core idea of tool-based agentic AI that we will build on in the next lesson.

## Check for Understanding

### Question 1

In what format is information about the available tools passed on to the agent in this lesson? 

Choices:
- A. CSV
- B. Python
- C. JSON
- D. Text

<details>
<summary> View Answer </summary>
<strong>Answer: C. JSON </strong>  <br>
The information about available tools (name, description, parameters) is passed in the JSON format to the OpenAI client. 
</details>

### Question 2

How does the agent know whether a tool must be used or not to answer a given query?

Choices:
- A. Tool descriptions 
- B. The agent uses its own internal reasoning
- C. Both A & B
- D. It chooses randomly

<details>
<summary> View Answer </summary>
<strong>Answer: C. Both A & B</strong>  <br>
In general the agent will use its own internal reasoning to determine whether any of the provided tools need to be used to answer the given query. However, a big factor in this decision is the tool descriptions in the JSON schema.
</details>
