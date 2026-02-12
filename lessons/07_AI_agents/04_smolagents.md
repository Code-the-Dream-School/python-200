# Code-based Agents and Smolagents

In the previous lessons, we saw how tool-based agents use the ReAct loop (shown below) to determine when to employ a subset of certain external tools to generate the appropriate response to a user query. There are two goals for this lesson: 
- Introduce code-based agents as a powerful alternative to tool-based agents that allow for the agent to develop and utilize its own tools in case the existing tools are incapable of performing the required tasks to answer the user query. 
- Introduce Huggingface's [smolagents](https://huggingface.co/blog/smolagents) as a flexible and user-friendly library for the creation of both tool-based and code-based agents. Just like LlamaIndex was used in [last week's lesson](../06_AI_augmentation/03_framework_rag_llamaindex_svs.md) to develop RAG frameworks, smolagents is used to develop agentic frameworks. 

<!-- >> **Note:** LlamaIndex has its own packages to develop agentic frameworks. You can find more information [here](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/) and [here](https://developers.llamaindex.ai/python/framework/understanding/agent/). -->

![react loop](./resources/react_loop.png)

Code-based agents use the same ReAct loop as tool-based agents, but can be thought of as having access to an additional coding and compiling tool that can create more tools as needed! In addition to using the available external tools, the LLM can reason internally to request the creation of a new tool (within a certain scope) to obtain the result it needs to generate the response. In reality, the LLM has been trained to generate the required code and has access to a python interpreter to compile the code. While this increases the risk of erroneous/malicious code being generated, the agent now has greater agency in obtaining the necessary information to answer the query in a better way. [This Huggingface article](https://huggingface.co/docs/smolagents/v1.24.0/en/tutorials/secure_code_execution#code-agents) discusses this in greater detail. 

In this lesson, we will leverage smolagents in the development of both tool-based and code-based agents on the data analysis example from the previous lesson. More references and sources can be found here: [Huggingface article](https://huggingface.co/docs/smolagents/en/index), [Medium article](https://kargarisaac.medium.com/exploring-the-smolagents-library-a-deep-dive-into-multistepagent-codeagent-and-toolcallingagent-03482a6ea18c), [Youtube video](https://www.youtube.com/watch?v=dSGS6-iGhyo).

## Implementation

### Setting up

In this implementation, Smolagents will utilize an OpenAI model as its internal LLM. You can install smolagents in your virtual environment through pip.

    pip install smolagents

The following code snippet loads the OpenAI API Key from the `.env` file which will be used later. The requisite smolagents imports are also handled here.

```python
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

RESOURCES_DIR = Path("resources")
```

Upon successful imports and reading of the OpenAI API Key, the output should look like the following:

    Successfully loaded environment variables from .env

Next, we will define the set of tools that both the tool-based and code-based agents will have access to.

### Define smolagents tools

Since we will be using the same data analysis example as the previous lesson, the `CsvManager` class with its internal tools defined below are the same.

```python
class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.

        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.

        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."

        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"

        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None

        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."

        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."
```

To define the tools list that will be fed to the agents, we first instantiate the `CsvManager` class.

```python
csv_manager = CsvManager(resources_dir=RESOURCES_DIR)
```

Then, we wrap the `CsvManager` methods into smolagents tools. The beauty of smolagents is that we can simply use the `@tool` decorator to signify the method is a tool for the agent to use, instead of creating the entire JSON schema like we did for the previous lessons. 

```python
@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.

    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".

    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> list[str] | dict:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list[str] | None = None) -> dict:
    """Return summary stats for selected columns (or all columns). 
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.

    Args:
        columns: Column names to summarize. If None, summarizes all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str | None = None, plot_type: str = "line") -> str | dict:
    """Plot from the active CSV.

    Args:
        y: Column name to plot on the y-axis. 
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.

    Returns:
        Generates and shows the plot. 
        Retirms a short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)
```

There are a few constraints on the tool definitions that must be followed:
- The function name must be descriptive enough for the LLM to understand its use
- Type hints must be present at both the input and output level for proper use (for example: `load_csv(filename: str) -> dict`)
- Google-style docstrings (the text between the """) must be used to describe the goal, arguments, and returned objects in natural language of the tool

You can learn more about these constraints [here](https://huggingface.co/learn/agents-course/en/unit2/smolagents/tools).

Finally, we define the tools list for the agents.

```python
TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
]
```

Next, we instantiate our smolagents tool-based agent.

### Create and test tool calling agent

We can instantiate the tool-based agent (called `ToolCallingAgent` in smolagents) with a given model and a system prompt. We will use a gpt-4o-mini instance as our internal LLM (called using smolagents' `OpenAIServerModel` class with our previously loaded API Key) with the system prompt informing it to help analyze the data in the resources drive.

```python
model_to_use = "gpt-4o-mini"  # default model ID
model = OpenAIServerModel(
    api_key=api_key,
    model_id=model_to_use,
)

SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)

tool_agent = ToolCallingAgent(tools=TOOLS,
                         model=model,
                         instructions=SYSTEM_PROMPT,)
```

It is important to note that all the work of instantiating our tool-calling agent from the previous lessons is reduced to a single line of code! 

In addition to the `tools`, `model`, and `instructions` parameters, you can define the `max_steps` parameter as the maximum number of ReAct loop steps the agent can take to answer a query. By default, this value is 6. 

Now let's test this agent on two prompts to see how it performs.

> **Note:** All smolagents agent classes have an integer `verbosity_level` parameter that controls the verbosity of the agent's output. By default, this value is 1 signifying minimal output. You can set it to 0 to remove all outputs or increase it for more detailed outputs on the agent's "thought process." Generally, higher verbosity is useful for development/debugging.

*Test 1:*

To generate a response to a query, you have to simply call the agent's `run` method. Here, we want the agent to list out the available CSV files in the resources directory.

```python
tool_agent.run("List the csv files in resources")
```

The following screenshot shows the agent's response. Note that you may get slightly different outputs.

![toolagent test1](./resources/lesson4_toolagent_test1.png)

You can see that, in the minimal verbosity setting the agent describes its actions and observations, along with the duration and number of input and output tokens for the particular step. The agent takes two steps, using the `list_csv_files` method to answer the query correctly.

Next, let's test the agent's performance on a query that cannot be answered using the available tools.

*Test 2:*

We will prompt the agent to load the CSV file and plot the distance and duration with red points. The available tools allow the agent to perform all the tasks except change the color of the points. 

```python
prompt = """Load bike_commute.csv, and plot distance traveled (x)
            versus duration (y) as a scatter. Make the points red."""

tool_agent.run(prompt)
```

The following screenshots show the verbose outputs of the agent. Note that you may get slightly different outputs.

![toolagent test2-1](./resources/lesson4_toolagent_test2_step1.png)

![toolagent test2-2a](./resources/lesson4_toolagent_test2_step2a.png)

![toolagent test2-2b](./resources/lesson4_toolagent_test2_step2b.png)

![toolagent test2-3](./resources/lesson4_toolagent_test2_step3.png)

As you can see, the agent is able to load the `bike_commute.csv` file correctly at the end of Step 1. The agent calls the `plot_data` tool and plots the correct variables, but is not able to change the color of the points since its not part of the capabilities of the `plot_data` tool. Yet, tellingly the agent believes that it created the plot correctly and does not acknowledge its inability to change the color of the points. This is a clear case of hullinication and a major limitation of tool-based agents.

We will now look at code-based agents that can address this issue by creating and employing new tools if needed. You can also have them modify your existing tools to add new capabilities but this is risky since the edits made by the agent may make the tool unusable. Hence, it is generally better to let the agent create its own tools if needed. 

### Create and test code agent

Code-based agents represent a significant shift from tool-based agents in the ability to respond to queries. The agent is no longer limited to just the tools you provide, but can also create its own tools to perform the task it deems necessary to answer the query. This is like having a junior developer in your team that can code the tools needed in real-time. 

*Coding Prompt:*

```python
CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
  that the plot_data tool cannot control, DO NOT call plot_data.
  Instead, write matplotlib code directly so the plot matches the request.
  If code execution fails, do not fall back to plot_data when the user requested styling (like color). 
  Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""
```

Since the code agent can create its own tools now, the system prompt is a lot more comprehensive to define the scope of the actions the agent can take. Here, we explicitly state that the agent should prefer the available tools wherever possible and to write new matplotlib code in case plot styling is needed.

*Initializing the Agent:*

```python
code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)
```
Just like the `ToolCallingAgent`, the code-based (`CodeAgent`) agent can also be initialized with a single line of code. We provide the model, tools list, system prompt, and the additional set of authorized imports (libraries that the agent can import, here `matplotlib`, `pandas`, `numpy` to keep it simple). Specifying the additional imports is not strictly necessary but helps to constrain the agent's scope for creating new tools, thereby reducing the chances of hallucination. We also specify the maximum number of steps in the ReAct loop as 8.

As with the tool-based agent, we will test the code-based agent on the two queries from before. We also pass the `csv_manager` as an additional argument as the code agent can make use of it.

*Test 1:*

We first prompt the agent to list the available CSV files.

```python
code_agent.run("List the available csv files please", 
               additional_args={"csv_manager": csv_manager})
```

The following screenshots capture the outputs of the agent. Note that you may get slightly different outputs.

![codeagent test1a](./resources/lesson4_codeagent_test1_step1.png)

![codeagent test1b](./resources/lesson4_codeagent_test1_step2.png)

As with the tool-based agent, the code-based agent uses the `list_csv_files` as is expected to output the `bike_commute.csv` file. 

*Test 2:*

Now, we will test the code-based agent on the prompt that the tool-based agent could not answer correctly.

```python
prompt1 = """Load bike_commute.csv, and plot distance traveled (x)
            versus duration (y) as a scatter. Make the points red."""
code_agent.run(prompt1,
    additional_args={"csv_manager": csv_manager},
)
```

The following screenshots capture the set of steps the agent takes to answer the query. Note that you may get slightly different outputs.

![codeagent test2 step1](./resources/lesson4_codeagent_test2_step1.png)

![codeagent test2 step2](./resources/lesson4_codeagent_test2_step2.png)

![codeagent test2 step2b](./resources/lesson4_codeagent_test2_step2b.png)

![codeagent test2 step3](./resources/lesson4_codeagent_test2_step3.png)

Just like the tool calling agent, the agent is able to use the `load_csv` tool to load the `bike_commute.csv` data in Step 1. Step 2 shows the major departure from the tool calling agent. This time the code agent creates its own code to create the scatter plot with red data points. Note that the agent accesses the dataframe through the `csv_manager` object passes as an additional argument. The plot created is the expected output for the prompt and the agent could have stopped there. But interestingly, the agent goes through another ReAct loop to create new code to save the created figure as well. This was not required and shows that the code-based agent is not infallible, even with all its capabilities.

<!-- WE CAN INCLUDE THIS TEST AS WELL, BUT I THINK THIS MAKES THE LESSON TOO LONG. COMMENTING THIS SECTION FOR NOW

The previous query tested the ability of the code agent to extend the plotting capabilities of the available `plot_data` tool. We will test the code agent on a query that requires correlation determination, a capability that is not related to any of the available tools.

*Test 3:*

We will now prompt the agent to load the `bike_commute.csv` data and find the correlation between the distance traveled and the duration.

```python
prompt2 = """Load bike_commute.csv, and find the correlation between 
            distance traveled and duration (y)."""
code_agent.run(prompt2, 
    additional_args={"csv_manager": csv_manager},
)
```

The following snapshots capture the different steps the agent takes to answer the query. Note that you may get slightly different outputs.

![codeagent test3 step1](./resources/lesson4_codeagent_test3_step1.png)

![codeagent test3 step2n3](./resources/lesson4_codeagent_test3_step2n3.png)

Just like before, the agent loads the CSV file using the available tool in Step 1. In Step 2, the agent generates a simple two line script that determines the correlation between the distance and duration using the pandas `corr` method. The output of Step 2 is the Pearson correlation coefficient (computed as default using the `corr` method) between the distance and duration values. For context, the Pearson correlation coefficient measures the degree of linear correlation between two variables. However interestingly, no output is generated at the end of Step 2 which necessitates another step to generate the final output. -->

So far we have seen tool-based and code-based agents respond to a single query at a time. Next we will create a chatbot that leverages the code-based agent.

### Create and test Code-based Chat Agent

As before, we initialize the code-based agent to be used in our chatbot. This time we will use a gpt-4o instance as our internal LLM for the code agent.

```python
chat_model = OpenAIServerModel(
    api_key=os.environ["OPENAI_API_KEY"],
    model_id="gpt-4o",
)
chat_agent = CodeAgent(
    tools=TOOLS,
    model=chat_model, 
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=10,
)
```

We will use a simple while loop to simulate the agent's chat with a user. For the agent to have context of the previous conversations, we simple set `reset=False` for the `run` method of the agent. This makes sure that the agent state is not reset after a query has been answered.

```python
print("CSV analysis agent at your service. Here to help look at your CSV data!")
print("Type a question. Type 'exit' to quit.\n")
print("To start, try 'list csv files' or 'load bike_commute.csv'\n")

# while True:
k = 0
while k < 5:
    user_msg = input("You: ")
    if user_msg.strip().lower() in {"quit", "exit"}:
        print("Exiting chat.")
        break
    
    print(f"User: {user_msg}")
    response = chat_agent.run(user_msg,  
                                  additional_args={"csv_manager": csv_manager}, 
                                  reset=False)
    print(f"Agent: {response}")
    k += 1
```

We will limit the number of conversations to 5 to reduce token usage. The following outputs and screenshots are organized into discussions on each individual query in the conversation to enable better understanding. Note that you may get slightly different outputs. The first output is the set of print statements in the first three lines of code in the above code section.

    CSV analysis agent at your service. Here to help look at your CSV data!
    Type a question. Type 'exit' to quit.
    To start, try 'list csv files' or 'load bike_commute.csv'

*Query 1*

    User: list csv files

![chatagent q1](./resources/lesson4_chatagent_q1.png)

Like we have seen in the previous section, the agent is able to list the `bike_commute.csv` file using the `list_csv_files` tool after the first step. Next, we load the data in this file.

*Query 2:*

    User: load the csv file

![chatagent q2](./resources/lesson4_chatagent_q2.png)

Similar to earlier outputs, the agent is able to load the data in the `bike_commute.csv` file using the `load_csv` tool. Next, we analyze the data using the agent.

*Query 3:*

    User: plot the average speed against the distance using red dots

![chatagent q3a](./resources/lesson4_chatagent_q3a.png)

![chatagent q3b](./resources/lesson4_chatagent_q3b.png)

![chatagent q3c](./resources/lesson4_chatagent_q3c.png)

![chatagent q3d](./resources/lesson4_chatagent_q3d.png)

Just like before, the agent is able to generate the appropriately plot within two steps. Interestingly, it creates erroneous code in the beginning. The compiling error is shown in red, indicating that the column names are incorrect. Note that since we explicitly stated in the system prompt that the agent should not revert back to the available tools in case of coding errors. The agent follows this instruction to create the correct code and generate the correct prompt in the next step. Next, we ask the agent to determine the correlation between the speed and distance but with a more general query.

*Query 4:*

    User: How are the average speed and distance related to each other?

![chatagent q4](./resources/lesson4_chatagent_q4.png)

Note that the query is very general and can be answered either conceptually based on the previous plot/data or numerically by computing a correlation coefficient. The agent chooses to do the latter, generating a two line code using the `corrcoef` method from `numpy` to compute the Pearson correlation coefficient (degree of linear correlation) between the distance and speed. Lastly, we will ask a more conceptual query to test the ability of the code agent to rely purely on its internal LLM reasoning. 

*Query 5:*

    User: What does the value of the correlation coefficient mean?

![chatagent q5](./resources/lesson4_chatagent_q5.png)

When queried for the meaning of the particular correlation coefficient value, the agent is able to answer after just one step that the value of 0.13 implies a weak positive relationship between the distance and average speed. This is correct and consistent with the data. Additionally, the agent did not have to use or create any external tools to answer the query. You will also notice that the step numbers do not reset after each query is answered, implying that the agent is aware of the previous conversations.

Congratulations! You have explored smolagents and its ability to create tool-based and code-based agents easily with minimal lines of code in a hands-on manner. Smolagents also provides handy tool handling capabilities that allows us to bypass the tedious process of creating the JSON schema we used in the previous lessons. Code agents specifically represent a huge leap in the capabilities of an agent to not only use but create tools that it needs to perform a given task and answer the query effectively. Combine these agents with RAG frameworks that allow the leveraging of relevant information from external data sources and you have the building blocks of the modern day chatbot created and deployed by many SaaS (Software-as-a-Service) companies.

It is worth mentioning that the content covered in this lesson merely scratches the surface of agents and agentic frameworks. There are many more concepts that can be covered here such as:
- Debugging and tracking agent runs with [Open Telemetry](https://huggingface.co/docs/smolagents/v1.5.0/en/tutorials/inspect_runs) and [Pheonix](https://github.com/Arize-ai/phoenix)
- [Multi-agent systems](https://huggingface.co/learn/agents-course/en/unit2/smolagents/multi_agent_systems)

You are highly encouraged to explore these topics at your convenience, all of which are covered in the huggingface [online course](https://huggingface.co/learn/agents-course/en/unit0/introduction) on agents.

In the next lesson we will look at a demo of Github Copilot, a code agent that can modify your code based on the instructions you provide, on a small ETL (Extract-Transform-Load) repository. This represents the next step in the ability of code agents to modify existing tools to debug and improve their functionality as opposed to creating new tools as needed.

## Check for Understanding

### Question 1

Tool-based agents can modify the available tools to improve their working capabilities, if needed. True or False?

Choices:
- A. True
- B. False

<details>
<summary> View Answer </summary>
<strong>Answer: B. False </strong>  <br>
 As seen in the response of the tool-based agent to the second test query, the agent is not able to add the capability of modifying the marker style of the plotting tool. It can only use the tool when needed.
</details>

### Question 2

When using the `CodeAgent`, what is a critical security feature provided by smolagents that prevents the agent from running malicious commands on your system?

Choices:
- A. It requires the user to manually type "Allow" every time a code script is generated. 
- B. It uses a restricted Python interpreter that limits access to dangerous functions and imports.
- C. It encrypts the Python code before sending it to the LLM
- D. It only allows the agent to write commands, not actual code

<details>
<summary> View Answer </summary>
<strong>Answer: B. It uses a restricted Python interpreter that limits access to dangerous functions and imports. </strong>  <br>
The smolagents "CodeAgent" is equipped with a secured, sandboxed execution environment which carefully manages what the agent is allowed to do to keep the host system safe.
</details>