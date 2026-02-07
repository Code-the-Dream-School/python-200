# Smolagents



## Implementation

### Setting up

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

Output:

    Successfully loaded environment variables from .env

### Define smolagents tools

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

```python
csv_manager = CsvManager(resources_dir=RESOURCES_DIR)
```

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

### Create and test tool calling agent

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

*Test 1:*

```python
tool_agent.run("List the csv files in resources")
```

Output:

![toolagent test1](./resources/lesson4_toolagent_test1.png)

*Test 2:*

```python
prompt = """Load bike_commute.csv, and plot distance traveled (x)
            versus duration (y) as a scatter. Make the points red."""

tool_agent.run(prompt)
```

Output:

![toolagent test2a](./resources/lesson4_toolagent_test2_step2a.png)

![toolagent test2b](./resources/lesson4_toolagent_test2_step2b.png)

![toolagent test2c](./resources/lesson4_toolagent_test2_step3.png)

### Create and test code agent

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

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)
```

*Test 1:*

```python
code_agent.run("List the available csv files please", 
               additional_args={"csv_manager": csv_manager})
```

Output:

![codeagent test1a](./resources/lesson4_codeagent_test1_step1.png)

![codeagent test1b](./resources/lesson4_codeagent_test1_step2.png)

*Test 2:*

```python
prompt1 = """Load bike_commute.csv, and plot distance traveled (x)
            versus duration (y) as a scatter. Make the points red."""
code_agent.run(prompt1,
    additional_args={"csv_manager": csv_manager},
)
```

![codeagent test2 step1](./resources/lesson4_codeagent_test2_step1.png)

![codeagent test2 step2](./resources/lesson4_codeagent_test2_step2.png)

![codeagent test2 step2b](./resources/lesson4_codeagent_test2_step2b.png)

![codeagent test2 step3](./resources/lesson4_codeagent_test2_step3.png)

*Test 3:*

```python
prompt2 = """Load bike_commute.csv, and find the correlation between 
            distance traveled and duration (y)."""
code_agent.run(prompt2, 
    additional_args={"csv_manager": csv_manager},
)
```

![codeagent test3 step1](./resources/lesson4_codeagent_test3_step1.png)

![codeagent test3 step2n3](./resources/lesson4_codeagent_test3_step2n3.png)

### Create and test Code-based Chat Agent

```python
chat_model = OpenAIServerModel(
    api_key=os.environ["OPENAI_API_KEY"],
    model_id="gpt-4o",
)
chat_agent = CodeAgent(
    tools=TOOLS,
    model=chat_model, # changed from frontier model
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=10,
)
```

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

Output:

    CSV analysis agent at your service. Here to help look at your CSV data!
    Type a question. Type 'exit' to quit.

    To start, try 'list csv files' or 'load bike_commute.csv'

*Query 1*

    User: list csv files

![chatagent q1](./resources/lesson4_chatagent_q1.png)

*Query 2:*

    User: load the csv file

![chatagent q2](./resources/lesson4_chatagent_q2.png)

*Query 3:*

    User: plot the average speed against the distance using red dots

![chatagent q3a](./resources/lesson4_chatagent_q3a.png)

![chatagent q3b](./resources/lesson4_chatagent_q3b.png)

![chatagent q3c](./resources/lesson4_chatagent_q3c.png)

![chatagent q3d](./resources/lesson4_chatagent_q3d.png)

*Query 4:*

    User: How are the average speed and distance related to each other?

![chatagent q4](./resources/lesson4_chatagent_q4.png)

*Query 5:*

    User: What does the value of the correlation coefficient mean?

![chatagent q5](./resources/lesson4_chatagent_q5.png)

Congratulations!

Maybe mention some future directions:
- Debugging and tracking agents with [open telemetry](https://huggingface.co/docs/smolagents/v1.5.0/en/tutorials/inspect_runs) (Pheonix)
- [Multi-agent systems](https://huggingface.co/learn/agents-course/en/unit2/smolagents/multi_agent_systems)


## Check for Understanding

### Question 1


Choices:
- A. 
- B. 
- C. 
- D. 

<details>
<summary> View Answer </summary>
<strong>Answer: </strong>  <br>
 
</details>

### Question 2



Choices:
- A. 
- B. 
- C. 
- D. 

<details>
<summary> View Answer </summary>
<strong>Answer: </strong>  <br>

</details>