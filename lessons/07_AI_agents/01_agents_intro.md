# Introduction to AI Agents
AI agents are autonomous systems that can use external tools to achieve specific goals. A helpful way to think about the difference between an agent and a traditional LLM is that a basic LLM answers questions, while an agent is able to perform actions using *tools*. This is why agentic AI has become so popular recently: it breaks LLMs out of the the "text-in, text-out" box, and let's them interact with the (software) world in much more interesting ways. 

## AI Agents vs. Standard LLMs
In standard LLM interaction, you provide a prompt, and the model produces a text response. As we have seen, LLMs are not inherently capable of using external tools like running Python functions: they are trained only to generate text.  Agents, by contrast, are LLMs that are provided with the power to harness external tools (such as Python functions such as web scrapers or data analysis back ends) to reach goals provided by the user. 

There are many different frameworks for thinking about and building agents, but the most common is called the ReAct framework, which stands for "Reasoning + Acting".

ReAct interleaves reasoning and action in a structured loop: the LLM is given an overarching task and a set of tools, and then is let loose to reason about the problem. It first decides whether it needs additional tools to accomplish the task ("Reasoning"). If not, it responds to the query directly. If it does need tools, it selects an appropriate tool and provides the necessary input. The tool is executed, and the result is fed back to the LLM ("Acting"). The LLM will then *observe* and use this new information to inform its next reasoning step. The agent continues with this cycle of reasoning, acting, and observing until it reaches a final answer.

This ReAct loop is illustrated in the following diagram, which was made by CTD instructor Roshan Suresh Kumar:

![React framework gif](resources/react_loop.png)

If you would like to learn more about the ReAct framework, there are many resources online, such as [this article](https://blog.dailydoseofds.com/p/intro-to-react-reasoning-and-action).

ReAct is not a single library or tool: it is a *design pattern* that builds on the inherent reasoning abilities of LLMs

This pattern helps LLMs escape from brittle, one-shot responses. Instead of trying to solve everything at once, the LLM can think step-by-step, verify intermediate results, and adjust its plan as needed. In theory, this makes agents more reliable and easier to debug, because you can see not just the final answer, but the sequence of thoughts and actions that led there.

### Tool-Based and Code-Based Agents
Not all agents work the same way. One distinction that is becoming more prevalent is between "tool-based" agents and "code-based" agents.

*Tool-based agents* interact with the world by calling predefined tools that are explicitly enumerated to the LLM. These tools might include things like a plotting function or a calculator. The agent's job is to decide which tools to use and when, given the task at hand. This approach is generally safer and easier to control, because the agent can only do what the available tools allow.

*Code-based agents* take this idea one step further. Instead of selecting from a fixed list of tools, the agent is allowed to write and execute code to reach its goal. The agent writes code, runs it, inspects the results, and then decides what to do next. This is extremely powerful and flexible. While careful sandboxing and guardrails are required to keep things safe, in practice LLMs are surprisingly good at writing correct code, which makes code-based agents very effective. We will see an example of this in a hands-on lesson.

Conceptually, however, these two approaches are not as different as they may first appear. From the perspective of the ReAct design pattern, writing and running code is just another kind of tool. A code-based agent is simply an agent whose primary tool is a code execution environment, and whose observations come from the results of running that code.

In practice, many real systems blend both approaches. An agent might use predefined tools for common tasks, while falling back on code execution for more open-ended or complex problems. Understanding this distinction helps explain why some agents feel constrained, while others feel almost like autonomous programmers -- even though they are all built on the same underlying reasoning-and-acting loop.

### Frameworks for Building Agents
Building an agent from scratch is a useful way to build intuition for how they work, and we will build a couple in order to demystify agents and LLM tool use. However, just like with RAG, things can get complex very quickly, and there are [many agentic frameworks](https://github.com/Azure-Samples/python-ai-agent-frameworks-demos/) that have been created to handle this complexity for you. Just to name a few popular ones:

**[smolagents](https://huggingface.co/docs/smolagents/en/index)** is a lightweight framework from HuggingFace that emphasizes simplicity. It is especially well-suited for Code the Dream, because it it is a simple, Pythonic, and flexible framework. Also, their code-based agents are extremely powerful and almost magical in their ability to write and execute code. 

<img src="resources/langchain.jpg" alt="langchain logo" width="300" style="float: right;"/>

**[LangChain](https://www.langchain.com/agents)** is a general-purpose framework for building LLM-powered applications, including agents. LangChain provides many tools for building production-grade systems quickly. The trade-off is complexity of framework: we initially planned to use LangChain for Code the Dream, but we found smolagents more suitable for a one-week lesson. 

**[LlamaIndex](https://developers.llamaindex.ai/python/framework/understanding/agent/)**: while LlamaIndex is best known for RAG applications, it also can be used to build agents. We will show how in the assignments this week. 

All three frameworks solve the same core problem: turning LLMs into systems that can act with the help of external tools.  

### Abstractions and interfaces for agents
As agents become more common, an important practical question arises: how do agents actually connect to tools, data sources, and external systems in a consistent way? Most LLM frameworks have their own methods for interacting with tools, which can lead to fragmentation and compatibility issues. To deal with this, there are a few possible solutions. 

One, just like there exist abstraction layers for LLMs that let you interface with multiple LLM providers in a consistent way, there are abstraction layers for agents that let you interface with multiple agent frameworks in a consistent way, such as [any-agent](https://github.com/mozilla-ai/any-agent) from Mozilla. 

<img src="resources/mcp.jpg" alt="mcp logo" width="250" style="float: right;"/>

Also, the [Model Context Protocol](https://en.wikipedia.org/wiki/Model_Context_Protocol) (MCP), created by Anthropic, is an emerging standard that defines a *common interface* for exposing tools and resources to agents. It is a protocol that standardizes how agents discover tools, understands their inputs and outputs, and calls them safely, much like a USB standardizes how devices connect to computers. It involves setting up an MCP server that exposes tools in a structured way, and then agents can connect to that server to access the tools. MCP has the potential to greatly simplify the development of agentic systems in the future.

## Next steps
AI agents represent a shift from "text exchangers" to "tool users". The core idea of taking a goal and iterating with tools until it is reached -- is simple, but can lead to extremely powerful results. In the rest of the lessons this week, we will move from these concepts to hands-on examples, where you will see how agents behave in practice.