# Introduction to AI Agents
AI agents are autonomous systems that can perform tasks on behalf of users by leveraging a combination of external tools, and decision-making processes. A helpful way to think about the difference between an agent and a traditional LLM is that a basic LLM answers questions, while an agent is able to perform actions using *tools*. This is why agentic AI has become so popular recently: it breaks LLMs out of the the "text-in, text-out" box, and let's them interact with the (software) world in much more interesting ways. 

## AI agents vs. standard LLMs
A standard LLM interaction is reactive: you provide a prompt, and the model produces a text response. As we have seen, LLMs are not inherently capable of taking actions or interacting with the world, they are trained only to generate text.  Agents, by contrast, are LLMs that are provided with the power to harness external tools (such as Python functions) to reach goals provided by the user. 

There are many different frameworks for thinking about and building agents, but the most common is called the ReAct framework, which stands for "Reason + Act".

ReAct interleaves reasoning and action in a structured loop: the LLM is given an overarching task and a set of tools, and then is set loose to reason about the problem, decide on an action using the tools, observe the result, and repeats until the goal is achieved. This is illustrated in the following figure from an article about the [ReAct framework](https://blog.dailydoseofds.com/p/intro-to-react-reasoning-and-action): 

![React framework gif](resources/react_agents.gif)

This pattern helps agents avoid brittle, one-shot responses. Instead of trying to solve everything at once, the agent can think step-by-step, verify intermediate results, and adjust its plan as needed. In theory, this makes agents more reliable and easier to debug, because you can see not just the final answer, but the sequence of thoughts and actions that led there.

ReAct is not a single library or tool: it is a design pattern that includes LLMs in the loop. Many agent frameworks employ some variation of this loop under the hood.

### Frameworks: smolagents, LangChain, and LlamaIndex
While building an agent from scratch is possible (indeed, we will build a couple in order to demysify agents and LLM tool use). However, just like with RAG, things can get complex very quickly, and there are [many agentic frameworks](https://github.com/Azure-Samples/python-ai-agent-frameworks-demos/) that have been created to handle this complexity for you. Just to name a few:

**[smolagents](https://huggingface.co/docs/smolagents/en/index)** is a lightweight framework from HuggingFace that emphasizes simplicity and transparency. It is especially well-suited for educational settings, because the agent loop is explicit and easy to inspect. Code-based agents are a first-class concept, which makes smolagents a good fit for learning how agents actually work under the hood. smolagents is the framework we will be using for the hands-on portion of this lesson, partly because it is simple and easy to learn, and because their code-based agents are so powerful and flexible. 

**[LangChain](https://www.langchain.com/agents)** is a general-purpose framework for building LLM-powered applications, including agents. It provides abstractions for tools, memory, chains, and agents, making it easier to assemble complex systems quickly. The trade-off is complexity. LangChain can feel heavy, and understanding what is happening internally requires more effort. We initially planned to use LangChain for Code the Dream, but the learning curve was much too steep for our purposes. 

**[LlamaIndex](https://developers.llamaindex.ai/python/framework/understanding/agent/)** isn't just for RAG, but also has an agentic framework that can be very powerful for building agents that interact with external data sources. 

All three frameworks solve the same core problem: turning LLMs into systems that can act with the help of external tools.  

### Tool-Based vs. Code-Based Agents
Not all agents work the same way. A useful distinction that is becoming more prevalent is between tool-based agents and code-based agents.

*Tool-based agents* interact with the world by calling predefined tools that are explicitly enumeratred to the LLM. These tools might include things like a plotting functions, a calculator, or any other well-defined software tools. The agent's job is to decide which tools to use and when, given the task. This approach is generally safer and easier to control, because the agent can only do what the available tools allow.

*Code-based agents*, on the other hand, are given free reign to generate and execute novel code to reach their given goal. Instead of selecting from a fixed set of tools, the agent writes code, runs it, inspects the output, and continues from there. This is extremely powerful and flexible. While careful sandboxing and guardrailes are required to keep things safe, in practice, it turns out that LLMs are surprisingly good at writing correct code, so code-based agents can be very effective. We will see an example of this in a hands-on lesson. 

Many real systems blend tool- and code-based approaches, but understanding the distinction helps explain why some agents feel constrained while others feel almost like autonomous programmers.

### Abstractions and interfaces for agents
As agents become more common, an important practical question arises: how do agents actually connect to tools, data sources, and external systems in a consistent way? Most LLM frameworks have their own methods for interacting with tools, which can lead to fragmentation and compatibility issues. To deal with this, there are a few possible solutions. 

One, just like there exist abstraction layers for LLMs that let you interface with multiple LLM providers in a consistent way, there are abstraction layers for agents that let you interface with multiple agent frameworks in a consistent way, such as [any-agent](https://github.com/mozilla-ai/any-agent) from Mozilla. 

Also, the [Model Context Protocol](https://en.wikipedia.org/wiki/Model_Context_Protocol) (MCP), created by Anthropic, is an emerging standard that defines a *common interface* for exposing tools and resources to agents. It is a protocol that standardizes how agents discover tools, understand their inputs and outputs, and calls them safely, much like a USB standardizes how devices connect to computers. 

## Next steps
AI agents represent a shift from "text exchange" to "tool building"." The core ideas - autonomy, iterative reasoning, and tool use -- are simple, but powerful. In the rest of the lessons, we will move from these concepts to hands-on examples, where you will see how agents behave in practice.