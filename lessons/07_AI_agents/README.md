# Week 7: AI Agents: Building Autonomous Systems

Welcome to Week 7 of Python 200! Last week we gave LLMs access to external information through RAG. This week we take the next step: giving them the ability to *act*. An AI agent is a system that can autonomously plan and execute a sequence of steps to accomplish a goal -- calling tools, running code, querying databases, and deciding what to do next based on the results.

Agents are increasingly showing up in data engineering workflows: automated ETL pipelines, data quality monitoring, code generation, and more. By the end of the week you will have built agents from scratch and with a production framework, and you will have a clear picture of where they are genuinely useful and where they are not.

> For an introduction to the course, and a discussion of how to set up your environment, please see the [Welcome](../README.md) page.

## Topics
1. [Overview](01_agents_intro.md)  
What is an agent, and how does it differ from a simple LLM call? We introduce the core concepts: tools, the ReAct (reason + act) loop, and the distinction between tool-based and code-based agents. We also survey the main frameworks (smolagents, LangChain, LlamaIndex) before choosing one for the rest of the week.

2. [Hello, agent](02_minimal_agent.md)  
The "Hello world" of agents -- we build and call a simple tool from scratch to see the basic mechanics before adding complexity.

3. [Building a simple ETL Agent](03_analysis_agent.md)  
A more realistic application: we build an agent that can extract, transform, and load data using the ReAct framework, defining tools for each step and letting the agent orchestrate them based on user input.

4. [Smolagents](04_smolagents.md)  
An introduction to smolagents, HuggingFace's lightweight framework for building agents. We cover tool definition, running agents, and the difference between tool-based and code-based agents in a production-ready context.

5. [Demo: AI Paired Programmer](05_code_agent_demo.md)    
A demonstration of AI code assistant agents in practice -- using one to evaluate a project, fix failing tests, and generate documentation. We also discuss the strengths and limitations of code agents, and how tools like GitHub Copilot and Claude Code compare.

## Week 7 Assignments
Once you finish the lessons, head on over to the [assignments](../../assignments/README.md) to get more hands-on practice with the material.