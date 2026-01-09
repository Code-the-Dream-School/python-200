# Week 7: AI Agents: Building Autonomous Systems

> Note this is a draft. We'll finish it out once the lessons are more finalized.
 
Welcome to the Week 7 in Python 200. This week we will focus on AI *agents*, which are systems that can autonomously perform tasks when given a goal and a set of tools. AI agents are becoming increasingly important in the field of artificial intelligence.

> To fill in later: brief motivational preview here. Briefly explain why this lesson matters, what students will be able to do by the end, and what topics will be covered. Keep it tight and motivating.

> For an introduction to the course, and a discussion of how to set up your environment, please see the [Welcome](../README.md) page.  

## Topics
1. [Overview](01_agents_intro.md)  
An overview of AI agents, including definitions, types of agents, and real-world applications. Introduce the concept of autonomous systems and how they differ from traditional AI models. Introduction to the ReAct framework for agents, and the distinctions between tool-based and code-based agents. We will discuss different frameworks for building agents, including smolagents, Langchain, Llamaindex. 

2. [Hello, agent](02_time_agent.md)  
We will introduce a minimal agent, a kind of "Hello word" of agents, showing how to build and call a simple tool (function) from scratch. 

3. [Building a simple ETL Agent](03_etl_agent.md)  
ETL agent from scratch. We will build an agent that can perform a simple ETL (Extract, Transform, Load) task using the ReAct framework. The agent will be able to extract data from a source (e.g., a CSV file), transform the data (e.g., cleaning, filtering, aggregating), and load the data into a target destination (e.g., a database or another file). This will involve defining the tools (functions) for each step of the ETL process, and implementing the agent logic to orchestrate these tools based on user input.

4. [Smolagents](04_smolagents.md)  
An introduction to the smolagents framework for building AI agents, which is HuggingFace's lightweight and flexible framework for creating agents. We will cover the basics of smolagents, including how to define tools, create and run agents. We will demonstrate both tool-based and code-based agents using smolagents, and discuss best practices for building effective agents.

5. [Demo: AI Paired Programmer](05_github_copilot.md)    
A demonstration of how to install and use Github Copilot as a code-assistant agent. We will show how to use it to evaluate an entire code project, fix tests that are not passing, and build a jupyter notebook to demonstrate the project's features. We will also discuss the strengths and limitations of using AI code assistants when writing code. 
