# Week 6: Augmenting AI: From Memory to Retrieval

Welcome to Week 6 of Python 200! Last week we learned to interact with LLMs through APIs and to prompt them effectively. But LLMs have a fundamental limitation: their knowledge is frozen at training time. Ask one about your company's internal documents, a recent news event, or a database it has never seen, and it will either guess or hallucinate. This week we look at the main strategies for getting around that -- injecting context into prompts, fine-tuning models on new data, and retrieval-augmented generation (RAG). We will spend most of our time on RAG, which is the most practical and widely used approach in production pipelines.

> For an introduction to the course, and a discussion of how to set up your environment, please see the [Welcome](../README.md) page.  

## Topics
1. [Intro to LLM Knowledge Augmentation](01_llm_augmentation_intro.md)  
An overview of the three main strategies for augmenting an LLM's knowledge: context injection (adding information directly to the prompt), fine-tuning (retraining the model on new data), and retrieval-augmented generation (RAG, giving the model access to an external data store at query time). We discuss the trade-offs of each before diving into RAG for the rest of the week.

2. [Keyword-based RAG](02_keyword_rag.md)  
To illustrate the concept of RAG, we will implement a very simple RAG pipeline where we give an LLM access to a folder of PDFs that it searches to help it generate better answers. This is meant to illustrate the core RAG ideas in a simple context that we build from scratch. 

3. [Semantic RAG](03_semantic_rag.md)  
Keyword search finds documents that contain your exact words -- but what if the document says "automobile" and you searched for "car"? Semantic RAG solves this by using embeddings (which we saw last week) to search by meaning instead of exact words. We build a semantic RAG pipeline from scratch to illustrate the concept, and use Docker to spin up a pgvector database for persistent vector storage.

4. [Llamaindex](04_llamaindex.md)  
Introduction to LlamaIndex, a production-ready framework for building RAG pipelines. We will re-implement what we built in lesson 3, but using LlamaIndex -- the main takeaway being roughly a 10:1 reduction in code, plus the reliability that comes from using a well-maintained framework rather than rolling our own.

## Week 6 Assignments
Once you finish the lessons, head on over to the [assignments](../../assignments/README.md) to get more hands-on practice with the material.

