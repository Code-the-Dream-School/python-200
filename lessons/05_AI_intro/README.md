# Week 1: Introduction to AI

Welcome to the Week 5 in Python 200, Introduction to Artificial Intelligence! 

> To fill in later: brief motivational preview here. Briefly explain why this lesson matters, what students will be able to do by the end, and what topics will be covered. Keep it tight and motivating.

> For an introduction to the course, and a discussion of how to set up your environment, please see the [Welcome](../README.md) page.  

## Topics
1. [Introduction to language processing](01_intro_language_processing.md)  
Explain what LLMs are and how they differ from earlier NLP models, and why language models have become so much more powerful recently. Discuss how language is converted to vectors (tokenization). Give broad overview of AI landscape.

2. [OpenAI Chat Completions API](02_open_ai_api.md)  
Intro and overview of openai api chat completions endpoint. Go over required params (messages/model), but also the important optional params (max_tokens, temperature, top_p etc). Mention responses endpoint (more friendly to tools/agents). Discuss and demonstrate use of moderations endpoint.

3. [Abstraction layers](03_abstractions.md)  
Instead of getting locked into a single vendor or style, there are a few packages that provide an abstraction layer across LLM providers and local LLMs (you can run inference locally using Ollama). Here we'll discuss a few of these (langchain, liteLLM, any-llm). 

4. [Prompt engineering](04_prompt_engineering.md)  
There are better and worse ways to get responses from a model, here we'll go over the fundamentals of *prompt engineering*. Zero shot, one shot, few-shot, and chain of thought prompting.

5. [Chatbots](05_chatbots.md)  
LLMS don't remember what came before, they are intrinsically stateless. Discuss why this is important and how to use messages and roles witht he APIs to build a thin persistence layer to get memory of conversation and build chat application. 


6. [Ethics, bias, and responsible AI](06_ai_ethics.md)    
LLMs are just ML models trained on data, so are subject to the same biases as other models. But also, companies are literally building nuclear reactors to fuel the GPU cores needed for training and inference. A lot of people are telling us how AI is going to change our lives for the better, but we should ask whose lives, and how exactly will they be improved?