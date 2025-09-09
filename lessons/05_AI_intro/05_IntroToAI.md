# Introduction to AI

> ### Contributor Note 
> This is a working template for Python 200, Week 5. Each section below needs to be expanded into a full lesson. Use the code ideas and goals as a starting point — feel free to add examples, exercises, and links to visualizations or datasets. 

Welcome to the Week 8 in Python 200, Introduction to Artificial Intelligence! 

> To fill in later: brief motivational preview here. Briefly explain why this lesson matters, what students will be able to do by the end, and what topics will be covered. Keep it tight and motivating.

> For an introduction to the course, and a discussion of how to set up your environment, please see the [Welcome](00_Welcome.md) lesson. 

# Table of Contents
1. [Introduction to language processing](#1-introduction-to-language-processing)
2. [OpenAI Chat Comletions API](#2-openai)
3. [Abstraction layers](#3-abstraction-layers)
4. [Prompt engineering](#4-prompt-engineering)
5. [Chatbots](#5-chatbots)
6. [Ethics, bias, and responsible AI](#6-ethic-bias-and-responsible-ai)
7. [Wrap-up](#7-wrap-up)

## 1. Introduction to Language Processing
Explain what LLMs are and how they differ from earlier NLP models, and why language models have become so much more powerful recently. Discuss how language is converted to vectors (tokenization). Give broad overview of AI landscape.

### Code ideas
- Tokenization
- Visualize semantic similarity using PCA
  
## 2. OpenAI 
Intro and overview of openai api chat completions endpoint. Go over required params (messages/model), but also the important optional params (max_tokens, temperature, top_p etc). Mention responses endpoint (more friendly to tools/agents). Discuss and demonstrate use of moderations endpoint.

### Code ideas
- Get students set up with openai api.
- Teach basics of chat completions api (necessary params, important params).
- Moderatoins endpoint (this is cool free useful makes openai fairly unique).

## 3. Abstraction layers
Instead of getting locked into a single vendor or style, there are a few packages that provide an abstraction layer across LLM providers and local LLMs (you can run inference locally using Ollama). Here we'll discuss a few of these (langchain, liteLLM, any-llm). 

### Code ideas
- Go into detail with liteLLM (this is probably the most commonly used LLM wrapper)

## 4. Prompt Engineering
There are better and worse ways to get responses from a model, here we'll go over the fundamentals of *prompt engineering*. Zero shot, one shot, few-shot, and chain of thought prompting.

### Code ideas
- Provide different types of problems that respond well to different types of prompt engineering tasks (e.g., math vs analogical reasoning).

## 5. Chatbots
LLMS don't remember what came before, they are intrinsically stateless. Discuss why this is important and how to use messages and roles witht he APIs to build a thin persistence layer to get memory of conversation and build chat application. 

### Code ideas
- Build different chatbot personalities for different types of tasks.
- Use liteLLM for this to easily switch out different vendors (e.g., local vs openai etc)

## 6. Ethic, bias, and responsible AI
LLMs just ML models trained on data, so are subject to the same biases as other models. But also, companies are literally building nuclear reactors to fuel the GPU cores needed for training and inference. A lot of people are telling us how AI is going to change our lives for the better, but we should ask whose lives, and how exactly will they be improved?

There are tons of great resources and videos on this, for instance this great resource [from Amherst college library](https://libguides.amherst.edu/c.php?g=1350530&p=9969379).

## 7. Wrap-up
Summarize the key takeaways from this lesson. Discuss connection to assignment (and give link to assignment). Preview next lesson, especially if it is connected to this one. 

