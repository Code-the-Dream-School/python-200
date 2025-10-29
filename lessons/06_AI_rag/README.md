# Week 6: Augmenting AI: From Memory to Retrieval

> Note this is a draft.
 
Welcome to the Week 6 in Python 200. This week we will focus on ways to augment the knowledge base of LLMs. As we saw last week, LLMs are models trained on data just like other models. And they lack information, and can hallucinate, make things up. It is important to augment, or expand their knowledge base, so they have all the information needed to perform optimally with updated or proprietary information. 

> To fill in later: brief motivational preview here. Briefly explain why this lesson matters, what students will be able to do by the end, and what topics will be covered. Keep it tight and motivating.

> For an introduction to the course, and a discussion of how to set up your environment, please see the [Welcome](../README.md) page.  

## Topics
1. [Overview](01_llm_augmentation_review.md)  
Here we will discuss the three main types of ways to augment LLMs knowledge base (adding required information its prompt (context injection), fine-tuning, and retrieval-augmented generation (RAG)), and when each method should be used. We will focus on different types of RAG before diving into practical examples.

2. [Naive keyword-based RAG](02_keyword_rag.md)  
Parse a set of PDFs into text chunks, and perform basic keyword matching on chunks, and inject the best-matching chunk into the prompt. Evaluate results. Highlights the limitations of brittle keyword search. This will be VERY simple, to illustrate the most basic concepts of RAG (like 100 lines of code). Introduce concept of evaluation of RAG system using deepeval.

3. [Semantic RAG](03_semantic_rag.md)  
Adapt Nir Diamant's semantic RAG example: 
https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/simple_rag.ipynb

    Use sentence embeddings to represent text chunks, and store embeddings in a FAISS index. Retrieve top-k semantically similar chunks for each query: inject retrieved content into the prompt. Re-evaluate using same criteria as 1. Expect noticeable improvements in relevance and accuracy. Also build a semantic store using pgvector for more production-level pipeline. 

4. [Llamaindex](04_llamaindex.md)  
INtroduction to Llamaindex, and implement what we did in step 3, but using Llamaindex. The main point here will be about a 10:1 code reduction, and using a production-ready framework that is maintained by a group of developers, versus rolling our own. In professional settings, we will most likely be using something like llamaindex, but it is  important to understand what is going on on the back end (hence why we are rolling our own in step 3. 

