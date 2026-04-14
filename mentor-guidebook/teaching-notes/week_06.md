# Week 6: Augmenting AI with RAG

## Overview

Students learned how to give LLMs access to information they weren't trained on, using Retrieval-Augmented Generation (RAG). The week builds from a conceptual comparison (RAG vs. fine-tuning vs. prompt engineering), to a simple keyword-based RAG implementation, to semantic RAG using vector embeddings and FAISS, to production-ready RAG using the LlamaIndex framework. The running example is a Q&A chatbot for a fictional solar company (BrightLeaf Solar).

## Key Concepts

**Why RAG?** — LLMs have a training cutoff and don't know your private documents. RAG solves this by retrieving relevant documents at query time and injecting them into the prompt. It's usually cheaper and more controllable than fine-tuning.

**The RAG pipeline** — Three steps: (1) Retrieve relevant documents from a knowledge base using the user's query, (2) Augment the prompt by inserting those documents as context, (3) Generate a response using the model + context. The lessons build each piece.

**Keyword vs. semantic retrieval** — Keyword RAG matches based on word overlap (fast, simple, brittle). Semantic RAG converts text to embedding vectors and finds documents that are *conceptually similar*, even if they use different words. Semantic search is almost always better in practice.

**Vector embeddings and FAISS** — Text gets converted to high-dimensional vectors where similar meanings are close together. FAISS is a library for fast similarity search in that vector space. This is the core of semantic RAG.

**LlamaIndex** — A framework that handles the RAG pipeline for you (document loading, chunking, indexing, retrieval). The lesson shows that what takes ~100 lines of manual code collapses to ~10 with LlamaIndex. This is the pattern used in production.

## Common Questions

- **"Why not just paste all my documents into the prompt?"** — Context windows have limits, and large prompts are expensive. RAG retrieves only the relevant parts. Also, large context windows don't guarantee the model pays equal attention to everything in them.
- **"What's the difference between an embedding and a token?"** — Tokens are the input units (roughly word-pieces). Embeddings are fixed-length vector representations of text meaning, produced by an embedding model. They're related but serve different purposes.
- **"Why does LlamaIndex need the OpenAI API key for retrieval?"** — By default, LlamaIndex uses OpenAI's embedding model to convert documents and queries to vectors. You can configure it to use other embedding models, but OpenAI is the default.
- **"What is 'chunking'?"** — Splitting documents into smaller pieces before indexing. Retrieving a 200-word paragraph is usually more precise than retrieving a 20-page document. LlamaIndex handles chunking automatically.

## Watch Out For

- **API key required for all exercises** — Embedding calls cost money (though they're very cheap). Students without a working API key will hit errors in lessons 3 and 4. Confirm early in the session that everyone's `.env` is set up.
- **FAISS installation** — FAISS can be finicky to install on some systems. If students hit installation errors, the `faiss-cpu` package is usually the right one to use on machines without a GPU.
- **LlamaIndex version changes** — LlamaIndex has changed its API substantially between versions. If students hit import errors, check that their installed version matches what the lesson uses.
- **Confusing retrieval quality with generation quality** — If the chatbot gives a wrong answer, the bug could be in retrieval (wrong document fetched) or generation (model misinterpreted good context). Encourage students to print the retrieved context to debug.

## Suggested Activities

1. **RAG failure mode demo:** Run the BrightLeaf chatbot with a question that is *not* in any of the company documents. What does it do? Does it make something up, or say it doesn't know? Discuss: how would you make the chatbot more reliable about saying "I don't know"?

2. **Keyword vs. semantic comparison:** Give students the same query to run against both the keyword RAG and semantic RAG systems. Ask: did they retrieve the same documents? What happens with a query that uses different words but means the same thing (e.g., "earnings" vs. "revenue")?

3. **Design your own RAG system:** Ask students: "If you were building a RAG system for a real organization — a school, a clinic, a local business — what documents would you index? What queries would users ask? What would be the biggest failure risks?"
