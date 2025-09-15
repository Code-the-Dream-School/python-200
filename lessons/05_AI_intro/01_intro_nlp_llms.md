# Introduction: Large Language Models
General overview aned introduction. GPT etc etc. 

> Welcome to the first lesson in the AI module! In this lesson, we will look at how Large Language Models (LLMs) like ChatGPT work. First, we will explore field of natural language processing (NLP) more generally. Next, we will demystify how models such as GPT are built, explain key ideas like tokenization, embeddings, and self-attention, and connect these concepts back to the machine learning skills you’ve already learned. By the end, you’ll have a clearer sense of how modern AI models represent meaning, generate new text, and why this matters for the AI tools you will be learning about in later lessons.

## 1. Natural language processing (NLP)
While we will provide a brief overview of NLP, we could easily spend an entire course on the topic. If you want to learn more about NLP, check out the following resources:

- [Brief overview from deeplearning.ai](https://www.deeplearning.ai/resources/natural-language-processing/)
- [YouTube video summary](https://www.youtube.com/watch?v=Uuz8ZTV5vdA)
- [Full Stanford online course](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4)

### What is NLP?
NLP is a subfield of artificial intelligence (AI) that aims to give computers the ability to process and generate meaningful human language. This is a very difficult problem: there is a wide gap between the subtle nuances of natural human communication, on one hand (things like slang, jokes, sarcasm, and cultural references), and the rigid, logical structures of computer programs on the other. 

The goals of NLP span a wide spectrum, including:
- *Text classification*, where content is sorted into predefined categories (e.g., spam)
- *Machine translation*, which converts text from one language to another
- *Sentiment analysis*, where the emotional tone of text as positive, negative, or neutral is determined. 
- Search engines that retrieve and synthesize precise responses to user queries.
- *Conversational AI*, which powers chatbots and virtual assistants that engage in fluid, multi-turn dialogues. 

Ultimately, NLP's aims to make human-computer interaction as intuitive as human-to-human exchanges, so it can be used in fields as diverse as healthcare diagnostics, explaining complex legal documents, and personalized education. 

### NLP Methods
The field of NLP has seen a transformation in methodology over the past 50 years. It has seen a progression from rigid, rule-based approaches to data-driven, adaptive techniques that leverage machine learning and neural networks. 

In its early days, NLP depended on rule-based systems and hand-crafted grammars to parse linguistic inputs. In these days, human experts manually encoded explicit linguistic rules into NLP systems. These methods were brittle, and struggled with the variability in real-world language. 

The shift to statistical methods in the late 20th century marked a shift in method, incorporating probabilities to model patterns in language. This paved the way for machine learning, where algorithms learn directly from examples. Today, the dominant paradigm is deep learning. As we saw last week, deep learning is a subset of machine learning that uses neural networks with multiple layers to automatically extract features from raw data. 

At the forefront of this approach are large language models (LLMs), such as those powering tools like GPT, which are pre-trained on billions of internet-scale sources of text. This modern approach has dramatically advanced performance, though it still grapples with challenges like bias in training data and computational demands.

The release of ChatGPT (from OpenAI) on November 30, 2022, was a watershed moment in the history of NLP. It lead to a massive surge in public awareness and usage of LLMs. This easily accessible chatbot allowed millions to interact directly with an advanced LLM. Overnight, the app amassed over a million users, and almost instantly generated awareness of the power of AI. It also accelerated its adoption in industries like education, customer service, and content creation. 

The newest wave of LLMs inspired a surge in research, and spawned ethical debates on issues like misinformation (LLM hallucinations), job displacement, and even concerns about [conscious AI](https://www.scientificamerican.com/article/google-engineer-claims-ai-chatbot-is-sentient-why-that-matters/). For an interesting discussion of the impact of ChatGPT on the field of NLP, see the [oral history in Quanta Magazine](https://www.quantamagazine.org/when-chatgpt-broke-an-entire-field-an-oral-history-20250430/). 

Since 2022, LLMs and NLP have shifted from being mostly academic curiosities, to tools that attract billions in venture capital that are reshaping how millions of people learn and interact with computers. 

In the rest of this lesson, we will learn some of the technical basics of how LLMs like ChatGPT work, and try to demystify their operations. Ultimatley, they are just another machine learning model, and they are trained to predict the next token in a string of tokens.   


# 2 General overview of gpt and LLM
What is large
Pretrained models
Tokenizing Embedding (we will cover this)
Transformer and attention to create better embeddings
Next word prediction: autoregression -- tons of training data



## 2. What Are Language Models (and LLMs)?
Show the progression from early NLP to modern LLMs and why LLMs are different.

**Topics to cover:**
- What is a language model? (next-word prediction, completion)
- Early NLP (rule-based → bag-of-words → word2vec)
- The shift to transformers and LLMs
- What makes a model 'large'? (parameters, data, compute)
- Rise of pretrained models (e.g., BERT, GPT)

**External resources:**
- [A Visual Intro to Language Models – Jay Alammar](https://jalammar.github.io/blog/)
- [How GPT Works (YouTube) – Jay Alammar](https://www.youtube.com/watch?v=bAUM1tG4q6Q)
- [What is a Language Model? (Medium)](https://medium.com/analytics-vidhya/what-is-a-language-model-7412c6c2da5e)

## 3. Key Concepts Behind LLMs
 Explain the core ideas behind modern LLMs. Demystifying LLMs.

**Topics to cover:**
- Transformers and self-attention  
  - Why attention beats RNNs for language
  - Intuition: attending to relevant words
- Embeddings: how language becomes math
- Pretraining vs. fine-tuning  
  - Pretraining on large corpora
  - Fine-tuning options:  
    - Full fine-tuning  
    - Adapter layers  
    - Prompt-/prefix-/LoRA-style tuning (brief overview)

**External resources:**
- [Transformers – A Friendly Introduction (YouTube)](https://www.youtube.com/watch?v=4Bdc55j80l8)
- [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face: Pretraining vs. Fine-tuning](https://huggingface.co/transformers/training.html)
- [OpenAI: Guide to Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)

## 4. How Language Becomes Numbers: Tokenization
Explain tokenization clearly, show real examples, and connect to embeddings.

**Topics to cover:**
- What is tokenization?  
  - Text → tokens → numeric IDs → vectors  
  - Example: “The cat sat.” → `['The', 'cat', 'sat', '.']`
- WordPiece / Byte-Pair Encoding (BPE) basics
- Why subwords help (e.g., “unhappiness” → “un”, “happi”, “ness”)

**External resources:**
- [Tokenization Explained (YouTube) – AssemblyAI](https://www.youtube.com/watch?v=oI4a5FVtxbY)
- [Byte Pair Encoding for Beginners (Medium)](https://towardsdatascience.com/byte-pair-encoding-subword-tokenization-algorithm-77828a70bee0)

**Code placeholder:**
```python
# Show raw tokens for a sentence using an OpenAI or Hugging Face tokenizer.
# Example:
# tokens = tokenizer("The cat sat on the mat.")
# print(tokens)  # ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']
```

## 5. Embeddings and SEmantic Relationships 
Show how similar meanings cluster in space using embeddings.

**Topics to cover:**
- Embedding space: high-dimensional vectors with semantic structure
- Similar words cluster (e.g., king, queen, prince)
- Project to 2D with PCA (or UMAP) for visualization

**Code placeholder:**
```python
# 1) Choose 10–20 words/phrases
# 2) Get embeddings (e.g., OpenAI 'test-embedding-3-small')
# 3) Reduce to 2D with PCA
# 4) Plot with matplotlib and label points
```

**External resources:**
- [OpenAI Docs – Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Word Embeddings Visualization with PCA (YouTube)](https://www.youtube.com/watch?v=T6XKQ2ZGW8I)
- [Visualizing Embeddings – Distill.pub](https://distill.pub/2016/misread-tsne/)

## 6. Summary: Why This Matters

**Key takeaways:**
- Language models power many modern AI applications.
- LLMs learn semantic structure by representing meaning in vector space.
- Tokenization and embeddings explain what models “know.”
- Connects to prior ML concepts (classification, clustering) by adding representation learning.

## 7. Optional Further Exploration

**Links:**
- [Hugging Face Course – Chapter 1](https://huggingface.co/course/chapter1)
- [OpenAI Cookbook – Embedding Visualization Notebook](https://github.com/openai/openai-cookbook/blob/main/examples/Embeddings_visualization.ipynb)
- [Cohere Text Embedding Playground](https://txt.cohere.com/)

