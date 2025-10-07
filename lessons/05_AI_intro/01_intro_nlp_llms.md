# Introduction: Large Language Models
Welcome to the first lesson in the AI module! In this lesson, you’ll get a behind-the-scenes look at how Large Language Models (LLMs) like ChatGPT work. We’ll start with a quick tour of natural language processing (NLP)—the field that helps computers understand and generate human language. Then you’ll learn how models like GPT are built, connecting them to many of the machine learning concepts you learned in the previous module. By the end, you’ll have a solid, intuitive sense of how AI models represent meaning, and generate text.

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
- *Conversational AI*, which powers chatbots and virtual assistants that engage in fluid, multi-turn dialogues. 

Ultimately, NLP aims to make human-computer interaction as intuitive as human-to-human exchanges, so it can be used in fields as diverse as healthcare diagnostics, explaining complex legal documents, and personalized education. 

### NLP Methods
The field of NLP has seen a transformation in methods over the past 50 years. It has seen a progression from rigid, rule-based approaches to data-driven, adaptive techniques that leverage machine learning and neural networks. 

In its early days, NLP depended on rule-based systems and hand-crafted grammars to parse linguistic inputs. In these days, human experts manually encoded explicit linguistic rules into NLP systems. These methods were brittle, and struggled with the variability in real-world language. 

The shift to statistical methods in the late 20th century marked a shift in method, incorporating probabilities to model patterns in language. This paved the way for machine learning, where algorithms learn directly from examples. Today, the dominant paradigm is deep learning. As we saw last week, deep learning is a subset of machine learning that uses neural networks with multiple layers to automatically extract features from raw data. 

At the forefront of this approach are large language models (LLMs), such as those powering tools like GPT, which are pre-trained on billions of internet-scale sources of text. 

The release of ChatGPT (from OpenAI) on November 30, 2022, was a watershed moment in the history of NLP. It lead to a massive surge in public awareness and usage of LLMs. This easily accessible chatbot allowed millions to interact directly with an advanced LLM. The app amassed over a million users in one day, instantly creating awareness of the power of AI. It also accelerated its adoption in industries like education, customer service, and content creation. 

The newest wave of LLMs inspired a surge in research, and spawned ethical debates on issues like misinformation (LLM hallucinations), job displacement, and even concerns about [conscious AI](https://www.scientificamerican.com/article/google-engineer-claims-ai-chatbot-is-sentient-why-that-matters/). For an interesting discussion of the impact of ChatGPT on the field of NLP, see the [oral history in Quanta Magazine](https://www.quantamagazine.org/when-chatgpt-broke-an-entire-field-an-oral-history-20250430/). 

Since 2022, LLMs have shifted from being mostly academic curiosities to tools that attract billions in investment every year, and they are reshaping how people learn and interact with computers. 

In the rest of this lesson, we will learn some of the technical basics of how LLMs like ChatGPT work, and try to demystify their operations. Ultimateley, they are just another machine learning model, and they are trained to predict the next token in a string of tokens. 

## 2. Large language models (LLMs)
### LLMs: autocomplete at scale
Modern LLMs are machine learning models that are trained to predict the next word in a sequence, given all the words that came before. Imagine starting a sentence, and the model is tasked with filling in the blank: 

    The cat sat on the ___

The model looks at the context -- the first five words -- and generates a probability distribution to find the most likely next word. It might estimate that "mat" has a 70% chance, "floor" 20%, "sofa" 5%, and so on. It then picks the most likely candidate (or sometimes samples from that distribution to keep things more varied). 

This simple "predict the next word" trick turns out to be extremely powerful. By repeating it over and over, LLMs can generate entire paragraphs, answer questions, write code, or carry on conversations.

There is an excellent discussion of this at 3blue1brown (the following will open a video at YouTube):

[![Watch the video](https://img.youtube.com/vi/LPZh9BOjkQs/hqdefault.jpg)](https://www.youtube.com/watch?v=LPZh9BOjkQs)

You have likely seen a similar mechanism on your phone when writing text and it suggests the next word using its *autocomplete* feature. Basically what LLMs do is autocompletion on a large scale. What makes LLMs *large* is the amount of data used to train them, and the size of the models. 

LLMs are trained on enormous collections of text, including books, Wikipedia, articles, and large parts of the internet. The models also contain billions (sometimes even trillions) of parameters, which allow the model to capture much more subtle patterns in language. It's this large scale, as well as the underlying transformer architecture (which we will discuss below) that makes modern LLMs so much more fluent and flexible than your phone's autocomplete feature. 

### How LLMs Learn: Self-supervised learning 
The training process for LLMs is different from what we saw in the ML module -- there, we learned that humans provide labeled data as ground truth to help train the models. Instead, LLMs use what’s called *self-supervised learning*. Because the "correct next word" is already present in every text sequence, the data effectively labels itself. 

For example, in the phrase "The cat sat on the mat," the model can practice by hiding "mat" and predicting it from the context. This setup is also called *autoregression*, because the model predicts each word based on all the words before it.

With this approach, you can train on billions or trillions of examples without having to manually annotate ground-truth data. Over time, the model learns facts, grammar, and reasoning patterns simply by getting better at predicting the next word.

There is one wrinkle we should cover regarding how LLMs learn before moving on to more technical matters. There are really *two* different learning modes for LLMs. First, by training on huge bodies of text in the next-word-prediction task, we end up with *foundational* or *pretrained* or *base* models. These are general purpose models that embody information from extremely broad sources. 

However, just foundational models don't work well in special-purpose jobs like personal assistants, chatbots, etc. To get good performance on such specialized tasks, a second training step is needed, where these foundational models are *fine-tuned* on a labeled dataset that is tailored to a specific task or application.

![pretrained vs fine-tuned llm](resources/pretrained_finetuned_llm.jpg)

In other words, fine-tuning takes a foundational model and adjusts it for specific purposes, such as answering questions, following instructions, or writing in a particular style. There are various ways to do this. One, supervised fine-tuning (SFT) follows a more traditional ML approach where the model is given paired examples of inputs and desired outputs. 

Another is [reinforcement learning from human feedback](https://www.youtube.com/watch?v=T_X4XFwKX8k) (RLHF). With RLHF the model adapts (using reinforcement learning procedures) to produce responses that are ranked more highly by human judges. 

The result with fine-tuning is the production of specialized models built on top of the same foundation -- one model might become a customer service chatbot, another a medical assistant, and another a coding helper. The distinction between the general pretrained model and its fine-tuned variants is key to understanding why LLMs are so adaptable in practice.

While in this course we will not go through the process of building your own LLM, the excellent book [Build a Large Language Model from Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka, walks you through this in detailk using PyTorch if you are interested. The above picture is adapted from his book.

In the next section we will dig into the details about how LLMS actually work: as we said, it isn't just that they are *large*, but their *architecture*, that makes them so powerful. 


## 3. LLM architecture
In this section we will walk step-by-step through the following simplifed LLM architecture diagram, which is adapted from Chapter 2 of Raschka's excellent book:

![LLM architectural](resources/llm_architecture.jpg)

What is an LLM video: 
https://www.youtube.com/watch?v=5sLYAQS9sWQ

There are three main steps that it is important to focus on when understanding how LLMs get so good at predict the next word in a sequence:

- tokenization
- token embedding
- attention

### Tokenization: From raw text to token IDs
You can learn more about tokenization at the following resources:
- [Super Data Science video](https://www.youtube.com/watch?v=ql-XNY_qZHc)
- [Huggingface introduction](https://huggingface.co/learn/llm-course/en/chapter2/4)

Tokenization is the process of breaking chunks of text into smaller pieces that are in the LLM's vocabulary. For example, the sentence "The cat sat on the mat." might be split into tokens like ["The", " cat", " sat", " on", " the", " mat", "."]. These tokens are then mapped to unique integer IDs.

![LLM tokenization](resources/llm_tokenization.jpg)

Importantly, tokens are not always whole words. To keep the vocabulary a manageable size, many tokenizers break rare or complex words into smaller chunks. For example, "blueberries" might become ["blue", "berries"]. This makes it possible to represent *any* string of text, even if it never appeared in training.

You can [play online]( https://platform.openai.com/tokenizer) with a popular tokenizer, *tiktoken*. There, you can explore how tiktoken breaks down text into parts and creates numerical ids for each token. 

[add practical exercise here]


### Embeddings: From IDs to semantics
Once a tokenizer has converted text into integer IDs, these don't "mean" anything -- the tokenizer has just created some arbitrary numbers to stand in for words or subwords. We start ascribing meanings when the token IDs are passed through an *embedding layer*, which assigns each token a tensor, a numerical array of values:

![token embeddings](resources/embedding_layer_rashka.jpg)

 In the image each token ID is mapped to a very small vector of numbers (three values). In reality the embedding layer typically maps each token to a tensor of thousands of numbers. 
 
 At first these token embeddings are assigned random numbers. However, as training proceeds, semantically similar tokens will end up closer together. So the token embeddings for `brother` and `sister` end up near each other, while `car` and `bicycle` cluster elsewhere. Over time, this training process produces what we might call a semantic similarity space: a geometric landscape where semantically related tokens cluster together in a heirarchical way:

![semantic space](resources/semantic_space.jpg)
Conceptual drawing of token embeddings arranged in semantic space. Words with related meanings cluster together. Note here the high-dimensional token embedding is projected to a low-dimensional space for visualization.

Embedding resources:
https://medium.com/@saschametzger/what-are-tokens-vectors-and-embeddings-how-do-you-create-them-e2a3e698e037

Slightly more technical intro to embedding: 
https://www.youtube.com/watch?v=lPTcTh5sRug

### Attention: Context-aware embeddings
 *Attention* was the final puzzle piece needed to make modern LLMs so powerful. It was introduced by a team of researchers at Google Brain in a groundbreaking paper called [Attention is all you need](https://arxiv.org/abs/1706.03762).

In the above static embedding space, each token has a fixed default neighborhood: `apple` lies near other fruits, `car` clusters with other vehicles, and `queen` sits close to `king` and `sister`. But language isn't static -- meanings shift with context. In a sentence about fruit salad, `apple` should clearly belong with oranges and grapefruits. In a sentence about smartphones, `Apple` should move toward phones and computers. Static token embeddings can't make this adjustment on their own. This is where *attention* comes in: it allows tokens to dynamically reshape their position based on surrounding words, pulling apple toward "fruit" or "phone" as needed. Attention is the mechanism that enriches the thin semantics of embeddings with contextual cues, helping to resolve ambiguities that emerge in natural language. 


To get an quick intuitive understanding of attention, before reading on, please watch the following short video: 

[![Watch the video](https://img.youtube.com/vi/0aG4cSfFvC0/hqdefault.jpg)](https://www.youtube.com/shorts/0aG4cSfFvC0)


Each token starts as a point in embedding space, but attention allows them to influence one another. Because "apple" and "orange" are semantically similar, attention gives them a high weight of mutual influence. As a result, the embedding for "orange" is nudged slightly toward "apple," and vice versa. Now consider "Apple released a new phone." Here, the word "Apple" receives more weight from "phone" and "released" than from unrelated words, so its embedding is shifted toward the electronics cluster. 

In general, attention acts as a *similarity-based weighting system*: words that are closer to each other exert a stronger pull. This process turns a static semantic map into a context-sensitive representation of meaning.

![Attention mechanism](resources/attention_influence.jpg)

We are leaving out the mathematical details here, because this is a conceptual overview. There are *many* in-depth treatments online, such as: 

- [Video](https://www.youtube.com/watch?v=OxCpWwDCDFQ) 
- [Web page](https://cohere.com/llmu/what-is-attention-in-language-models)
  
Attention mechanism was the magic sauce that supercharged progress in NLP. When combined with massive training data, large models, and the self-supervised task of predicting the next token, it enabled LLMs to generate remarkably human-like speech patterns.

### Transformer: attention + MLP
The transformer architecture is the thing that provides the "T" in GPT, which stands for "generative pretrained *transformer*. 

Attention is your context engine. After tokenization and embeddings, each token "looks" at the others and borrows what it needs, with stronger weights from the most relevant neighbors. That gives every token a richer, context-aware vector. This enriched token is then fed through a neural network (a multilayered perceptron, or MLP), that processes the outputs of these enriched embeddings, boosting the ability to predict the next word in the sequence. 

GPT models stack many of these these transformer blocks together (GPT3 has 96 transformer blocks), generating highly contextualized and refined representations of tokens. 

We're skipping many details here: for instance, *multi-head attention* runs several attention patterns in parallel within a transformer block and then mixes them; there are many other details in the ML pipeline that are needed to make the pipeline robust. Our point is more conceptual. 

### Predicting the next word: de-embedding
Crucially, the heavy lifting already happened inside the transformer stack. But we still need to predict the next word in our word sequence! 

After all those transformer blocks have done their job, we're left with a final vector for each token in the input. Imagine the last word in the sequence — its vector lives somewhere in a 3D space (or, in a real model, maybe 768 or 12288 dimensions, but let’s keep it 3D for now). 

Now comes the final moment. We ask: given that 3D vector, which of the 50,000 possible tokens in the vocabulary should come next? This is a classic prediction problem. The token with the highest score wins. That’s the predicted next word. Everything builds up to this. One vector in, one token out.

This is done by a very simple linear neural network (two layers) that is a translator from the model's embedding space to actual tokens. We can think of it as a "de-embedding". 


## 4. Demo: Visualizing embeddings
In the following demonstration we will visualizing text embeddings based on their similarity. 

> Aside on handling API key (put in README.md)

First, load API key. 

```python
from dotenv import load_dotenv

if load_dotenv():
    print("Successfully loaded api key")
```

Generate movie summary dictionary that we will use to detect semantic similaties.

```python
movie_summaries = [
    # Marvel Superhero Movies
    {
        "title": "Iron Man (2008)",
        "summary": "Billionaire genius Tony Stark builds a high-tech suit to escape captivity and becomes Iron Man, fighting global threats with his wit and advanced technology."
    },
    {
        "title": "The Avengers (2012)",
        "summary": "Earth’s mightiest heroes, including Iron Man, Captain America, Thor, and Hulk, unite to stop Loki and his alien army from conquering the planet."
    },
    {
        "title": "Black Panther (2018)",
        "summary": "T’Challa, king of Wakanda, embraces his role as Black Panther to protect his nation and the world from a powerful enemy threatening their vibranium resources."
    },
    {
        "title": "Spider-Man: No Way Home (2021)",
        "summary": "Peter Parker, unmasked as Spider-Man, teams up with alternate-universe heroes to battle villains from across the multiverse after a spell goes wrong."
    },
    {
        "title": "Captain Marvel (2019)",
        "summary": "Carol Danvers unlocks her cosmic powers as Captain Marvel, joining the fight against the Kree-Skrull war while uncovering her lost memories on Earth."
    },
    # Christmas-Themed Movies
    {
        "title": "Home Alone (1990)",
        "summary": "Young Kevin is accidentally left behind during Christmas vacation and must defend his home from bumbling burglars with clever traps and holiday spirit."
    },
    {
        "title": "Elf (2003)",
        "summary": "Buddy, a human raised by elves, journeys to New York City to find his real father, spreading Christmas cheer in a world that’s lost its festive spark."
    },
    {
        "title": "The Polar Express (2004)",
        "summary": "A young boy boards a magical train to the North Pole, embarking on a heartwarming adventure that tests his belief in the magic of Christmas."
    },
    {
        "title": "A Christmas Carol (2009)",
        "summary": "Ebenezer Scrooge, a miserly old man, is visited by three ghosts on Christmas Eve, learning the value of kindness and the true meaning of the holiday."
    },
    {
        "title": "Love Actually (2003)",
        "summary": "Interwoven stories of love, loss, and connection unfold in London during the Christmas season, celebrating the messy beauty of human relationships."
    },
    # Romantic Comedies
    {
        "title": "When Harry Met Sally... (1989)",
        "summary": "Harry and Sally’s evolving friendship over years sparks debates about love and friendship, culminating in a heartfelt realization during a New Year’s Eve confession."
    },
    {
        "title": "The Proposal (2009)",
        "summary": "A high-powered executive forces her assistant into a fake engagement to avoid deportation, leading to unexpected romance during a chaotic family weekend in Alaska."
    },
    {
        "title": "Crazy Rich Asians (2018)",
        "summary": "Rachel Chu accompanies her boyfriend to Singapore, facing his ultra-wealthy family’s disapproval in a whirlwind of opulence, tradition, and newfound love."
    },
    {
        "title": "10 Things I Hate About You (1999)",
        "summary": "A rebellious teen, Kat, is wooed by bad-boy Patrick in a modern Shakespearean tale of high school romance, deception, and heartfelt connection."
    },
    {
        "title": "Notting Hill (1999)",
        "summary": "A humble London bookseller falls for a famous American actress, navigating fame, cultural clashes, and personal insecurities to pursue an unlikely love story."
    }
]
```

### Generate embeddings 

The `text-embedding-3-small` model converts text into numerical vectors and captures the "meaning" of the input text (in this case the movie reviews). Above, we discussed embedding vectors for single tokens, but OpenAI's model will do this for any length of text, such as our movie summaries. 

```python
client = OpenAI()

embeddings = []
for summary in movie_summaries:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=summary
    )
    embeddings.append(response.data[0].embedding)
embeddings = np.array(embeddings)
print(embeddings.shape)
```


### Examine embeddings in 2d using PCA

PCA reduces high-dimensional embeddings (1500-dimensions) to a lower-dimensional 2D for intuitive plots, making embeddings easy to visualize. It makes more clear the similarity relations among the embeddings, revealing the semantic structure between the summaries. 

We discussed above just how important this kind of perspective is in the development of LLMs/self-attention etc!

```python
# Do PCA to project to lower-dimensional embedding space
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Visualize embeddings
plt.figure(figsize=(8, 6))
for i, summary in enumerate(movie_summaries):
    title = movie_titles[i]
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
    plt.text(embeddings_2d[i, 0] + 0.02, embeddings_2d[i, 1], title, size=8)
plt.title("2D Visualization of Summary Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
```

We will see in the assignment that you can feed any text you'd like into the embedding model, it is a lot of fun to probe the semantic map embodied in these models. 


## 5. Key points
Congrats we just covered the basics of natural language processing and modern LLM function! Some of the key points we covered:

- Modern NLP helps computers understand and generate human language using data-driven deep learning rather than hand-crafted rules.
- Large Language Models (LLMs) like GPT are trained with self-supervised learning to predict the next word in a sequence -- essentially, autocomplete at scale.
- Tokenization, embeddings, and attention let models capture word meaning and context dynamically.
- Transformers incorporate attention layers and neural networks to generate  context-aware text predictions.

While in subsequent lessons we will use APIs that rely on models built with these architectures, it's important to understand what’s happening under the hood. Hopefully, knowing a little bit about how tokenization, embeddings, and attention works will help demystify LLMs and gives you intuition for why they can generate language so effectively (and sometimes make such strange errors, which we will discuss in Lesson N).