# Semantic RAG

In the previous lesson we looked at a very basic implementation of RAG that used a keyword matching approach between the user query and the data in the external database. Now we'll go over a more complex but efficient RAG implementation, one that is closer to industry-standard RAG frameworks. Remember that even if this implementation looks more complex, the same steps as the keyword-based RAG will be followed. 

The basic semantic search based RAG framework is illustrated in the figure below.
![Naive Semantic RAG](resources/semantic_rag.png)

Immediately you will note some key differences between keyword RAG and semantic RAG.
- In Keyword RAG, we directly used the text from the documents to ascertain its similarity with the query. Here the text is broken into "chunks" and converted into vector embeddings (a list of numbers uniquely representing the chunk) which are then used in the similarity assessment.
- In Keyword RAG, the similarity score used to retrieve the context was simply based on the number of exact word matches. Here, similarity score is computed in the space of the vector embeddings instead of text. We will look at this score later in the lesson.

Although it is important to note that the same basic steps from Keyword RAG (and indeed any RAG framework) are being followed here. Refering to the image above:
- Retrieve: Text extraction, split text into chunks, generate embeddings, data indexing, retrieve similarity, pass chunks
- Augment: Takes place within Gemini LLM
- Generate: Generate Answer

Lets look deeper into the implementation below. In order to focus on the details of the approach, this implementation only looks at how a query is answered instead of looking at a Q&A loop. To help with understanding, we'll map each section of code to the steps in the figure above.

In case you are interested, here are some additional resources: [Youtube video](https://www.youtube.com/watch?v=HREbdmOSQ18), [Non-brand Data article](https://www.nb-data.com/p/simple-rag-implementation-with-contextual)

## Implementation

As before, we are using the example of the fictitious Brightleaf Solar company. Make sure the script has access to the directory with the files (`brightleaf_pdfs`).

### Setting up

Note that as before we store and retrieve the OpenAI API Key in the `.env` file. Then, we create the OpenAI API client object as we ave done before.

```python
# Load .env (expects OPENAI_API_KEY)
if load_dotenv():
    print("Loaded openai api key")
else:
    print("no api key loaded check out .env")
    
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

Next, we load the files to create a corpus that will serve as our database.

### Load and clean PDF text

The first step is to extract the text from the PDF files (the "Text Extraction" step in the figure). The code snippet below loads the files from the directory and cleans the text before storing it in a dictionary.

```python
def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for p in reader.pages:
        t = p.extract_text() or ""
        t = re.sub(r"[“”]", '"', t)
        t = re.sub(r"[’]", "'", t)
        t = re.sub(r"[–—]", "-", t)
        t = re.sub(r"\s+", " ", t).strip()
        parts.append(t)
    return "\n".join(parts)

def load_corpus(pdf_dir: Path):
    corpus = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        text = read_pdf_text(pdf)
        corpus.append({"doc_id": pdf.name, "text": text})
    return corpus

DATA_DIR = Path("brightleaf_pdfs")  # <- your BrightLeaf PDF folder
assert DATA_DIR.exists(), f"{DATA_DIR} not found. Put PDFs there."

corpus = load_corpus(DATA_DIR)
print(f"Loaded {len(corpus)} docs:", [c["doc_id"] for c in corpus])
```

This script is very similar to the one in the previous lesson. A new addition is the `assert` method used to make sure that the directory exists and the text from the files can be extracted and cleaned. Like before, PDFReader is used to extract the text from the pdf files. 
<!--The cleaning process involves the use of the regular expression operator `re.sub()`. Its syntax is given below:
```python
re.sub(pattern, repl, string)
```
Basically, it replaces all occurences of `pattern` in `string` with `repl`.-->

As there are 6 documents in the directory, the output of the code snippet will look like the one below.

```
Loaded 6 docs: ['earnings_report.pdf', 'employee_benefits.pdf', 'mission_statement.pdf', 'partnerships.pdf', 'product_specs.pdf', 'security_policy.pdf']
```
Now that we have extracted the text from all documents, we start the process of chunking the documents into bite-sized portions for more efficient context retrieval.

### Breaking down document text - "Chunking"

Chunking is the process of breaking down text into smaller portions (or "chunks"), usually for the purpose of efficient search and retrieval. Chunking breaks each document into smaller pieces so the vector database can store and retrieve relevant sections. [Here](https://medium.com/@jagadeesan.ganesh/understanding-chunking-algorithms-and-overlapping-techniques-in-natural-language-processing-df7b2c7183b2) is a great medium article talking about the main purpose and different ways of chunking for more context, in case you're interested. 
<!--Chunking is commonly used in all modern RAG implementations. Chunking is useful in cases of large documents in a corpus and addresses the shortcoming of the keyword RAG approach in the previous lesson wherein the entire document was retrieved as context for a query. This also helps with reducing token usage.--> 
The function below is an implementation of chunking. This is the "Split Text Into Chunks" step in the figure, although we do not use LangChain here.

```python
def simple_chunks(text: str, chunk_size=900, overlap=150):
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        chunk = text[i:i+chunk_size]
        end = chunk.rfind(". ")
        if end > int(chunk_size * 0.6):
            chunk = chunk[:end+1]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, len(chunk) - overlap)
    return chunks

docs = []
for d in corpus:
    for i, ch in enumerate(simple_chunks(d["text"])):
        docs.append({"doc_id": d["doc_id"], "chunk_id": i, "text": ch})
print("Total chunks:", len(docs))
```
The function creates chunks of a specifiable size (here the default value is 900 characters) with some overlap between subsequent chunks (default value is 150 characters). <!-- The chunks are created by "sliding" a window of size `chunk_size` over the text in a document. The reason for some overlap between chunks is to preserve context within chunks (i.e. to reduce cases where the full context for a query is split between two chunks). In the below function, we additionally limit the size of chunks by stopping at periods (".") if they appear later than 60% of the chunk size (i.e. closer to the end of the chunk). Note that each document is "chunked" individually. Usually, `chunk_size` and `overlap` are hyperparameters that you can tune to improve the quality of responses of the RAG framework. -->
Based on the default values chosen, a total of 930 chunks are obtained across all documents, thus the output of the code is as below.

```
Total chunks: 930
```

Now that we have chunked our documents, we will get into the key ideas behind semantic RAG - starting with embeddings.

### From text to embeddings

Next, the text chunks are converted to vector embeddings. This is a key feature in modern RAG implementations. A vector embedding is essentially a list of numbers that uniquely represents a piece of text, audio, image or any other form of data. [Here](https://www.ibm.com/think/topics/embedding) is a great IBM article talking about embeddings in general and their uses. We've also discussed embeddings in the [Intro to AI lesson](../05_AI_intro/01_intro_nlp_llms.md). The embedding model used in this example is OpenAI's "text-embedding-3-small." You can learn more about it in the documentation [here](https://platform.openai.com/docs/models/text-embedding-3-small). 
<!-- The reason we use embeddings instead of text in LLMs and RAG is that we can perform efficient search operations that account for nuances in context that would be very difficult or even impossible in the text space. -->

<!-- There are many embedding models out there that map text to vectors. The model has been trained on the corpus of text in the internet. -->
The function below converts the text chunks into vector embeddings and stores them in a vector database. This corresponds to the "Generate Embeddings" and "Data Indexing" steps in the figure, although we do not use SBERT here.

```python
def embed_texts(texts, model="text-embedding-3-small", batch_size=128):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([np.array(e.embedding, dtype="float32") for e in resp.data])
    return np.vstack(vecs)

chunk_texts = [d["text"] for d in docs]
emb = embed_texts(chunk_texts)
dim = emb.shape[1]
print(emb.shape)
```
The function collects the chunks into batches (of default size 128) and queries the embedding model to get the corresponding embedding vectors for each chunk. <!-- It must be noted that changing the batch size does not change the corresponding embedding for a given chunk. A higher batch size allows for more efficient conversion into embeddings, upto a certain limit. Refer to the documentation to know more. --> 
Every text chunk is converted to a vector of float values of size 1536 by the embedding model. Since there are 930 chunks, the output of the code will be as shown below.

```
(930, 1536)
```

So far, we have extracted the text from the documents into a corpus, broken the text down into chunks, and mapped the chunk text onto the embeddings space. The next step is to do semantic search in this embeddings space for context retrieval in response to a query.

<!-- ### FAISS and determining similarity

As mentioned earlier, moving from text to embedding space allows for a more efficient search to find semantically similar to a given text. [FAISS](https://faiss.ai/index.html) (Facebook AI Similarity Search) is a library developed by Facebook AI Research to enable similarity search across different embeddings spaces. Its primarily written in C++, but has Python wrappers allowing it to be used in Python projects but with faster compilation. Recall that embeddings can represent anything from text to audio, images or video. FAISS can be used to search across any of these embeddings spaces.

 To demonstrate how te embeddings space can be used for semantic similarity search, observe the plot below. This plot represents the embedding values for different words, reduced to two dimensions using Principal Component Analysis (PCA, which you should be familiar with from your ML lessons).

![Embeddings Semantic Similarity](resources/openai_embeddings_similarity.png)

Here, the actual numbers are not as important as the relative positions of the words. You will see that semantically similar words like "uncle" and "aunt", "cat" and "dog" are close together in the reduced embeddings space. Also, words like "policy", "goals" and "mission" are close together with "company" being nearby. It is important to note that this semantic proximity happens not just for words, but also for multiple sentences. This proximity is what allows the use of similarity metrics to retrieve the appropriate context for a given query. 

There are many metrics you can use to ascertain semantic similarity between two embeddings. For more information on the different search metrics in FAISS, have a look at [this](https://medium.com/walmartglobaltech/transforming-text-classification-with-semantic-search-techniques-faiss-c413f133d0e2) medium article. In this implementation, we use cosine similarity. 

Cosine similarity $c$ between two vectors $A$ and $B$ is calculated as follows:
$$c(A,B) = \frac{A.B}{||A||.||B||}$$
$A.B$ is the inner (dot) product between $A$ and $B$ and $||A||$ and $||B||$ are the magnitudes of the two vectors. For our case, $A$ and $B$ are two embeddings representing two pieces of text. The cosine similarity score varies from -1 to 1, with 1 implying high degree of similarity. So from the previous example, "uncle" and "aunt" will have a higher cosine similarity score than "uncle" and "company." [This](https://www.ibm.com/think/topics/cosine-similarity) IBM article discusses cosine similarity and compares it with other metrics. 

Now that we've gone over how cosine similarity between text embeddings is calculated, lets look at how cosine similarity in FAISS is used for context retrieval given a query. -->

### Semantic Retrieval

In this semantic RAG implementation, the idea is to retrieve relevant chunks and augment them to the query to produce the response from the LLM. We use cosine similarity to retrieve the most relevant chunks to the query. The higher the similarity score, the more relevant the chunk is to the query. Learn more about cosine similarity from [this](https://www.ibm.com/think/topics/cosine-similarity) IBM article if you're interested. We use [FAISS](https://faiss.ai/index.html) (Facebook AI Similarity Search) to search through the database of chunk embeddings. The code snippet below goes over the process of creating an index (a database of embeddings) in FAISS and searches this index using the cosine similarity to retrieve the most relevant chunks to the given query. This corresponds to the "Generate Embeddings" for the query, "Pass Query", "Retrieval Similarity", and "Pass Chunks" steps in the figure.

<!-- Firstly, all embeddings are normalized to have a magnitude of 1 (using `normalize_L2`). This is done so that similarity only depends on the relative positions of the text and not absolute positions. Next, the index that uses cosine similarity is created using `IndexFlatIP` (IP stands for Inner Product). Since all embeddings are normalized, the inner product between two embeddings is equal to the cosine similarity score. Then, the embeddings for the text chunks from our documents is added to this index. Since there are 930 chunks, the output of the print statement is as below.
```
FAISS index ready. Vectors: 930
``` -->

Before we build the FAISS index, it's worth pausing to focus on what an index means in the context of RAG pipelines. An index is a vector store that holds our embeddings and is optimized for fast similarity search over those embeddings. The term is used loosely -- sometimes it refers to storage, sometimes to search, and sometimes to both at once.

In our case, FAISS gives us a simple, in-memory way to store embeddings and retrieve the most similar ones, which corresponds to the "Data Indexing" and "Retrieval Similarity" steps in the figure above. FAISS is not a full SQL-based database like the pgvector setup we will see later. Instead, it provides a temporary (ephemeral) vector storage solution that exists only while your script is running. This lets us keep the focus on understanding the retrieval workflow without needing to set up a database server

```python
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(dim)
index.add(emb)
print("FAISS index ready. Vectors:", index.ntotal)
id2meta = {i: docs[i] for i in range(len(docs))}
```

Now that we've created the FAISS index, we define a function (`retrieve`) that retrieves the relevant context from this index given the query. 

```python
def retrieve(query, k=3, model="text-embedding-3-small"):
    q_emb = embed_texts([query], model=model)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        meta = id2meta[int(idx)]
        hits.append({"score": float(score), **meta})
    return hits

for h in retrieve("What is BrightLeaf Solar's mission?", k=3):
    print(round(h["score"], 3), h["doc_id"], f"chunk {h['chunk_id']}", f"text: {h['text']}")
```
Once the FAISS index is created, the given query is converted into an embedding using the embedding model and then input to the FAISS index's search function. The search function computes the cosine similarity between the query embedding and the chunk embeddings in the index and returns the `k` highest scoring (most relevant) chunks. Here `k` is set to a default of 3. The 3 highest scoring chunks are appended into the `hits` list and will be used as retireved context to be augmented with the query to generate the response.

As an example, we look at the retrieved chunks for the query "What is Brightleaf Solar's mission?" and their corresponding cosine similarity scores. The output of the print calls is shown below.

<code>
0.828 mission_statement.pdf chunk 0 text: Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our mission is to make solar power practical, affordable, and accessible to communities that have historically been left behind in the transition to clean energy. We are not only engineers and builders, but educators, partners, and advocates for a more resilient and equitable power grid. Every installation represents more than energy savings-it is an investment in long-term community well-being. Expanded Vision The company began its journey in the southeastern United States, where energy poverty and infrastructure challenges remain significant barriers to renewable adoption.<br><br>
0.734 mission_statement.pdf chunk 5 text: ransparent financing, and community ownership. We aim to ensure that the economic benefits of solar generation remain within the communities we serve. By 2030, BrightLeaf's goal is to help 100,000 households transition to renewable energy-not through isolated projects, but through a national network of resilient, community-led power systems.<br> <br>
0.703 employee_benefits.pdf chunk 68 text: ation, BrightLeaf ensures its workforce evolves alongside the renewable energy sector.
</code>

The first thing to note is that each of the chunks is of different size. This is because the chunks stop at periods if they occur at more than 60% of the way to full size. The second and third chunks start randomly from partially completed words. This is a byproduct of the chunking process and further modifications can be made to improve the understandability of each chunk. The third chunk is at the end of the employee_benefits.pdf file. Hence its small size. Although it is not really related to the query, it is still chosen because of its high similarity with the query and the fact that we ask for the top 3 chunks. The important observation here is that the first two chunks pertain **exactly** to the query, demonstrating the power of using semantic similarity based search in the embeddings space. Also note that in comparison to the keyword RAG from the previous lesson where the entire mission_statement.pdf text would be used as retrieved context for this query, basic semantic RAG would only use these chunks as retrieved context. This would result in fewer used tokens to query the LLM while getting a good response to the query. 

Once we retrieve the relevant context, all that is left to do is to augment the context with the query and input the modified prompt to the LLM.

### Augment and Generate

All of the previous steps described are part of the "Retrieve" step in the RAG process. The function that "Augments" the retrieved context with the user query and "Generates" the response is shown below. This corresponds to the "Generate Answer" step in the figure.

```python
def ask_llm(query, contexts, use_rag, model="gpt-4o-mini", temperature=0.2, max_chars=3500):
    ctx = "\n\n---\n\n".join(c["text"] for c in contexts)
    ctx = ctx[:max_chars]
    if use_rag:
        prompt = (
            "Use ONLY the provided context to answer the question.\n"
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        )
    else:
        prompt = query
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
```

This `ask_llm` function is similar to the corresponding function in the Keyword RAG implementation from the previous lesson, but there are some modifications. <!-- The retrieved chunks are combined into a single string with a maximum characters limitation. This is included to reduce token usage when querying the LLM. Now the instructions clearly state that the LLM must use only the provided context and say it does not know if the context is not pertinent to the query. This reduces the chances of hallucinations by the LLM. The additional `temperature` parameter is also provided to the model's completions method to allow for variability in the responses. -->
Note the modified prompt is also different from the Keyword RAG lesson. As before, the `use_rag` flag is provided to toggle between test cases with and without RAG's context.

<!-- Although this implementation incorporates some complex concepts like chunking, embeddings, and sementic similarity, it is still rather naive and basic. The mechanism to chunk documents is simplistic. It doesn't leverage additional contextual similarity that comes from the domain of application (the fact that the LLM must answer questions purely about the company). The model used is more generalized and some words that are similar in the corporate world may not be close in the embeddings space as captured by the OpenAI embeddings model. There have been more advanced embedding representations of text, more nuanced approaches that leverage the domain of application, and more efficient ways of searching across the embeddings developed over the years. Nonetheless, it is still powerful. -->
To demonstrate the power of this semantic RAG framework, lets see it in action through an example.

### Example - Semantic RAG in action
The code snippet below is used to test this naive semantic RAG implementation. 
The test query is "When did BrightLeaf partner with SunSpan and what did they focus on?"

```python
q = "When did BrightLeaf partner with SunSpan and what did they focus on?"
ctx = retrieve(q, k=3)
for c in ctx:
    print("Context score:", c["score"])
    print("Context doc:", c["doc_id"])
    print("Context chunk:", c["chunk_id"])
    print("Context text:", c["text"])
    print()
print(ask_llm(q, ctx, use_rag=False))
```
The `retrieve` method is run to retrieve the top 3 chunks relevant to the query. The three chunks, along with the document and chunk IDs and the similarity scores are printed. The output of those print calls is shown below. 

<code>
Context score: 0.7309314012527466
Context doc: partnerships.pdf
Context chunk: 1
Context text: . This partnership targets legacy industrial zones in Ohio and Michigan, pairing solar and wind power systems with SunSpan's real-time grid analytics. The collaboration has already produced measurable results, with the first hybrid sites showing 18 percent improved grid stability and 30 percent lower emissions. SunSpan and BrightLeaf also co-authored a white paper on the economic benefits of decarbonizing regional manufacturing supply chains. Academic and Workforce Collaborations Beyond corporate partnerships, BrightLeaf works closely with universities and community colleges to train the next generation of clean-energy professionals. The Bright Scholars Initiative offers scholarships and internships for students in sustainability, electrical engineering, and environmental data analytics. Collaboration with technical schools ensures a pipeline of skilled labor for new projects.<br><br>

Context score: 0.713233470916748
Context doc: partnerships.pdf
Context chunk: 0
Context text: EcoVolt Energy (2022 Partnership) BrightLeaf's collaboration with EcoVolt Energy, established in 2022, focused on delivering microgrid solutions to rural communities in Georgia and South Carolina. The initiative combined BrightLeaf's solar generation systems with EcoVolt's battery storage expertise. Together, they launched five pilot sites that reduced community energy costs by an average of 25 percent. The partnership also created 40 permanent local jobs and helped utilities in the region study the integration of renewable storage into traditional grids. SunSpan Microgrids (2025 Partnership) In 2025, BrightLeaf joined forces with SunSpan Microgrids to develop hybrid renewable infrastructure for the Midwest. This partnership targets legacy industrial zones in Ohio and Michigan, pairing solar and wind power systems with SunSpan's real-time grid analytics.

Context score: 0.6759465336799622
Context doc: mission_statement.pdf
Context chunk: 0
Context text: Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our mission is to make solar power practical, affordable, and accessible to communities that have historically been left behind in the transition to clean energy. We are not only engineers and builders, but educators, partners, and advocates for a more resilient and equitable power grid. Every installation represents more than energy savings-it is an investment in long-term community well-being. Expanded Vision The company began its journey in the southeastern United States, where energy poverty and infrastructure challenges remain significant barriers to renewable adoption.
</code>

It can be seen that the first two retrieved chunks are **exactly** relevant to the query. They are part of the same initial section of the partnerships.pdf document. In fact, the overlapping text from chunk 0 to chunk 1 can be seen as well. Chunk 0 directly answers the "when" part of the query and chunk 1 directly talks about the partnership with SunSpan and so they have the highest cosine similarities. The third chunk talks about the general mission of Brightleaf Solar, which is somewhat relevant to the nature of the collaboration between Brightlead and SunSpan. But it is not directly related to the query. 

Initially, we set `use_rag=False` and we see confident hallucination from the LLM as seen in the printed response below.

<code>
BrightLeaf partnered with SunSpan in January 2022. The partnership focused on enhancing the development of solar energy projects, specifically by leveraging BrightLeaf's data extraction and processing capabilities to streamline project development and improve efficiency in the solar energy sector.
</code>

Not only does it get the month and year of the collaboration wrong but also the nature of the collaboration. This is to be expected since the LLM has never been trained on the information of this fictitious company Brightleaf Solar. 

Now, when we set `use_rag=True` and look at the response, the difference is light and day as seen below.

<code>
BrightLeaf partnered with SunSpan in 2025, focusing on developing hybrid renewable infrastructure for the Midwest, specifically targeting legacy industrial zones in Ohio and Michigan by pairing solar and wind power systems with SunSpan's real-time grid analytics.
</code>

It gets the month and year of the collaboration right and also the nature and geographical region of the collaboration. So the retrieved context was appropriate to the query and the LLM used the retrieved context appropriately. 

Congratulations!! You've now gone through a more complex RAG framework that utilizes embeddings and semantic similarity based search. You're now equipped with the basic tools to understand the more modern RAG frameworks utilized by industry today. Feel free to play with the hyperparameters and observe the response.

In this basic implementation, the chunking of documents and storing into the FAISS index is rather crude. This is fine as there are few documents to work with. However, in industry companies have to deal with a large database of large documents and online data repositories that are consistently being modified. The approach used here is simply not sufficient to handle the large quantities of data in industry. In the next section, we will look at the use of pgvector, postgres and docker as a means to create, maintain, and query from an online, efficiently vectorized, and easily reproducible database.

### Working with an online database

Previously, you built a small semantic RAG system using OpenAI embeddings and  FAISS as a local in-memory vector store. In this notebook, you will rebuild the same basic pipeline using a **real database**, using the following:

- PostgreSQL running in Docker
- The `pgvector` extension for vector similarity search

So far, our RAG systems have stored embeddings in memory. This works for small demos, but becomes impractical as soon as you want to build something real. A database like PostgreSQL with the pgvector extension solves several problems that in-memory systems cannot. Here are the five reasons production teams switch to **PostgreSQL + pgvector**:

1. An in-memory vector store disappears when the notebook stops running. That means every time you restart your environment, you need to re-embed all your documents from scratch. If embeddings come from a paid API, this becomes expensive very quickly. A real vector database stores embeddings permanently, so you only pay for embedding once and reuse them forever. 

2. In-memory stores do not scale. They work for a few dozen or a few hundred vectors, but real applications often need tens of thousands or millions of embeddings. A database is built for this. It handles large datasets, indexes them efficiently, and gives fast retrieval even as your collection grows.

3. A database allows multiple tools (or multiple people) to access the same stored embeddings. With an in-memory store, everything lives inside one Python process. With PostgreSQL, the vectors live in a shared, durable service that any script or application can connect to.

4. Databases provide important features that ordinary in-memory structures lack: durability, reliability, backups, transactions, authentication, and predictable performance. These are the things production systems rely on and the reason databases have been industry workhorses for decades.

5. pgvector is a PostgreSQL extension designed specifically for vector search. It adds the ability to store and query embeddings directly inside the database, using indexes that support efficient similarity search. This combines the familiarity of SQL with the power of vector retrieval in a single place.

To summarize, a real database gives us persistence, scale, reliability, shared access, and cost savings by avoiding unnecessary re-embedding.

<!-- Up to now our RAG demos have kept everything in memory (a FAISS index living inside one Python process). That's great for learning, but it breaks the moment you want a real application. Here are the five reasons production teams switch to **PostgreSQL + pgvector**:

1. The biggest problem is persistence: if the notebook restarts, you lose everything and have to regenerate all embeddings again. That costs money and time. A real system needs its data to survive restarts, crashes, and deployments, and PostgreSQL with pgvector gives us that durability.

2. Another limitation with in-memory vector stores is scale. An in-memory store is limited by your local machine's RAM, which usually means tens or maybe hundreds of thousands of embeddings at most. That is nowhere close to what production systems require. A PostgreSQL database, on the other hand, can store millions or billions of embeddings.

3. Using a database also allows multiple applications to access the same vector store. In-memory vectors only exist inside the Python process that created them, so a Streamlit app, backend API, or teammate on another machine cannot see your data. But when the vectors live in a PostgreSQL db server, they become a shared resource: your API, notebook, and background jobs can all read the same embeddings.

4. Traditional in-memory stores also provide no reliability features. There are no backups, no permission systems, no transactions, and no ability to restore your data if something goes wrong. PostgreSQL gives you all of the reliability and features of a real database.

In sum, in memory vector stores are great for learning some basic RAG features, but PostgreSQL + pgvector is the default choice for 90 % of real-world RAG applications. --> 

### A quick primer on Docker, PostgreSQL, and pgvector

[Docker](https://www.docker.com/) is an industry-standard tool that runs applications inside containers. A container is a self-contained environment that includes everything the application needs to run: the operating system, libraries, configuration, and the application itself. Because of this, a container behaves the same on Windows, macOS, and Linux.

[PostgreSQL](https://www.postgresql.org/) is an open-source object-relational database management system, developed at the UC Berkeley Computer Science department. It builds on POSTGRES and supports a variety of complex SQL queries. We will use PostgreSQL to create and query the database in the docker container.

[pgvector](https://github.com/pgvector/pgvector) is a PostgreSQL extension that adds first-class support for vector data. Normally, relational databases are great at storing tables, text, numbers, and JSON, but they have no idea what to do with embeddings. Without pgvector, you would have to store embeddings as arrays or blobs and write your own similarity-calculation code. pgvector solves this by giving PostgreSQL a new data type called VECTOR, along with built-in functions for similarity search. That means you can store embeddings in the database exactly the same way you store other data, and then run queries like "find the 3 closest embeddings to this query vector" directly inside SQL. 

The main benefit is that you get a real database as your vector store: instead of running a separate specialized vector engine, you can keep everything inside one well-vetted SQL database system. This is why pgvector has become one of the most widely used vector-store backends in production RAG systems.

To use `pgvector`, we need a PostgreSQL database with the `pgvector` extension installed.

We could try to install PostgreSQL and pgvector separately on every local machine, but that would look different on each operating system and would take a lot of time to debug. Instead, we use Docker to run everything in a clean, isolated environment that works the same everywhere.

> **Note:** We will use psycopg2-binary in this demo. Psycopg2 is the standard PostgreSQL driver for Python. It is the library that lets Python code connect to a PostgreSQL database and do standard CRUD operations. It allows you to open a connection, run SQL queries, fetch results, insert/update rows, etc.

### Setting up the database on Docker

For our purposes, Docker has one job in this lesson: run a PostgreSQL server that already has the pgvector extension installed. 

To install Docker, follow the instructions for:

- [Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
- [Windows](https://docs.docker.com/desktop/setup/install/windows-install/)
On Linux 

> **Important:** On Windows you must enable WSL2  during installation (it is the default). Without WSL2, Docker won’t run containers properly. (WSL is the Windows Subsystem for Linux, which allows you to run Linux on Windows).

If you are on Linux, you don't need Docker desktop, you can just set up the Docker engine to interface in your cli [as discussed here](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04).

In Windows you need to start the Docker engine for the next step. This merely means starting Docker Desktop.

In your local machine, set up a folder structure like:

```text
docker_files/
  pgvector_store/
    docker-compose.yml
```

The `docker-compose.yml` file should look like this:

```yaml
version: "3.9"
services:
  pgvector:
    image: ankane/pgvector
    environment:
      POSTGRES_USER: ctd
      POSTGRES_PASSWORD: ctdpassword
      POSTGRES_DB: ctd_rag
    ports:
      - "5432:5432"
```

This configuration tells Docker to:

- Download and run the `ankane/pgvector` image from Docker Hub
- Start a PostgreSQL server with:
  - username: `ctd`
  - password: `ctdpassword`
  - database: `ctd_rag`
- Expose the database on `localhost:5432`

To start the database, open a terminal in `docker_files/pgvector_store/` and run:

```bash
docker compose up -d
```

You should only need to do this once per session. If you restart your computer or stop the container, you may need to run it again.

Once that finishes, within Docker Desktop (in Windows) you should see something similar to the image below.

![Docker screenshot](resources\docker_desktop_running.png)

Now, we set up our OpenAI API client as usual and setup a connection to the docker container using postgreSQL.

```python
import os, re, math, json
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2

# Load .env (expects OPENAI_API_KEY)
if load_dotenv():
    print("Loaded OPENAI_API_KEY from .env")
else:
    print("No API key loaded; check your .env file.")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# BrightLeaf PDF directory (same as in 02_semantic_rag.ipynb)
DATA_DIR = Path("brightleaf_pdfs")
assert DATA_DIR.exists(), f"{DATA_DIR} not found. Put BrightLeaf PDFs there."

# Connection string for the Dockerized Postgres + pgvector database
PG_CONN_STR = "postgresql://ctd:ctdpassword@localhost:5432/ctd_rag"

def get_connection():
    """Create a new connection to the Postgres database."""
    return psycopg2.connect(PG_CONN_STR)
```

The code up until `PG_CONN_STR` is the same as the semantic RAG implementation from earlier. Note that `PG_CONN_STR` contains the username, password, and port that was set in the `yaml` file earlier. `psycopg2.connect(PG_CONN_STR)` connects to the postgres database `ctd_rag` in the docker container.

The loading, chunking and embedding of the text in the documents(i.e. the "text extraction," "split text into chunks," and "generate embeddings" steps in the figure at the beginning of the lesson) is the same as the custom semantic RAG implementation as seen previously. Next, we load the chunks onto the docker container database. To do this, we first initialize the SQL database in the docker container.

### Initialize SQL table in docker container

After we connect to the Postgres server running in Docker, the code below is used to:

- Enable the `pgvector` extension
- Create a `rag_chunks` SQL table with the columns:
  - `doc_id`
  - `chunk_id`
  - `text`
  - `embedding` (a `vector(dim)` column)

We will use the `dim` measured from the embeddings (refer to the "from text to embeddings" section above).

```python
with get_connection() as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Optional: clear out any old data to avoid duplicate inserts while developing
        cur.execute(
            "DROP TABLE IF EXISTS rag_chunks;"
        )

        # Create table for document chunks + embeddings
        cur.execute(
            f"""
            CREATE TABLE rag_chunks (
                id SERIAL PRIMARY KEY,
                doc_id TEXT,
                chunk_id INT,
                text TEXT,
                embedding vector({dim})
            );
            """
        )

print("Database and rag_chunks table are ready.")
```

The `CREATE EXTENSION IF NOT EXISTS` query can be thought of as an import statement to enable pgvector. We then pass the standard SQL commands through the connection cursor to the Postgres database to create the `rag_chunks` table (after deleting any previously created duplicates). This table will house the embedded chunks from the documents. Note that once we enable pgvector, we have access to the `vector` datatype which we use to store the vector embeddings of the chunks. 

Next, we insert the chunks into the created table.

### Insert chunks into table

We now insert every chunk and its embedding into the `rag_chunks` table. This is the "data indexing" step in the figure at the beginning of the lesson.

Each row is associated with a chunk and has:
- `doc_id`: PDF file name
- `chunk_id`: chunk index within that document
- `text`: the chunk text
- `embedding`: the vector for that chunk

```python
with get_connection() as conn:
    with conn.cursor() as cur:
        for meta, vec in zip(docs, emb):
            cur.execute(
                """
                INSERT INTO rag_chunks (doc_id, chunk_id, text, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    meta["doc_id"],
                    meta["chunk_id"],
                    meta["text"],
                    vec.astype(float).tolist(),  # convert to plain Python floats
                ),
            )

print("Inserted", len(docs), "rows into rag_chunks.")
```

In the code above, we iterate through the chunks (`docs`) and their corresponding embeddings (`emb`), passing the document ID (`doc_id`), chunk ID (`chunk_id`), chunk text (`text`), and the chunk vector embedding (`vec`) to the `rag_chunks` table. Note that we don't need to pass a value of `id` to the table since it is designated as `SERIAL` and `PRIMARY KEY`, meaning it is self-incrementing (serial) and it must be unique and non-null (primary key).

> **Important**: We only do this embedding step ONCE. In a real RAG system, you never want to re-embed your entire document collection every time you run a notebook or restart your app. That would waste money (thousands of unnecessary embedding API calls), waste time (minutes to hours of preprocessing), make the system slow to start. By storing all embeddings permanently in Postgres + pgvector, we pay the cost ONE TIME, and from then on all queries are cheap, all retrievals are instant, nothing needs to be rebuilt. This is why real production RAG systems always use a persistent vector store. RAG isn’t cheap unless your vector store is persistent.

### 7. Semantic search with pgvector

Now that the database holds all of our chunks and embeddings, we can:
- Embed a query
- Use the `<->` operator in SQL to compute vector distances. This is a special operator for pgvector for computing distances between embedding vectors. This is what makes pgvector so powerful. The pgvector [github page](https://github.com/pgvector/pgvector) lists all distance measures you can use as similarity metrics.
- Retrieve the top `k` most similar chunks

<!-- - `<->` - L2 distance
- `<#>` - (negative) inner product
- `<=>` - cosine distance
- `<+>` - L1 distance
- `<~>` - Hamming distance (for binary vectors)
- `<%>` - Jaccard distance (for binary vectors) -->

Before we perform semantic search on the SQL database, we must first convert the query to the vector embedding. This is the "generate embeddings" step for the user query in the figure at the beginning of the lesson. The function below does this.
```python
def embed_query(query_text: str) -> np.ndarray:
    """Create an embedding for a single query string."""
    q_vec = embed_texts([query_text])
    return q_vec[0]
```

The function below searches the database for the top `k` most relevant chunks to the query. The function below performs the "pass query" and "retrieve similarity" steps in the figure at the beginning of the lesson.

```python
def search_pgvector(query_text: str, k: int = 5):
    q_emb = embed_query(query_text)
    q_vec = q_emb.astype(float).tolist()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, chunk_id, text,
                       embedding <-> %s::vector AS distance
                FROM rag_chunks
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """,
                (q_vec, q_vec, k),
            )
            rows = cur.fetchall()

    return [
        {
            "doc_id": r[0],
            "chunk_id": r[1],
            "text": r[2],
            "distance": float(r[3]),
        }
        for r in rows
    ]
```
Once the query embedding is obtained, the SQL query is called to retrieve the top `k` chunks from the `rag_chunks` database that have the lowest distance to the query embedding (`embedding <-> %s::vector`). Note that since we're using distance as the similarity metric, smaller distance implies greater relevance of the chunk to the query. If you were to use cosine similarity (like in the semantic RAG implementation before this), you would retrieve the chunks in descending order of `embedding <-> %s::vector`. So the SQL query would be modified to have `ORDER BY embedding <-> %s::vector DESC`.

Let's see the SQL based semantic search in action through a quick test.

#### Quick Test

The test query is "When did BrightLeaf partner with SunSpan?" and we want the LLM to base its response on the top 3 retrieved context chunks. The code snippet below feeds this to the `search_pgvector()` method and prints the retrieved chunks one at a time.

```python
# Quick test to make sure our function works
test_results = search_pgvector("When did BrightLeaf partner with SunSpan?", k=3)
for r in test_results:
    print(f"[{r['doc_id']} chunk {r['chunk_id']}] dist={r['distance']:.4f}")
    print(r["text"], "...\n")
```
The output looks like below:

<code>
[partnerships.pdf chunk 1] dist=0.7433
. This partnership targets legacy industrial zones in Ohio and Michigan, pairing solar and wind power systems with SunSpan's real-time grid analytics. The collaboration has already produced measurable results, with the first hybrid sites showing 18 percent improved grid stability and 30 percent lower emissions. SunSpan and BrightLeaf also co-authored a white paper on the economic benefits of decarbonizing regional manufacturing supply chains. Academic and Workforce Collaborations Beyond corporate partnerships, BrightLeaf works closely with universities and community colleges to train the next generation of clean-energy professionals. The Bright Scholars Initiative offers scholarships and internships for students in sustainability, electrical engineering, and environmental data analytics. Collaboration with technical schools ensures a pipeline of skilled labor for new projects. ... <br>

[partnerships.pdf chunk 0] dist=0.7587
EcoVolt Energy (2022 Partnership) BrightLeaf's collaboration with EcoVolt Energy, established in 2022, focused on delivering microgrid solutions to rural communities in Georgia and South Carolina. The initiative combined BrightLeaf's solar generation systems with EcoVolt's battery storage expertise. Together, they launched five pilot sites that reduced community energy costs by an average of 25 percent. The partnership also created 40 permanent local jobs and helped utilities in the region study the integration of renewable storage into traditional grids. SunSpan Microgrids (2025 Partnership) In 2025, BrightLeaf joined forces with SunSpan Microgrids to develop hybrid renewable infrastructure for the Midwest. This partnership targets legacy industrial zones in Ohio and Michigan, pairing solar and wind power systems with SunSpan's real-time grid analytics. ... <br>

[mission_statement.pdf chunk 0] dist=0.8248
Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our mission is to make solar power practical, affordable, and accessible to communities that have historically been left behind in the transition to clean energy. We are not only engineers and builders, but educators, partners, and advocates for a more resilient and equitable power grid. Every installation represents more than energy savings-it is an investment in long-term community well-being. Expanded Vision The company began its journey in the southeastern United States, where energy poverty and infrastructure challenges remain significant barriers to renewable adoption. ...
</code>

When you compare with the "Example - Semantic RAG in action" section which uses the same query and is run to retrieve the same number of context chunks, you will notice that the exact same chunks are retrieved inspite of the pgvector search function using distance instead of cosine similarity. Generally the choice of similarity metric should not affect the retrieval process, but in rare circumstances this can happen. Additionally, each chunk contains relevant information to the query which signifies that the retrieval process is working properly. 

Now we have all the pieces to plug in the postgreSQL + pgvector database into the RAG pipeline.

### Using pgvector in the RAG pipeline

The `answer_with_pgvector` function combines the pgvector based retrieval process with the "pass chunks" and "generate answer" steps in the figure at the beginning of the lesson.

```python
def answer_with_pgvector(question: str, k: int = 5) -> str:
    """Run semantic search in Postgres and feed the top k chunks to the LLM."""
    hits = search_pgvector(question, k=k)
    answer = ask_llm(question, hits)
    return answer
```

The `ask_llm` function is the same as the one in the "Augment and Generate" section in this lesson, key difference is it uses the context retrieved using pgvector instead of the FAISS-based semantic search approach used before.

Let's look at the full RAG pipeline in action through an example.

### Example - pgvector-based semantic RAG in action

We use the same query as the previous example and run it to retrieve the 3 most relevant chunks as before. The `search_pgvector` method is run to check the retrieved context documents. The `answer_with_pgvector` generates the response based on the pgvector based RAG pipeline. 

```python
q = "When did BrightLeaf partner with SunSpan and what did they focus on?"
ctx_hits = search_pgvector(q, k=3)
print("Context docs:", [c["doc_id"] for c in ctx_hits])
print()
print(answer_with_pgvector(q, k=3))
```

The output of the code snippet above looks like below:
```text
Context docs: ['partnerships.pdf', 'partnerships.pdf', 'mission_statement.pdf']

BrightLeaf partnered with SunSpan in 2025, focusing on developing hybrid renewable infrastructure for the Midwest, specifically targeting legacy industrial zones in Ohio and Michigan by pairing solar and wind power systems with SunSpan's real-time grid analytics.
```

Like before, the top 3 most relevant retrieved chunks come from the "partnerships.pdf" and "mission_statement.pdf" documents. The generated response also answers the query in an accurate, concise manner.

Congratulations!! We now have a persistent database that we can query, we don't have to re-index every time we start this notebook, it will persist. Others can access it as long as they have the db params (the address, password, etc). To test this, try two things: 
- Close the notebook/script file but keep the database running, reopen it (clear all outputs if applicable) and run only the code necessary to use the RAG pipeline to answer a query (not the code to read, chunk, embed, and add the text into the docker database)
- Stop the docker container, restart it and run only the code necessary to use the RAG pipeline to answer a query (not the code to read, chunk, embed, and add the text into the docker database)

This implementation had several lines of code, consisting of custom functions. The next lesson looks at a library developed by Meta called llama index that condenses a lot of the operations done here into single lines of code. Additionally, llama index also includes metrics to evaluate RAG frameworks. 

## Check for Understanding

### Question 1

In the smeantic RAG implementation using the in-memory vector store shown in this lesson, on which metric is the top `k` most relevant chunks to the query obtained?

Choices:
- A. Cosine similarity using FAISS
- B. L2 similarity using FAISS
- C. L1 similarity using FAISS
- D. Keyword matching using FAISS


<details>
<summary> View Answer </summary>
<strong>Answer:</strong> A. Cosine similarity using FAISS <br>
In our implementation of the in-memory semantic RAG, we use FAISS' internal storage index that searches for the most relevant document chunks using cosine similarity.
</details>

### Question 2

In our implementation of pgvector and PostgreSQL enabled semantic RAG, what role does the `ankane/pgvector` docker container play?

Choices:
- A. It serves as an external platform to host the SQL database with the chunk embeddings
- B. It has PostgreSQL and pgvector installed that can be used to retrieve relevant chunks from the SQL database
- C. Both A & B
- D. It contains the documents and code for the implementation


<details>
<summary> View Answer </summary>
<strong>Answer:</strong> C. Both A & B <br>
When you run the "ankane\pgvector" docker container, it has an installation of the pgvector library that we use to query the SQL database and retrieve chunks relevant to the user query. The docker container also provides a clean, isolated environment to create and query the SQL database of chunk embeddings, since it is essentially a mini linux OS that will have the same behavior irrespective of the system OS.
</details>



<!-- ### Beyond Semantic RAG - Modern RAG frameworks

Although this implementation is undeniably powerful, there have been many advances in RAG over the years. Since generative AI and LLMs have been such hot topics in research and industry over the last 5 years or so, there has been a lot of effort invested in developing ever more efficient RAG techniques. Check out [this](https://www.youtube.com/watch?v=sGvXO7CVwc0) youtube video and [this](https://wandb.ai/site/articles/rag-techniques/) article to learn more about recent advances in RAG. 

It is useful to note that the overwhelming effort in the development of RAG has been on the "Retrieve" stage since the "Augment" stage is more to do with the design of the augmented prompt that is fed to the LLM. Many recent efforts have focused on the methods of storing chunks (such as relational databases, heirarchical databases) and searching through the database of chunks (graph RAG). Some implementations modify the user prompt to remove unnecessary context and make it easier to retrieve truly relevant context. Alternatively, other implementations have another layer of checks to make sure the retrieved context is really relevant to the query. For this purpose, reranking of the retrieved contexts through different measures of relevance is most common. One may also have an additional classification step to retrieve chunks that are in the same domain as the user query. 

The above is just a *small* snapshot of the many RAG approaches being developed. This field is gaining more traction everyday which is fueling even more research efforts. In an ever evolving field like Generative AI, it is therefore key to keep yourself abreast of the latest ideas, models, and trends. -->