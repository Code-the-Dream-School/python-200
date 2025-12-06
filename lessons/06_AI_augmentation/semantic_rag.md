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

Lets look deeper into the implementation below. In order to focus on the details of the approach, this implementation only looks at how a query is answered instead of looking at a Q&A loop. 

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

The code snippet below loads the files from the directory and cleans the text before storing it in a dictionary.

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

This script is very similar to the one in the previous lesson. A new addition is the `assert` method used to make sure that the directory exists and the text from the files can be extracted and cleaned. Like before, PDFReader is used to extract the text from the pdf files. The cleaning process involves the use of the regular expression operator `re.sub()`. Its syntax is given below:
```python
re.sub(pattern, repl, string)
```
Basically, it replaces all occurences of `pattern` in `string` with `repl`.

As there are 6 documents in the directory, the output of the code snippet will look like the one below.

```
Loaded 6 docs: ['earnings_report.pdf', 'employee_benefits.pdf', 'mission_statement.pdf', 'partnerships.pdf', 'product_specs.pdf', 'security_policy.pdf']
```
Now that we have extracted the text from all documents, we start the process of chunking the documents into bite-sized portions for more efficient context retrieval.

### Breaking down document text - "Chunking"

Chunking is the process of breaking down text into smaller portions (or "chunks"), usually for the purpose of efficient search and retrieval. Chunking is commonly used in all modern RAG implementations. [Here](https://medium.com/@jagadeesan.ganesh/understanding-chunking-algorithms-and-overlapping-techniques-in-natural-language-processing-df7b2c7183b2) is a great medium article talking about the main purpose and different ways of chunking for more context. Chunking is useful in cases of large documents in a corpus and addresses the shortcoming of the keyword RAG approach in the previous lesson wherein the entire document was retrieved as context for a query. This also helps with reducing token usage. The function below is an implementation of the sliding window chunking approach.

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
The function creates chunks of a specifiable size (here the default value is 900 characters) with some overlap between subsequent chunks (default value is 150 characters). The chunks are created by "sliding" a window of size `chunk_size` over the text in a document. The reason for some overlap between chunks is to preserve context within chunks (i.e. to reduce cases where the full context for a query is split between two chunks). In the below function, we additionally limit the size of chunks by stopping at periods (".") if they appear later than 60% of the chunk size (i.e. closer to the end of the chunk). Note that each document is "chunked" individually. Usually, `chunk_size` and `overlap` are hyperparameters that you can tune to improve the quality of responses of the RAG framework. Based on the values chosen, a total of 930 chunks are obtained across all documents, thus the output of the code is as below.

```
Total chunks: 930
```

Now that we have chunked our documents, we will get into the key ideas behind semantic RAG - starting with embeddings.

### From text to embeddings

A key feature in modern RAG implementations is the use of embeddings to represent text. An embedding is essentially a list of numbers that uniquely represents a piece of text, audio, image or any other form of data. The reason we use embeddings instead of text in LLMs and RAG is that we can perform efficient search operations that account for nuances in context that would be very difficult or even impossible in the text space. [Here](https://www.ibm.com/think/topics/embedding) is a great IBM article talking about embeddings in general and their uses. 

There are many embedding models out there that map text to vectors. The one used in this example is OpenAI's "text-embedding-3-small." You can learn more about it in the documentation [here](https://platform.openai.com/docs/models/text-embedding-3-small). The model has been trained on the corpus of text in the internet. The function below converts the text in the chunks into embedding vectors.

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
The function collects the chunks into batches (of default size 128) and queries the embedding model to get the corresponding embedding vectors for each chunk. It must be noted that changing the batch size does not change the corresponding embedding for a given chunk. A higher batch size allows for more efficient conversion into embeddings, upto a certain limit. Refer to the documentation to know more. Every piece of text (irrespective of number of characters) is converted to a vector of float values of size 1536 by the model. Since there are 930 chunks, the output of the code will be as shown below.

```
(930, 1536)
```

So so far, we have extracted the text from the documents into a corpus, broken the text down into chunks, and mapped the chunk text onto the embeddings space. The next step is to do semantic search in this embeddings space for context retrieval in response to a query.

### FAISS and determining similarity

As mentioned earlier, moving from text to embedding space allows for a more efficient search to find semantically similar to a given text. [FAISS](https://faiss.ai/index.html) (Facebook AI Similarity Search) is a library developed by Facebook AI Research to enable similarity search across different embeddings spaces. Its primarily written in C++, but has Python wrappers allowing it to be used in Python projects but with faster compilation. Recall that embeddings can represent anything from text to audio, images or video. FAISS can be used to search across any of these embeddings spaces.

To demonstrate how te embeddings space can be used for semantic similarity search, observe the plot below. This plot represents the embedding values for different words, reduced to two dimensions using Principal Component Analysis (PCA, which you should be familiar with from your ML lessons).

![Embeddings Semantic Similarity](resources/openai_embeddings_similarity.png)

Here, the actual numbers are not as important as the relative positions of the words. You will see that semantically similar words like "uncle" and "aunt", "cat" and "dog" are close together in the reduced embeddings space. Also, words like "policy", "goals" and "mission" are close together with "company" being nearby. It is important to note that this semantic proximity happens not just for words, but also for multiple sentences. This proximity is what allows the use of similarity metrics to retrieve the appropriate context for a given query. 

There are many metrics you can use to ascertain semantic similarity between two embeddings. For more information on the different search metrics in FAISS, have a look at [this](https://medium.com/walmartglobaltech/transforming-text-classification-with-semantic-search-techniques-faiss-c413f133d0e2) medium article. In this implementation, we use cosine similarity. 

Cosine similarity $c$ between two vectors $A$ and $B$ is calculated as follows:
$$c(A,B) = \frac{A.B}{||A||.||B||}$$
$A.B$ is the inner (dot) product between $A$ and $B$ and $||A||$ and $||B||$ are the magnitudes of the two vectors. For our case, $A$ and $B$ are two embeddings representing two pieces of text. The cosine similarity score varies from -1 to 1, with 1 implying high degree of similarity. So from the previous example, "uncle" and "aunt" will have a higher cosine similarity score than "uncle" and "company." [This](https://www.ibm.com/think/topics/cosine-similarity) IBM article discusses cosine similarity and compares it with other metrics. 

Now that we've gone over how cosine similarity between text embeddings is calculated, lets look at how cosine similarity in FAISS is used for context retrieval given a query.

### Semantic Retrieval

The code snippet below goes over the process of creating an index (a database of embeddings) in FAISS that will use cosine similarity to search and retrieve the relevent context for a query.

```python
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(dim)
index.add(emb)
print("FAISS index ready. Vectors:", index.ntotal)
```
Firstly, all embeddings are normalized to have a magnitude of 1 (using `normalize_L2`). This is done so that similarity only depends on the relative positions of the text and not absolute positions. Next, the index that uses cosine similarity is created using `IndexFlatIP` (IP stands for Inner Product). Since all embeddings are normalized, the inner product between two embeddings is equal to the cosine similarity score. Then, the embeddings for the text chunks from our documents is added to this index. Since there are 930 chunks, the output of the print statement is as below.
```
FAISS index ready. Vectors: 930
```
Now that we have the database index of our embeddings, the function below searches this index using the cosine similarity to retrieve the most relevant chunks to the given query. 
```python
id2meta = {i: docs[i] for i in range(len(docs))}
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
First, the given query is converted into an embedding using the OpenAI embedding model used on the document chunks. The resulting embedding is normalized and then input to the FAISS index's search function. The search function essentially computes the cosine similarity between the query embedding and the chunk embeddings in the index and returns the similarity scores `D` and indices `I` of the `k` highest scoring chunks. Here `k` is set to a default of 3 but is generally a hyperparameter to tune. The 3 highest scoring chunks are appended into the `hits` list and will be used as retireved context to be augmented with the query to generate the response.

As an example, we look at the retrieved chunks for the query "What is Brightleaf Solar's mission?" and their corresponding similarity scores. The output of the print calls is shown below.

<code>
0.828 mission_statement.pdf chunk 0 text: Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our mission is to make solar power practical, affordable, and accessible to communities that have historically been left behind in the transition to clean energy. We are not only engineers and builders, but educators, partners, and advocates for a more resilient and equitable power grid. Every installation represents more than energy savings-it is an investment in long-term community well-being. Expanded Vision The company began its journey in the southeastern United States, where energy poverty and infrastructure challenges remain significant barriers to renewable adoption.<br><br>
0.734 mission_statement.pdf chunk 5 text: ransparent financing, and community ownership. We aim to ensure that the economic benefits of solar generation remain within the communities we serve. By 2030, BrightLeaf's goal is to help 100,000 households transition to renewable energy-not through isolated projects, but through a national network of resilient, community-led power systems.<br> <br>
0.703 employee_benefits.pdf chunk 68 text: ation, BrightLeaf ensures its workforce evolves alongside the renewable energy sector.
</code>

The first thing to note is that each of the chunks is of different size. This is because the chunks stop at periods if they occur at more than 60% of the way to full size. The second and third chunks start randomly from partially completed words. This is a byproduct of the chunking process and further modifications can be made to improve the understandability of each chunk. The third chunk is at the end of the employee_benefits.pdf file. Hence its small size. Although it is not really related to the query, it is still chosen because of its high similarity with the query and the fact that we ask for the top 3 chunks. The important observation here is that the first two chunks pertain **exactly** to the query, demonstrating the power of using semantic similarity based search in the embeddings space. Also note that in comparison to the keyword RAG from the previous lesson where the entire mission_statement.pdf text would be used as retrieved context for this query, basic semantic RAG would only use these chunks as retrieved context. This would result in fewer used tokens to query the LLM while getting a good response to the query. 

Once we retrieve the relevant context, all that is left to do is to augment the context with the query and input the modified prompt to the LLM.

### Augment and Generate

All of the previous steps described are part of the "Retrive" step in the RAG process. The function that "Augments" the retrieved context with the user query and "Generates" the response is shown below.

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

This `ask_llm` function is similar to the corresponding function in the Keyword RAG implementation from the previous lesson, but there are some modifications. The retrieved chunks are combined into a single string with a maximum characters limitation. This is included to reduce token usage when querying the LLM. The modified prompt is also different from the Keyword RAG lesson. Now the instructions clearly state that the LLM must use only the provided context and say it does not know if the context is not pertinent to the query. This reduces the chances of hallucinations by the LLM. The additional `temperature` parameter is also provided to the model's completions method to allow for variability in the responses. As before, the `use_rag` flag is provided to toggle between test cases with and without RAG's context.

Although this implementation incorporates some complex concepts like chunking, embeddings, and sementic similarity, it is still rather naive and basic. The mechanism to chunk documents is simplistic. It doesn't leverage additional contextual similarity that comes from the domain of application (the fact that the LLM must answer questions purely about the company). The model used is more generalized and some words that are similar in the corporate world may not be close in the embeddings space as captured by the OpenAI embeddings model. There have been more advanced embedding representations of text, more nuanced approaches that leverage the domain of application, and more efficient ways of searching across the embeddings developed over the years. Nonetheless, it is still powerful. To demonstrate the power of this semantic RAG framework, lets see it in action through an example.

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

Congratulations!! You've now gone through a more complex RAG framework that utilizes embeddings and semantic similarity based search. You're now equipped with the basic tools to understand the more modern RAG frameworks utilized by industry today. 

In this basic implementation, the chunking of documents and storing into the FAISS index is rather crude. This is fine as there are few documents to work with. However, in industry companies have to deal with a large database of large documents and online data repositories that are consistently being modified. The approach used here is simply not sufficient to handle the large quantities of data in industry. In a subsequent lesson we will look at the use of pgvector, postgres and docker as a means to create and maintain an online, efficiently vectorized, and easily reproducible database.  

This implementation had several lines of code, consisting of custom functions. A subsequent lesson looks at a library developed by Meta called llama index that condenses a lot of the operations done here into single lines of code. Additionally, llama index also includes metrics to evaluate RAG frameworks. 

### Beyond Semantic RAG - Modern RAG frameworks

Although this implementation is undeniably powerful, there have been many advances in RAG over the years. Since generative AI and LLMs have been such hot topics in research and industry over the last 5 years or so, there has been a lot of effort invested in developing ever more efficient RAG techniques. Check out [this](https://www.youtube.com/watch?v=sGvXO7CVwc0) youtube video and [this](https://wandb.ai/site/articles/rag-techniques/) article to learn more about recent advances in RAG. 

It is useful to note that the overwhelming effort in the development of RAG has been on the "Retrieve" stage since the "Augment" stage is more to do with the design of the augmented prompt that is fed to the LLM. Many recent efforts have focused on the methods of storing chunks (such as relational databases, heirarchical databases) and searching through the database of chunks (graph RAG). Some implementations modify the user prompt to remove unnecessary context and make it easier to retrieve truly relevant context. Alternatively, other implementations have another layer of checks to make sure the retrieved context is really relevant to the query. For this purpose, reranking of the retrieved contexts through different measures of relevance is most common. One may also have an additional classification step to retrieve chunks that are in the same domain as the user query. 

The above is just a *small* snapshot of the many RAG approaches being developed. This field is gaining more traction everyday which is fueling even more research efforts. In an ever evolving field like Generative AI, it is therefore key to keep yourself abreast of the latest ideas, models, and trends.