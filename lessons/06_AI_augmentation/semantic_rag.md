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

Lets look deeper into the implementation below.

Here are some additional resources: [Youtube video](https://www.youtube.com/watch?v=HREbdmOSQ18), [Non-brand Data article](https://www.nb-data.com/p/simple-rag-implementation-with-contextual)

## Example Implementation

### Setting up

```python
# Load .env (expects OPENAI_API_KEY)
if load_dotenv():
    print("Loaded openai api key")
else:
    print("no api key loaded check out .env")
    
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = Path("brightleaf_pdfs")  # <- your BrightLeaf PDF folder
assert DATA_DIR.exists(), f"{DATA_DIR} not found. Put PDFs there."
```

### Load and clean PDF text

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

corpus = load_corpus(DATA_DIR)
print(f"Loaded {len(corpus)} docs:", [c["doc_id"] for c in corpus])
```

```
Loaded 6 docs: ['earnings_report.pdf', 'employee_benefits.pdf', 'mission_statement.pdf', 'partnerships.pdf', 'product_specs.pdf', 'security_policy.pdf']
```

### Chunking document text

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
```
Total chunks: 930
```
### From text to embeddings

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
emb.shape
```

```
(930, 1536)
```

![Embeddings Semantic Similarity](resources/openai_embeddings_similarity.png)

### FAISS and determining similarity


```python
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(dim)
index.add(emb)
id2meta = {i: docs[i] for i in range(len(docs))}
print("FAISS index ready. Vectors:", index.ntotal)
```

```
FAISS index ready. Vectors: 930
```

### Semantic Retrieval

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

<code>
0.828 mission_statement.pdf chunk 0 text: Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our mission is to make solar power practical, affordable, and accessible to communities that have historically been left behind in the transition to clean energy. We are not only engineers and builders, but educators, partners, and advocates for a more resilient and equitable power grid. Every installation represents more than energy savings-it is an investment in long-term community well-being. Expanded Vision The company began its journey in the southeastern United States, where energy poverty and infrastructure challenges remain significant barriers to renewable adoption.<br><br>
0.734 mission_statement.pdf chunk 5 text: ransparent financing, and community ownership. We aim to ensure that the economic benefits of solar generation remain within the communities we serve. By 2030, BrightLeaf's goal is to help 100,000 households transition to renewable energy-not through isolated projects, but through a national network of resilient, community-led power systems.<br> <br>
0.703 employee_benefits.pdf chunk 68 text: ation, BrightLeaf ensures its workforce evolves alongside the renewable energy sector.
</code>


### Augment and Generate

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

### Example - Semantic RAG in action

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
Question: "When did BrightLeaf partner with SunSpan and what did they focus on"

Retrieved Contexts:
<code>
Context score: 0.7309314012527466
Context doc: partnerships.pdf
Context chunk: 1
Context text: . This partnership targets legacy industrial zones in Ohio and Michigan, pairing solar and wind power systems with SunSpan's real-time grid analytics. The collaboration has already produced measurable results, with the first hybrid sites showing 18 percent improved grid stability and 30 percent lower emissions. SunSpan and BrightLeaf also co-authored a white paper on the economic benefits of decarbonizing regional manufacturing supply chains. Academic and Workforce Collaborations Beyond corporate partnerships, BrightLeaf works closely with universities and community colleges to train the next generation of clean-energy professionals. The Bright Scholars Initiative offers scholarships and internships for students in sustainability, electrical engineering, and environmental data analytics. Collaboration with technical schools ensures a pipeline of skilled labor for new projects.

Context score: 0.713233470916748
Context doc: partnerships.pdf
Context chunk: 0
Context text: EcoVolt Energy (2022 Partnership) BrightLeaf's collaboration with EcoVolt Energy, established in 2022, focused on delivering microgrid solutions to rural communities in Georgia and South Carolina. The initiative combined BrightLeaf's solar generation systems with EcoVolt's battery storage expertise. Together, they launched five pilot sites that reduced community energy costs by an average of 25 percent. The partnership also created 40 permanent local jobs and helped utilities in the region study the integration of renewable storage into traditional grids. SunSpan Microgrids (2025 Partnership) In 2025, BrightLeaf joined forces with SunSpan Microgrids to develop hybrid renewable infrastructure for the Midwest. This partnership targets legacy industrial zones in Ohio and Michigan, pairing solar and wind power systems with SunSpan's real-time grid analytics.

Context score: 0.6759465336799622
Context doc: mission_statement.pdf
Context chunk: 0
Context text: Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. Our mission is to make solar power practical, affordable, and accessible to communities that have historically been left behind in the transition to clean energy. We are not only engineers and builders, but educators, partners, and advocates for a more resilient and equitable power grid. Every installation represents more than energy savings-it is an investment in long-term community well-being. Expanded Vision The company began its journey in the southeastern United States, where energy poverty and infrastructure challenges remain significant barriers to renewable adoption.
</code>

Without RAG:
<code>
BrightLeaf partnered with SunSpan in January 2022. The partnership focused on enhancing the development of solar energy projects, specifically by leveraging BrightLeaf's data extraction and processing capabilities to streamline project development and improve efficiency in the solar energy sector.
</code>

With RAG:
<code>
BrightLeaf partnered with SunSpan in 2025, focusing on developing hybrid renewable infrastructure for the Midwest, specifically targeting legacy industrial zones in Ohio and Michigan by pairing solar and wind power systems with SunSpan's real-time grid analytics.
</code>


### Beyond Semantic Search - Other RAG frameworks