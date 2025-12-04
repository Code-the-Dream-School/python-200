# Semantic RAG

In the previous lesson we looked at a very basic implementation of RAG that used a keyword matching approach between the user query and the data in the external database. Now we'll go over a more complex but efficient RAG implementation, one that is closer to industry-standard RAG frameworks. Remember that even if this implementation looks more complex, the same steps as the keyword-based RAG will be followed. 

The basic semantic search based RAG framework is illustrated in the figure below.
![Naive Semantic RAG](resources/semantic_rag.png)

Immediately you will note some key differences between keyword RAG and semantic RAG. 

Here are some additional resources: [Youtube video](https://www.youtube.com/watch?v=HREbdmOSQ18), [Non-brand Data article](https://www.nb-data.com/p/simple-rag-implementation-with-contextual)

## Example Implementation

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
emb[:2].shape, dim
```

### FAISS and determining similarity


```python
faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(dim)
index.add(emb)
id2meta = {i: docs[i] for i in range(len(docs))}
print("FAISS index ready. Vectors:", index.ntotal)
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
    print(round(h["score"], 3), h["doc_id"], f"chunk {h['chunk_id']}")
```


### Augment and Generate

```python
def ask_llm(query, contexts, model="gpt-4o-mini", temperature=0.2, max_chars=3500):
    ctx = "\n\n---\n\n".join(c["text"] for c in contexts)
    ctx = ctx[:max_chars]
    prompt = (
        "Use ONLY the provided context to answer the question.\n"
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

q = "When did BrightLeaf partner with SunSpan and what did they focus on?"
ctx = retrieve(q, k=3)
print("Context docs:", [c["doc_id"] for c in ctx])
print()
print(ask_llm(q, ctx))
```

### Beyond Semantic Search - Other RAG frameworks