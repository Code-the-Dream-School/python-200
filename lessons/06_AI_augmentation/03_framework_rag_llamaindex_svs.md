# Semantic RAG with LlamaIndex

In Step 2, we built a semantic RAG system manually using FAISS and OpenAI embeddings.Here we do the same thing with **LlamaIndex**, a framework that automates chunking, embedding, indexing, and retrieval in just a few lines of code.

Note the "index" in LlamaIndex refers directly to the same kind of semantic index we built manually in previous lessons. It is a package built around the concept of semantic indexes.

LlamaIndex is a framework for building LLM-powered agents over your data with LLMs and workflows. It can use RAG pipelines. 

As LLM offers an interface between humans and data. They are pre-trained on huge public data. At the same time, Llamaindex provides us with the facility to build a use case on our own data. That is, context augmentation (eg, RAG), where we make our data available to LLM to solve the problem. 

Here, agents are LLM assistants uses tools to perform a given task, like data extraction or research.
Workflows are multi-step processes that combine one or more agents to create a complex LLM application.

LlamaIndex can be used to ingest existing data and structured data, has helper functions for API integrations, and can also monitor apps. 

Use cases:
- Question Answering
- Chatbots
- Document Understanding and Data Extraction
- Autonomous Agents


Note that your environment has the following packages installed that are important for the current notebook: 

- `llama-index-core`:  base framework important for many core functionality in llamaindex -llama-index-core==0.14.10
- `llama-index-embeddings-openai` : embedding provider
- `pypdf` and `python-dotenv`: for PDF and key loading


## 2. Load API Key

We will look for the environment variables in .env file and check if they are loaded.


```python
from dotenv import load_dotenv

if load_dotenv():
    print("success")
else:
    print("oops")
```

    success
    

Basically what we've been building toward can now be done in four lines of code:

    docs = SimpleDirectoryReader("brightleaf_pdfs").load_data()
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is BrightLeaf Solar's mission?")

This replaces almost everything we built manually in Step 2 with a framework-focused solution that handles things for you. We'll walk through these code steps below, but stripped of all the explanatory padding, it really is that simple!

## 3. Import docs and build the index

Now, we will build an index. 
Vector Store is a system where we are able to store vector embeddings (it's a representation of data into the form of vectors or numerical floating point numbers). And, LlamaIndex can use a vector store as an index. It can store documents and be used to answer queries.  
We will import VectorStoreIndex and SimpleDirectoryReader from Llamaindex core. SimpleDirectoryReader is the simplest way to load data from local files into LlamaIndex.



```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from pathlib import Path
```

- Note: It is important to direct correct path to brightleaf_pdfs else LlamaIndex wont be able to process the files.


```python
# BrightLeaf PDF directory (same as in 02_semantic_rag.ipynb)
DATA_DIR = Path("brightleaf_pdfs")
assert DATA_DIR.exists(), f"{DATA_DIR} not found. Put BrightLeaf PDFs there."
```


```python
# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader("brightleaf_pdfs").load_data()
```

LlamaIndex has successfully ingested your PDFs and wrapped them into structured Document objects that can now be indexed, embedded, and queried by an LLM.
Document: LlamaIndex’s core data structure that represents one source file (like a PDF) after it’s loaded.
id_: A unique identifier automatically assigned to each document.
embedding: A numerical vector representation of the document used for semantic search; None means it hasn’t been created yet.
metadata: Descriptive information about the file (name, path, type, size, dates) used for filtering and source tracking.
excluded_embed_metadata_keys: Metadata fields that are ignored when generating embeddings to keep them semantically meaningful.
excluded_llm_metadata_keys: Metadata fields hidden from the LLM to avoid unnecessary or irrelevant context.
text_resource: The container holding the extracted text from the PDF.
MediaResource: An internal wrapper used by LlamaIndex to manage different content types (text, image, audio, video).
content / text: The raw extracted data from the PDF; in this case it shows PDF internals rather than cleaned text.
file_path / file_name / file_type: Identify where the document came from and what format it is.
Vector embeddings (later step): Used to compare documents and queries based on meaning rather than keywords.


```python
type(docs[0])
```




    llama_index.core.schema.Document




```python
len(docs)
```




    6



Look at docs[0]: Fill this in some, don't print entire thing, look at a few attributes:


```python
docs[0].doc_id, docs[0].text[:500]
```




    ('8190b231-abe2-45d8-a2cc-05dec4840041',
     '%PDF-1.4\n% ReportLab Generated PDF document http://www.reportlab.com\n1 0 obj\n<<\n/F1 2 0 R /F2 3 0 R /F3 4 0 R /F4 5 0 R\n>>\nendobj\n2 0 obj\n<<\n/BaseFont /Helvetica /Encoding /WinAnsiEncoding /Name /F1 /Subtype /Type1 /Type /Font\n>>\nendobj\n3 0 obj\n<<\n/BaseFont /Helvetica-BoldOblique /Encoding /WinAnsiEncoding /Name /F2 /Subtype /Type1 /Type /Font\n>>\nendobj\n4 0 obj\n<<\n/BaseFont /ZapfDingbats /Name /F3 /Subtype /Type1 /Type /Font\n>>\nendobj\n5 0 obj\n<<\n/BaseFont /Helvetica-Bold /Encoding /WinAnsiEncodi')



Check to see which vector store is being used on the back-end, check which embedding model is being used.


```python
print("Vector store in use:", type(index._vector_store).__name__)
```

    Vector store in use: SimpleVectorStore
    


```python
print("Embedding model:", type(Settings.embed_model).__name__)
```

    Embedding model: OpenAIEmbedding
    

### Create index
Here we're using OpenAI for embeddings and llamaindex's default fallback vector store `SimpleVectorStore` for our vector store. When we used FAISS previously, we manually created a vector index and handled the similarity search ourselves. LlamaIndex takes care of that for us. By default, it uses a small in-memory vector store called SimpleVectorStore. You can think of this as a lightweight, Python-based version of FAISS: it stores each chunk’s embedding and performs similarity search under the hood. It isn't meant for large production systems, but is great for learning the basic RAG workflow because there’s nothing to install or configure. Later, when we want a real database-backed vector store (like pgvector), we'll need to swap out this default back end.


```python
# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)
```


```python
# dir(index)
```

## 4. Create and use a query engine
After building the index, we can create a query engine. A query engine is a small helper object that wraps the entire retrieval process for us. When we give it a question, it automatically finds the most relevant chunks in the index and then sends those chunks, along with the question, to the LLM to produce an answer. This keeps our code simple: instead of calling separate retrieval and generation steps, we interact with one unified interface.

Below, we create the query engine and try a few questions about the PDF collection.


```python
#check LLM model currently being used
print("LLM model:", type(Settings.llm).__name__)
```

    LLM model: OpenAI
    


```python
query_engine = index.as_query_engine(similarity_top_k=25)
```


```python
questions = [
    "What is BrightLeaf Solar's mission?",
    "How did profits change between 2023 and 2024?",
    "Which partner joined most recently?"
]

for q in questions:
    print(f"\nQ: {q}")
    print("A:", query_engine.query(q))
```

You've implemented a semantic RAG pipeline using Llamaindex, basically using four lines of code!

What makes this so powerful is how much work is wrapped into so little code. LlamaIndex is a fully maintained, widely used framework built by a dedicated team, and it bundles together a large amount of engineering that we would otherwise have to implement ourselves. The library handles chunking, embedding, vector storage, retrieval, ranking, and passing the right context to the LLM, all through a clean and consistent interface. 

It also supports multiple RAG architectures beyond the simple naive semantic RAG architecture we are using here, which makes it adaptable to different real-world use cases as projects grow. And as we will see later, it even includes built-in tools for evaluating how well a RAG system is performing.


```python

```
