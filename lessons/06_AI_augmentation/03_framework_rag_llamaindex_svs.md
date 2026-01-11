# RAG Implementation with LlamaIndex

In the previous lessons, we looked at custom semantic RAG implementations that leveraged both an in-memory vector store and an online docker container-based vector store. Now we will implement the same semantic RAG frameworks as before using a library called [LlamaIndex](https://www.llamaindex.ai/). LlamaIndex provides a simple API that automates the database reading, chunking, storing, context retrieval and augmentation, and response generation steps. Moreover, LlamaIndex also includes support for many external data storage libraries, models, and embeddings making it a very versatile tool to develop your own custom RAG frameworks. LlamaIndex also includes methods to evaluate your RAG framework, which we will go over briefly towards the end of this lesson.

<!-- Note the "index" in LlamaIndex refers directly to the same kind of semantic index we built manually in previous lessons. It is a package built around the concept of semantic indexes. 

LlamaIndex is a framework for building LLM-powered agents over your data with LLMs and workflows. It can use RAG pipelines. 

As LLM offers an interface between humans and data. They are pre-trained on huge public data. At the same time, Llamaindex provides us with the facility to build a use case on our own data. That is, context augmentation (eg, RAG), where we make our data available to LLM to solve the problem. 

Here, agents are LLM assistants uses tools to perform a given task, like data extraction or research.
Workflows are multi-step processes that combine one or more agents to create a complex LLM application. 

LlamaIndex can be used to ingest existing data and structured data, has helper functions for API integrations, and can also monitor apps. 

Use cases:
- Question Answering
- Chatbots
- Document Understanding and Data Extraction
- Autonomous Agents -->

Here are some additional resources to look at: [Youtube video on introduction to LlamaIndex](https://www.youtube.com/watch?v=cCyYGYyCka4), [IBM article on LlamaIndex](https://www.ibm.com/think/topics/llamaindex).


> **Note:** Make sure your environment has the following packages installed that are important for the current notebook: 
> - `llama-index-core`:  the base library for llamaindex (make sure the llama-index-core version is 0.14.10)
> - `llama-index-embeddings-openai`: support for OpenAI's embedding models
> - `llama-index-vector-stores-postgres`: support for PostgreSQL
> - `psycopg2-binary`: pgvector library in Python to interact with the postgreSQL database
> - `pypdf` and `python-dotenv`: for PDF and key loading

We will first begin with an implementation of the in-memory semantic RAG framework. We will use the same Brightleaf Solar Company example as used in the previous lessons. Note that although LlamaIndex has support for FAISS, we will use LlamaIndex's internal functionality to search through the embeddings and retrieve the relevant context for a query. 

## In-memory Semantic RAG using LlamaIndex

As we have done for previous lessons, we will load the OpenAI API key from .env file and print `success` if it is loaded.

```python
from dotenv import load_dotenv

if load_dotenv():
    print("success")
else:
    print("oops")
```

Once the API key is loaded, we will load the Brightleaf company documents into LlamaIndex's storage index. 

<!-- Basically what we've been building toward can now be done in four lines of code:

    docs = SimpleDirectoryReader("brightleaf_pdfs").load_data()
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is BrightLeaf Solar's mission?")

This replaces almost everything we built manually in Step 2 with a framework-focused solution that handles things for you. We'll walk through these code steps below, but stripped of all the explanatory padding, it really is that simple! -->

### Read, Chunk, and Store Documents

LlamaIndex provides a default storage index called `VectorStoreIndex` to store the chunked and embedded text from the documents. The following code shows how to create the vector store index for the brightleaf documents.

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader("brightleaf_pdfs").load_data()

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)
```

> **Note**: Make sure the `brightleaf_pdfs` directory is in the appropriate location, you can additionally add an `assert` statement to check the existence of the directory if it is located elsewhere.

Compared to the custom implementation from the semantic RAG lesson, the above two lines of code capture the document reading, chunking, embedding, and storing steps, demonstrating a huge reduction in lines of code! LlamaIndex uses the inbuilt `SimpleDirectoryReader` method to read the documents and store the metadata and text into the `docs` list. Each element of this list is a `Document` object containing the metadata and text for each loaded document. To make sure the text has been read correctly, you can check the metadata and text for each document. For example, to check the first document's metadata you can run the following code.

```python
print(docs[0].metadata)
```
The output should look something like this.
```
{'page_label': '1',
 'file_name': 'earnings_report.pdf',
 'file_path': 'c:\\Users\\rosha\\Downloads\\CTD RAG\\rag\\brightleaf_pdfs\\earnings_report.pdf',
 'file_type': 'application/pdf',
 'file_size': 3658,
 'creation_date': '2025-10-22',
 'last_modified_date': '2025-11-10'}
```
Now, to make sure the text from the "earnings_report" document (or whichever document was read first) has been read correctly you can check the first 100 characters of the text by running the following code.
```python
print(docs[0].text_resource.text[:100])
```
The output should look something like this.
```
"Overview\nThis report summarizes BrightLeaf Solar's financial performance from 2021 through 2025. The"
```
If there is a reading error (usually due to the document not being in the UTF-8 encoding format), the text will have some random symbols that do not resemble the english language. 

<!-- Document: LlamaIndex’s core data structure that represents one source file (like a PDF) after it’s loaded.
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
     '%PDF-1.4\n% ReportLab Generated PDF document http://www.reportlab.com\n1 0 obj\n<<\n/F1 2 0 R /F2 3 0 R /F3 4 0 R /F4 5 0 R\n>>\nendobj\n2 0 obj\n<<\n/BaseFont /Helvetica /Encoding /WinAnsiEncoding /Name /F1 /Subtype /Type1 /Type /Font\n>>\nendobj\n3 0 obj\n<<\n/BaseFont /Helvetica-BoldOblique /Encoding /WinAnsiEncoding /Name /F2 /Subtype /Type1 /Type /Font\n>>\nendobj\n4 0 obj\n<<\n/BaseFont /ZapfDingbats /Name /F3 /Subtype /Type1 /Type /Font\n>>\nendobj\n5 0 obj\n<<\n/BaseFont /Helvetica-Bold /Encoding /WinAnsiEncodi') -->

Once the documents are read, the next step is to chunk the text, convert them into embeddings, and store them. `VectorStoreIndex` automatically handles the chunking, embedding and storing of the documents. By default, LlamaIndex stores the embeddings in a `SimpleVectorStore` object. You can check this by running the following code.

```python
print(type(index._vector_store).__name__)
```
The output should say the following.
```
SimpleVectorStore
```

LlamaIndex uses predefined defaults for the chunking parameters and the embedding model. If not specified, LlamaIndex uses OpenAI's "text-embedding-ada-002" model. You can change these values before creating the vector store index to make sure LlamaIndex uses your custom values. To check the default values you can run the following code.

```python
from llama_index.core import Settings

print(f"Default chunk size:  {Settings.chunk_size}")
print(f"Default chunk overlap: {Settings.chunk_overlap}")
print(f"Default embedding model: {Settings.embed_model.model_name}")
```
The output should be the following.
```
Default chunk size:  1024
Default chunk overlap: 200
Default embedding model: text-embedding-ada-002
```
To change these values and the embedding model to different OpenAI embedding model, you can do the following.
```python
## Optional: Change default chunking parameters and embedding model (Run before creating vector store index)
from llama_index.embeddings.openai import OpenAIEmbedding

# Specify the new model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Specify the new chunking parameters
Settings.chunk_size = 512
Settings.chunk_overlap = 50
```

Now that we have the text chunked, embedded and stored, the next step is to retrieve the relevant context based on the user query, augment it to the query and generate the response from the LLM.

<!-- ### Create index
Here we're using OpenAI for embeddings and llamaindex's default fallback vector store `SimpleVectorStore` for our vector store. When we used FAISS previously, we manually created a vector index and handled the similarity search ourselves. LlamaIndex takes care of that for us. By default, it uses a small in-memory vector store called SimpleVectorStore. You can think of this as a lightweight, Python-based version of FAISS: it stores each chunk’s embedding and performs similarity search under the hood. It isn't meant for large production systems, but is great for learning the basic RAG workflow because there’s nothing to install or configure. Later, when we want a real database-backed vector store (like pgvector), we'll need to swap out this default back end. 

```python
# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)
```

```python
# dir(index)
```
-->

### Retrieve Context, Augment, and Generate Response - The Query Engine
LlamaIndex uses a helper object called a "query engine" that wraps the entire context retrieval, augmentation, and response generation process. This contributes to further reduction in lines of code. 

LlamaIndex uses OpenAI's "gpt-3.5-turbo" model in the backend by default to generate the response to the user query. You can check this by running the following code.

```python
#check LLM model currently being used
print(f"LLM model: {Settings.llm.model}")
```
As mentioned earlier, the output should be the following.
```
LLM model: gpt-3.5-turbo
```
    
Below, we create the query engine from the vector store index and look at the responses and the retrieved contexts to three sample questions.

```python
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What is BrightLeaf Solar's mission?",
    "How did profits change between 2023 and 2024?",
    "Which partner joined most recently?"
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
        print("-" * 30)
```

When creating the query engine, we define the number of retrieved chunks by setting `similarity_top_k=3`. The vector store index query engine uses cosine similarity to search through the embeddngs to obtain the most relevant chunks. We can also look at the most relevant chunks and their corresponding similarity scores through the `response` object's `source_nodes`. The output will look like the following.

```
Q: What is BrightLeaf Solar's mission?
2026-01-10 18:11:58,038 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2026-01-10 18:11:59,518 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-10 18:11:59,655 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
A: BrightLeaf Solar's mission is to make solar power practical, affordable, and accessible to communities that have historically been left behind in the transition to clean energy. They aim to be educators, partners, and advocates for a more resilient and equitable power grid, with each installation representing an investment in long-term community well-being.
Node ID: 6db23967-7cb2-49a8-9814-6d381ce5b69e
Similarity Score: 0.9034
Text Snippet: Overview
BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a p...
------------------------------
Node ID: b6833bf4-3fea-487c-9843-6b52be02bedb
Similarity Score: 0.8534
Text Snippet: EcoVolt Energy (2022 Partnership)
BrightLeaf's collaboration with EcoVolt Energy, established in 202...
------------------------------
Node ID: 566668b4-d22b-49df-b166-1932fd899d92
Similarity Score: 0.8440
Text Snippet: Overview
This report summarizes BrightLeaf Solar's financial performance from 2021 through 2025. The...
------------------------------

Q: How did profits change between 2023 and 2024?
2026-01-10 18:12:00,355 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-10 18:12:00,541 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
A: Profits increased from 0.5 million USD in 2023 to 1.1 million USD in 2024.
Node ID: 566668b4-d22b-49df-b166-1932fd899d92
Similarity Score: 0.7936
Text Snippet: Overview
This report summarizes BrightLeaf Solar's financial performance from 2021 through 2025. The...
------------------------------
Node ID: 6db23967-7cb2-49a8-9814-6d381ce5b69e
Similarity Score: 0.7257
Text Snippet: Overview
BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a p...
------------------------------
Node ID: b6833bf4-3fea-487c-9843-6b52be02bedb
Similarity Score: 0.7239
Text Snippet: EcoVolt Energy (2022 Partnership)
BrightLeaf's collaboration with EcoVolt Energy, established in 202...
------------------------------

Q: Which partner joined most recently?
2026-01-10 18:12:01,469 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
A: SunSpan Microgrids joined most recently.
Node ID: b6833bf4-3fea-487c-9843-6b52be02bedb
Similarity Score: 0.7601
Text Snippet: EcoVolt Energy (2022 Partnership)
BrightLeaf's collaboration with EcoVolt Energy, established in 202...
------------------------------
Node ID: 566668b4-d22b-49df-b166-1932fd899d92
Similarity Score: 0.7209
Text Snippet: Overview
This report summarizes BrightLeaf Solar's financial performance from 2021 through 2025. The...
------------------------------
Node ID: 6db23967-7cb2-49a8-9814-6d381ce5b69e
Similarity Score: 0.7166
Text Snippet: Overview
BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a p...
------------------------------
```
Since we only want to augment the top 3 most relevant chunks, the response will only have 3 source nodes. As can be seen, the model is able to respond accurately and retrieve generally relevant chunks to each question.

Congratulations! You've implemented a semantic RAG pipeline using Llamaindex, heavily reducing the total lines of code compared to the custom implementation in the previous lesson!

What makes this so powerful is how much work is wrapped into so little code. LlamaIndex is a fully maintained, widely used framework built by a dedicated team, and it bundles together a large amount of engineering that we would otherwise have to implement ourselves. The library handles chunking, embedding, vector storage, retrieval, ranking, and passing the right context to the LLM, all through a clean and consistent interface. Of course, a big reason for this reduction in lines of code is that LlamaIndex is designed to be a plug-and-play tool with many hyperparameters predefined. Changing these default values will increase some lines of code, but the overall reduction in code lines is still significant.

It also supports multiple RAG architectures beyond the simple naive semantic RAG architecture we show here, which makes it adaptable to different real-world use cases as projects grow. As an example, we will look at the implementation for the online-database semantic RAG with pgvector and postgreSQL in LlamaIndex next. Additionally, LlamaIndex includes built-in tools for evaluating how well a RAG system is performing which we will look at later.

## Online docker database semantic RAG using LlamaIndex

This notebook is the "framework sequel" to the earlier RAG notebooks:

- Hand-rolled semantic RAG (FAISS)
- Hand-rolled Postgres + pgvector RAG (Docker)
- Framework RAG with LlamaIndex + SimpleVectorStore (SVS)

Here we use **LlamaIndex + pgvector** while reusing the **same Postgres server/container** you already set up.

## Key idea

- We reuse the same Postgres database **server**.
- We **do not reuse** the hand-rolled table schema (e.g., `rag_chunks`).
- Instead, LlamaIndex manages its own table (e.g., `li_brightleaf_pgvector`) to store nodes + embeddings.

## Prereqs

1. Your Postgres + pgvector container is running and reachable at `localhost:5432`.
2. You have PDFs in `brightleaf_pdfs/` (same as the previous notebooks).
3. Your OpenAI key is set:

```bash
export OPENAI_API_KEY="..."
```

(Or set it in the notebook environment.)

Make sure the following are installed in your virtual environment

    llama-index llama-index-vector-stores-postgres llama-index-embeddings-openai psycopg2-binary

```python
# Connection + indexing configuration
# Update these to match your Docker pgvector setup from the previous notebook.

PG_HOST = "localhost"
PG_PORT = 5432
PG_DATABASE = "ctd_rag"
PG_USER = "ctd"
PG_PASSWORD = "ctdpassword"

# LlamaIndex will manage THIS table (separate from your hand-rolled rag_chunks table).
LI_TABLE_NAME = "li_brightleaf_pgvector"

# Embedding model choice must match embed_dim.
EMBED_MODEL_NAME = "text-embedding-3-small"
EMBED_DIM = 1536

PDF_DIR = "brightleaf_pdfs"

# If you rerun "build" repeatedly, you will insert duplicates.
# Recommended: build once, then use the query-only section below.
# BUILD_INDEX = True
BUILD_INDEX = False
```

Check that postgres is reachable

```python
import psycopg2

conn = psycopg2.connect(
    host=PG_HOST,
    port=PG_PORT,
    dbname=PG_DATABASE,
    user=PG_USER,
    password=PG_PASSWORD,
)
cur = conn.cursor()
cur.execute("SELECT version();")
print(cur.fetchone()[0])
cur.close()
conn.close()
```

## Build (first run)

Read PDFs -> chunk -> embed -> store in Postgres via LlamaIndex

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

# Load .env (expects OPENAI_API_KEY)
if load_dotenv():
    print("Loaded openai api key")
else:
    print("no api key loaded check out .env")
    
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

```python
Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL_NAME)

docs = SimpleDirectoryReader(PDF_DIR).load_data()
print(f"Loaded {len(docs)} documents from: {PDF_DIR}")
```

Before we used fallback (simplevectorstore), now we explicitly call pgvector store using our db with parameters

```python 
vector_store = PGVectorStore.from_params(
    host=PG_HOST,
    port=PG_PORT,
    database=PG_DATABASE,
    user=PG_USER,
    password=PG_PASSWORD,
    table_name=LI_TABLE_NAME,
    embed_dim=EMBED_DIM,
)
```

In our previous SimpleVectorStore example, LlamaIndex used default in-memory storage behind the scenes, so we could build an index without thinking about storage. In this pgvector version, we want the "vector database" from the diagram to be Postgres instead of RAM, so we create a PGVectorStore and pass it in via a StorageContext. 

You can think of StorageContext as the wiring step that tells LlamaIndex, "store and search embeddings in Postgres." Its basically an abstraction layer between different kinds of vector stores. After that, the rest of the workflow is the same: LlamaIndex chunks the documents into nodes (which are called chunks in our hand-rolled example), embeds them, stores them, and later embeds the user query to retrieve the top-k most relevant nodes for the LLM.

```python
if BUILD_INDEX:
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    print(f"Indexed documents into Postgres table: {LI_TABLE_NAME}")
else:
    print("BUILD_INDEX is False; skipping indexing.")
```

Once you have the index constructed, we can build the query engine.

Also, we can jump here once index is built the first time (set BUILD_INDEX to false):

```python
# Query (works immediately after build)
from llama_index.core import VectorStoreIndex

# If you just built the index above, you already have `index` in memory.
# If not, attach to the existing Postgres-backed vector store.
if "index" not in globals():
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

qe = index.as_query_engine(similarity_top_k=3)
```

```python
question = "When did BrightLeaf partner with SunSpan and what did they focus on?"
response = qe.query(question)

print("Q:", question)
print()
print(response)
```

That's it! We've set up the system to build and query the pgvector store! Let's test it out. 

## Query-only pattern (later runs)
Use this pattern when the Postgres table already exists and is populated,
and you do NOT want to re-read PDFs or re-embed.

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL_NAME)

vector_store_q = PGVectorStore.from_params(
    host=PG_HOST,
    port=PG_PORT,
    database=PG_DATABASE,
    user=PG_USER,
    password=PG_PASSWORD,
    table_name=LI_TABLE_NAME,
    embed_dim=EMBED_DIM,
)

index_q = VectorStoreIndex.from_vector_store(vector_store=vector_store_q)
qe_q = index_q.as_query_engine(similarity_top_k=3)

question2 = "Which partner joined most recently?"
response2 = qe_q.query(question2)

print("Q:", question2)
print()
print(response2)
```

```python
# Optional reset: drop the LlamaIndex-managed table so you can rebuild cleanly
# This is useful for teaching and demos.

# import psycopg2

# drop_sql = f'DROP TABLE IF EXISTS "{LI_TABLE_NAME}";'

# conn = psycopg2.connect(
#     host=PG_HOST,
#     port=PG_PORT,
#     dbname=PG_DATABASE,
#     user=PG_USER,
#     password=PG_PASSWORD,
# )
# conn.autocommit = True
# cur = conn.cursor()
# cur.execute(drop_sql)
# cur.close()
# conn.close()

# print(f"Dropped table (if existed): {LI_TABLE_NAME}")
```

```python
# Optional: List tables

# import psycopg2

# list_sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';"

# conn = psycopg2.connect(
#     host=PG_HOST,
#     port=PG_PORT,
#     dbname=PG_DATABASE,
#     user=PG_USER,
#     password=PG_PASSWORD,
# )
# conn.autocommit = True
# cur = conn.cursor()
# cur.execute(list_sql)
# tables = cur.fetchall()

# cur.close()
# conn.close()

# print(f"Listed tables: {tables}")
```

## Notes / common issues

- If you see duplicate retrieval behavior, you probably re-ran the build cell and inserted more nodes.
  Use the reset cell (drop table) and rebuild.

- If you get an embedding dimension mismatch error:
  - Ensure `EMBED_MODEL_NAME` and `EMBED_DIM` agree.
  - For OpenAI `text-embedding-3-small`, `EMBED_DIM = 1536`.

- If you cannot connect to Postgres:
  - Confirm the Docker container is running.
  - Confirm the port mapping is `-p 5432:5432` (or update `PG_PORT`).
