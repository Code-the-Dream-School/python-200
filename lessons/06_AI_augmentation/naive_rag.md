# Naive Keyword-based RAG
In the previous lesson, we saw the different types of retrieval augmentated generation (RAG) used to augment the chatbot's response to improve it's accuracy. In this lesson, we'll go over the basic keyword-based RAG framework. Please note that this approach is too simple and is not used in industry, but it is useful to see the response augmentation process here. We will build on this approach in the next lesson with the more complex semantic search-based RAG approach.

 > **NOTE:**  To get the most out of this lesson, it is recommended that you run the examples yourself. Before beginning with the exercises, ensure that you have your OpenAI API key. If you don't have this, please reach out to your mentor. To run the exercises, you will need to create a virtual environment with the following packages installed: `OpenAI`, `pypdf`, and `dotenv`.

Here are some additional resources for your reference: [Basic RAG databricks article](https://docs.databricks.com/aws/en/generative-ai/retrieval-augmented-generation), [OpenAI RAG article](https://help.openai.com/en/articles/8868588-retrieval-augmented-generation-rag-and-semantic-search-for-gpts), [RAG vs Fine Tuning vs Prompt Engineering](https://www.youtube.com/watch?v=zYGDpG-pTho).

As seen in the previous lesson, at the highest level of abstraction every RAG approach follows the procedure shown in the figure below (taken from the Basic RAG databricks article). 

![Basic Rag Framework](resources/basic_rag.png)

The chatbot generates a response based on the user prompt and relevant supporting data from an external data repositoy (which can be documents, knowledge graphs or any other representation of data) that is not captured by the data used to train the chatbot. A good example of this is proprietary company information that is not in the public domain. The different RAG approaches differ in the methods used to determine the data relevant to the user prompt from the data repository. The most basic approach to isolate relevant data from th external data source is a simple keyword matching search. Let us go over an example implementation of this approach.

## Example Implementation

For this example, assume that you are developing a customer service chatbot for a company called Brightleaf Solar Solutions. The chatbot has access to the company's proprietary documents and must be able to leverage this data to improve its responses to user queries. 

The main code for the keyword-based RAG chatbot is given below. Note that in this example we store and retrieve the OpenAI API Key in the `.env` file.

```python
from dotenv import load_dotenv
import os
from pathlib import Path
from pypdf import PdfReader
from openai import OpenAI

# Load environment and OpenAI key
if load_dotenv():
    print("✅ Successfully loaded API key")
else:
    print("⚠️ Failed to load API key from .env file")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load all PDF text into a dictionary
pdf_dir = Path("brightleaf_pdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError("No PDFs found in the pdfs directory.")

docs = {f.name: extract_text_from_pdf(f) for f in pdf_files}
print(f"Loaded {len(docs)} BrightLeaf PDF(s).")

print("\nType 'quit' to exit.\n")
while True:
    query = input("Enter your query: ").strip()
    if query.lower() in {"quit", "exit"}:
        print("Goodbye.")
        break

    results = simple_keyword_retrieval(query, docs, verbose=True)
    context = results[0][1]
    answer = ask_llm(query, context)

    print("\n--- Response ---")
    print(answer)
    print("\n" + "=" * 60 + "\n")
```

The script begins with loading and extracting data from the requisite company documents (using the `extract_text_from_pdf` function). Then the chatbot conversation scripting is done in a similar manner to the chatbots lesson in Week 1. For each user query, the relevant context from the documents is retrieved using a simple-keyword matching approach (in the `simple_keyword_retrieval` function). This context is then augmented to the user query and input to the chatbot to generate the response (in the `ask_llm` function). We will go over each function in detail.

### Extract and store text from documents

In the `extract_text_from_pdf` function, the text in each document in the system path represented by the `pdf_path` variable is extracted and stored in the `text` variable. As seen in the main code, this function is called for each document in the directory. Now that we have the data from the company documents, we will create the function that retrieves the relevant context from the documents based on the user query. 

```python
def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text.strip())
    return "\n".join(text)
```

### Keyword-based context retrieval

The function for keyword-based context retrieval is shown below. Note that the `documents` argument for the function is a dictionary with the document names as keys and the list of words as the values, extracted using the `extract_text_from_pdf` function.

```python
def simple_keyword_retrieval(query, documents, verbose=True):
    """
    Naive keyword retrieval using token overlap scoring.
    - Removes stopwords and punctuation for cleaner matching.
    - Returns the single best-matching document.
    """
    import string

    stopwords = [
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    ]

    # Translator to remove punctuation (so "Solar?" -> "solar")
    translator = str.maketrans("", "", string.punctuation)

    # Tokenize query: lowercase, remove punctuation and stopwords
    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        # Tokenize document: lowercase, remove punctuation and stopwords
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }

        # Compute simple overlap score
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))

        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    # Sort by overlap score (descending)
    scores.sort(reverse=True)

    # Pick the single best match (if score > 0)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
```
First, the words in the query are converted to lower case and articles and other irrelevant words (part of the `stopwords` list) are removed. The text in the documents are treated similarly and an overlap score is computed between words in each document and the user query (excluding articles and other irrelevant words). The document content with the highest overlap score is returned. Next, we create the function that augments the retreived data as context with the user prompt to generate the response from the LLM.

### Augment with context and generate response

The `ask_llm` function to geenerate the chatbot response based on the user prompt augmented with the retrieved context is shown below. Note that we clearly demarcate the retrieved context and the user query with appropriate labels in the chatbot prompt. 

> If we remove the labels and simply augmented the user query and context, do you think the response would change? Would it improve or worsen?

```python
def ask_llm(query, context):
    """Ask the LLM using retrieved context."""
    prompt = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context.strip()}\n\n"
        f"Question: {query}\nAnswer:"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

```

Congratulations!! Now you have a chatbot that can generate responses to user queries with additional context from an external database! You have created your own RAG enabled chatbot! Feel free to explore with different prompts and test the responses with and without the retrieved context to understand and appreciate the utility of RAG. 

## Food for thought - Evaluating the efficacy of your chatbot

Now you have a chatbot that can generate responses to user queries. But can you guarantee that the responses are better suited to the user queries or more useful? How would you assess this? In the next lesson, we will introduce the deepeval tool which is designed to do just this.

Also, it is useful to note that any perceived improvements to the chatbot's responses are heavily dependent on the external database you provide and the user query. In the example provided here, we provide some documents for Brightleaf Solar Solutions. So it is reasonable to assume that if you ask queries directly related to the content in the documents, you will get a richer, more accurate response from the chatbot. 

Another interesting observation is that for every user query, the content in the documents is searched for relevant context. Depending on the number and size of the documents this can be a computationally expensive process. The scanning approach is also not very scalable, which brings up the need to search the external database more efficiently. This led to the usage of vector dataspaces, which we will look at in the next lesson.