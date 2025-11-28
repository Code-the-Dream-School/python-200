# Keyword-based RAG
In the previous lesson, we saw the different types of retrieval augmentated generation (RAG) used to augment the chatbot's response to improve it's accuracy. In this lesson, we'll go over the basic keyword-based RAG framework. Please note that this approach is too simple and is not used in industry, but it is useful to see the response augmentation process here. We will build on this approach in the next lesson with the more complex semantic search-based RAG approach.

 > **NOTE:**  To get the most out of this lesson, it is recommended that you run the examples yourself. Before beginning with the exercises, ensure that you have your OpenAI API key. If you don't have this, please reach out to your mentor. To run the exercises, you will need to create a virtual environment with the following packages installed: `OpenAI`, `pypdf`, and `dotenv`.

Here are some additional resources for your reference: [Basic RAG databricks article](https://docs.databricks.com/aws/en/generative-ai/retrieval-augmented-generation), [OpenAI RAG article](https://help.openai.com/en/articles/8868588-retrieval-augmented-generation-rag-and-semantic-search-for-gpts), [RAG vs Fine Tuning vs Prompt Engineering](https://www.youtube.com/watch?v=zYGDpG-pTho).

As seen in the previous lesson, at the highest level of abstraction every RAG approach follows the procedure shown in the figure below (taken from the Basic RAG databricks article). 

![Basic Rag Framework](resources/basic_rag.png)

The chatbot generates a response based on the user prompt and relevant supporting data from an external data repositoy (which can be documents, knowledge graphs or any other representation of data) that is not captured by the data used to train the chatbot. A good example of this is proprietary company information that is not in the public domain. The different RAG approaches differ in the methods used to determine the data relevant to the user prompt from the data repository. The most basic approach to isolate relevant data from th external data source is a simple keyword matching search. Let us go over an example implementation of this approach.

## Implementation

For this example, assume that you are developing a customer service chatbot for a company called Brightleaf Solar Solutions. The chatbot has access to the company's proprietary documents and must be able to leverage this data to improve its responses to user queries. 

### Setting up

Note that in this example we store and retrieve the OpenAI API Key in the `.env` file.

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
```

We will go over each function in detail.

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
    Keyword retrieval using token overlap scoring.
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
def ask_llm(query, context, with_rag):
    """Ask the LLM using retrieved context."""
    if with_rag:
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context.strip()}\n\n"
            f"Question: {query}\nAnswer:"
        )
    else:
        prompt = (
            f"Question: {query}\nAnswer:"
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

```

### Putting everything together

The main code for the keyword-based RAG chatbot is given below. 

```python
use_rag = False # Set to True to use keyword RAG, False otherwise

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
    answer = ask_llm(query, context, use_rag)

    print("\n--- Response ---")
    print(answer)
    print("\n" + "=" * 60 + "\n")
```
The script begins with loading and extracting data from the requisite company documents (using the `extract_text_from_pdf` function). Then the chatbot conversation scripting is done in a similar manner to the chatbots lesson in Week 1. For each user query, the relevant context from the documents is retrieved using a simple-keyword matching approach (in the `simple_keyword_retrieval` function). This context is then augmented to the user query and input to the chatbot to generate the response (in the `ask_llm` function).

### Example - Keyword RAG in action

Lets first set `use_rag` to `False` and run the script. If you now ask the query: "Explain the different products offered by Brightleaf Solar" you will find that the chatbot confidently hallucinates a generic answer. We expect the answer to be taken from the product_specs pdf document but this is not what we get. An example response is given below:

```
Brightleaf Solar offers a range of products and services focused on solar energy solutions. Here are the primary offerings:

1. **Solar Energy Systems**: Brightleaf Solar designs and installs custom solar energy systems for residential, commercial, and industrial clients. These systems are tailored to meet the specific energy needs and preferences of each customer.

2. **Solar Panel Installation**: The company provides professional installation services for solar panels, ensuring that the systems are set up efficiently and safely. This includes site assessment, design, and installation.

3. **Energy Storage Solutions**: Brightleaf Solar offers energy storage options, such as battery systems, that allow customers to store excess solar energy for use during non-sunny periods or power outages.

4. **Maintenance and Monitoring Services**: To ensure optimal performance, Brightleaf Solar provides ongoing maintenance and monitoring services for installed solar systems. This includes regular inspections, cleaning, and performance tracking.

5. **Financing Options**: Brightleaf Solar offers various financing solutions to make solar energy more accessible. This may include options like solar loans, leases, or power purchase agreements (PPAs) that allow customers to pay for solar energy over time.

6. **Consultation and Energy Audits**: The company provides consultation services to help customers understand their energy needs and the potential benefits of solar energy. This may include energy audits to assess current energy usage and identify opportunities for savings.

7. **Community Solar Programs**: Brightleaf Solar may also participate in community solar initiatives, allowing customers to benefit from solar energy without needing to install panels on their own property.

These products and services are designed to promote the adoption of solar energy, reduce energy costs, and contribute to a more sustainable future.
```

This type of response is usually obtained from averaging the information on similar solar companies from the internet, recall that the OpenAI GPT is trained on the information available on the internet. Now, set the `use_rag` flag to `True`, rerun the script and ask the same question. Now we expect the response to be richer based on the context obtained from the keyword-based retrieval process. You will notice that the response comes directly from the product_specs pdf:

```
BrightLeaf Solar offers two main products in their HelioPanel series: the HelioPanel X5 and the HelioPanel X7.

1. **HelioPanel X5**:
   - This is BrightLeaf's foundational module, designed for residential and community solar deployments.
   - It features high-efficiency monocrystalline cells and is laminated onto a carbon-fiber composite backsheet, which provides strength while keeping the module lightweight.
   - The X5 module is rated up to 400 watts in full sun and can be paired with microinverters for easy installation.
   - It supports both roof and ground mounting configurations with adjustable tilt brackets to optimize solar exposure throughout the year.
   - For community installations, X5 modules can be arranged in strings that connect to low-voltage AC collection lines, reducing conversion losses and simplifying maintenance.

2. **HelioPanel X7 (2025 Release)**:
   - The X7 builds on the reliability of the X5 while introducing enhancements for performance in diffuse light conditions.
   - It features a multilayer anti-reflective coating that broadens the usable light spectrum and a redesigned thermal layer that helps maintain higher output during overcast weather and high temperatures.
   - The module is constructed with recycled aluminum rails, reducing its embodied carbon footprint while maintaining structural integrity.
   - Integrated sensors provide real-time data on panel temperature, energy yield, and fault codes, which can be monitored through BrightLeaf’s app for predictive maintenance and quicker service responses.
   - Field tests have shown that the X7 delivers a median 17% increase in energy density compared to the X5 under mixed weather conditions.

Both products are manufactured in a facility powered by renewable energy, and BrightLeaf emphasizes sustainability through practices such as recirculating water in their processes and remelting scrap aluminum for use in mounting hardware. They also provide a 25-year performance warranty and a 10-year hardware replacement guarantee for their panels, along with additional services for commercial customers.
```

Congratulations!! Now you have a chatbot that can generate responses to user queries with additional context from an external database! You have created your own RAG enabled chatbot! Feel free to explore with different prompts and test the responses with and without the retrieved context to understand and appreciate the utility of RAG. 

## Food for thought - Evaluating the efficacy of your chatbot

Now you have a chatbot that can generate responses to user queries. But can you guarantee that the responses are better suited to the user queries or more useful? How would you assess this? In the next lesson, we will introduce the deepeval tool which is designed to do just this.

Also, it is useful to note that any perceived improvements to the chatbot's responses are heavily dependent on the external database you provide and the user query. In the example provided here, we provide some documents for Brightleaf Solar Solutions. So it is reasonable to assume that if you ask queries directly related to the content in the documents, you will get a richer, more accurate response from the chatbot. 

Another interesting observation is that for every user query, the content in the documents is searched for relevant context. Depending on the number and size of the documents this can be a computationally expensive process. The scanning approach is also not very scalable, which brings up the need to search the external database more efficiently. This led to the usage of vector dataspaces, which we will look at in the next lesson.