# Keyword-based RAG
In the previous lesson we introduced the idea of adding knowledge to a model using RAG. In this lesson, we will dive into our first real example: a very simple keyword-based RAG system. This is just to illustrate a minimal working RAG pipeline. It is purposely very simple, and too brittle to use in real life. We will build up to an implementation closer to the framework used in industry. At the highest level of abstraction every RAG approach follows the procedure shown in the figure below (taken from the Basic RAG databricks article). 

<!-- > **NOTE:**  To get the most out of this lesson, it is recommended that you run the examples yourself. Before beginning with the exercises, ensure that you have your OpenAI API key. If you don't have this, please reach out to your mentor. To run the exercises, you will need to create a virtual environment with the following packages installed: `OpenAI`, `pypdf`, and `dotenv`.-->

Here are some additional resources for your reference: [Basic RAG databricks article](https://docs.databricks.com/aws/en/generative-ai/retrieval-augmented-generation), [OpenAI RAG article](https://help.openai.com/en/articles/8868588-retrieval-augmented-generation-rag-and-semantic-search-for-gpts), [RAG vs Fine Tuning vs Prompt Engineering](https://www.youtube.com/watch?v=zYGDpG-pTho).

![Basic Rag Framework](resources/basic_rag.png)

The chatbot generates a response based on the user prompt and relevant supporting data from an external data repository (which can be PDFs, CSV files, databases, or any other representation of data) that is not captured by the data used to train the chatbot. A good example of this is proprietary company information that is not in the public domain. The different RAG approaches differ in the methods used to determine the data relevant to the user prompt from the data repository. The most basic approach to isolate relevant data from th external data source is a simple keyword matching search. Let us go over an example implementation of this approach.

## Implementation

For this example, assume that you are developing a customer service chatbot for a fictitious company called Brightleaf Solar Solutions. The chatbot has access to the company's proprietary documents (present in the "brightleaf_pdfs" directory) and must be able to leverage this data to improve its responses to user queries. The company's documents include its mission statement, product specifications, an earnings report, details about its partnerships, and its policies on security and employee benefits. Go through the documents as closely as you can so you can understand how information in these documents influence the generated responses. Here we will only concern ourselves with a simple Q&A loop in order to understand the RAG process through the most barebones example.

### Setting up

Note that in this example we store and retrieve the OpenAI API Key in the `.env` file. Then, we create the OpenAI API client object as we ave done before.

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

We will now go over each of the building blocks in the keyword-based RAG framework in detail before combining them together into the final Q&A loop.

### Extract and store text from documents

In the `extract_text_from_pdf` function, the text in each document in the system path represented by the `pdf_path` variable is extracted and stored in the `text` variable. This function is called for each document in the directory. 

> Note: In real RAG systems we usually split documents into smaller "chunks" and retrieve only a few relevant chunks. Here we keep things deliberately simple and retrieve at the document level so we can focus on the core idea of retrieval + generation. Moreover, the additional complexity and computational benefits of retrieving paragraph-sized chunks instead of a document's full text is not worth the benefits for this example.

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

`PDFReader` is a particularly useful method to retrieve the text from a pdf file. You can learn more about it [here](https://pypdf.readthedocs.io/en/stable/modules/PdfReader.html). Given a path to the file (and assuming that the file exists) it outputs a reader object that contains the text within the pages. Here, we simply loop through each page in the reader object, extract the text from each page and stack them into the list called `text`. The `strip()` function simply removes any leading or trailing whitespace from the text. Thus at the end the `text` object will simply be a list of strings, each element being the text in a page. For example, the "earnings_report.pdf" file only contains one page with text. So `text` will be a list of length 1 with the only element being the following.

<code>
"Overview\nThis report summarizes BrightLeaf Solar's financial performance from 2021 through 2025. The period\nincludes a growth phase, a temporary dip in 2023 due to supply constraints, and recovery in 2024-2025\ndriven by operations improvements and demand for the HelioPanel line. Our priority remains\nsustainable margins while investing in R&D; and workforce development.\nFinancial Summary (USD, millions)\n2021 Revenue: 2.8 | Net Profit: 0.3\n2022 Revenue: 4.5 | Net Profit: 0.8\n2023 Revenue: 4.0 | Net Profit: 0.5\n2024 Revenue: 6.2 | Net Profit: 1.1\n2025 Revenue: 7.1 | Net Profit: 1.3\nYear-by-Year Commentary\n●\n2021. First full year of HelioPanel X5 shipments. Community and residential projects established\nsteady orders and validated the modular design.\n●\n2022. Growth from multi-site deployments and school microgrids. Gross margin improved via vendor\nconsolidation and higher lamination yields.\n●\n2023. Temporary dip driven by polysilicon price volatility, freight bottlenecks, and a pause in a utility\ninterconnection queue that delayed installs.\n●\nLate 2023. Stabilization began as regional suppliers were qualified and multi-modal shipping\nreduced logistics risk.\n●\n2024. Efficiency programs and field-service telemetry cut downtime; pre-orders for HelioPanel X7\nsupported top-line growth.\n●\n2025. First commercial X7 installs plus SunSpan partnership at Midwestern industrial sites increased\nutilization and restored margin momentum.\nDrivers, Risks, and Mitigations\nRecovery drivers included localized sourcing, improved inverter interoperability, and predictive\nmaintenance from onboard sensors. Key risks remain input-price volatility, interconnection-policy\nuncertainty, and weather-related construction delays. Mitigations include multi-supplier contracts,\ninventory buffers for critical parts, and expanded installer training to shorten commissioning windows.\nForward Outlook\nBrightLeaf expects steady growth into 2026 as X7 adoption broadens and community-scale hybrid\nsystems mature. Investments will continue in tandem-cell R&D;, recycling pathways, and grid-services\nintegration. The objective remains sustainable revenue expansion aligned with product reliability,\ncustomer value, and community impact."
</code>

This can be slightly disconcerting to look at, but on closer inspection and comparison to the PDF file you will recognize that the `\n` symbols depict the line breaks and there is no distinguishing between the titles and content. Nonetheless, the content of the page is present here which is what we are after.

Now that we have the data from the company documents, we will create the function that retrieves the relevant context from the documents based on the user query. 

### Keyword-based retrieval

This is the main engine of this framework and the key step in improving the quality of the response. This is the "Retrieve" step in the image at the beginning of the lesson. It takes in the user prompt and the database of documents and outputs the supporting data that will be augmented to the user prompt. Lets look at this process through the function for keyword-based context retrieval shown below. It is not necessary to understand the code itself as much as what the code it doing.

```python
def simple_keyword_retrieval(query, documents, verbose=True):
    """
    Keyword retrieval using token overlap scoring.
    - Removes stopwords and punctuation for cleaner matching.
    - Returns the single best-matching document.
    - `documents`: dictionary with the document names as keys and the text as values, extracted using the `extract_text_from_pdf` function.
    """
    import string

    stopwords = [
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    ]

    # Translator to remove punctuation (so "Solar?" -> "Solar")
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
The main goal of the keyword-based retrieval approach is to retrieve the relevant context to the user prompt based on keyword matching. To do this, three main steps are done:
- Remove irrelevant words such as articles, prepositions etc. (stored in the `stopwords` list) and punctuations (using the string `maketrans` method)
- Store the reduced words as elements in a set
- Compare the sets of reduced words in the query and text in each document to determine the overlap score

The overlap score is computed for each document and is simply the number of common words between the query and text in the document. This score is stored in the `scores` list as a tuple with the document name and content. Finally, the document name and content corresponding to the highest overlap score is obtained.  

> Note that best returns a tuple with the document name and content **only** if the score is positive (*think about why this is the case and view the details below to see if you agree*).

<details>
<summary> View Details </summary>
Since the main goal of Retrieval Augmented Generation is to improve the quality of the response to the user query, it is important to augment the user query with context that has at least some relevance to it. If the score for a document is 0, this indicates that there is no relevant similarity between the text in the document and the user query. Augmenting this text with the user query could decrease the quality of the response (maybe even cause the API to hallucinate more). In this case, it is better not to augment any text to the user query.
</details>

Verbosity (the `verbose` boolean argument) is a common feature in many library functions. Setting the argument to true prints statements that allow a greater understanding of the inner workings of the method. In this case, setting `verbose=True` prints the `query_words` set, the overlap score for each document and the name and content for the document with the highest positive overlap score (if one exists). 

Let's see an example output. If we ask the query: "What is the mission of Brightleaf Solar?" and keep `verbose=True`, we get the following outputs:
```
Query tokens (filtered): ['brightleaf', 'mission', 'solar', 'what']
[earnings_report.pdf] overlap=1 -> ['brightleaf']
[employee_benefits.pdf] overlap=2 -> ['brightleaf', 'solar']
[mission_statement.pdf] overlap=3 -> ['brightleaf', 'mission', 'solar']
[partnerships.pdf] overlap=2 -> ['brightleaf', 'solar']
[product_specs.pdf] overlap=2 -> ['brightleaf', 'solar']
[security_policy.pdf] overlap=1 -> ['brightleaf']
Selected best match: mission_statement.pdf
```
It is clear to see that the documents are being ranked based on the number of exact keyword matches with the query. Since "mission_statement.pdf" has the highest number of matches, its text will be used as relevant content for the query.

Now that we can extract text from the documents and retrieve the relevant context from them for a given user prompt, next we augment this context to the user prompt and feed it to the LLM.

### Augment with context and generate response

<!--The other important factor determining the quality of a RAG framework is the "Generator," i.e. the way the retrieved context and user prompt is augmented to generate the response from the LLM. It is interesting to note that prompt engineering is a very useful skill to have for this purpose.  -->
The `ask_llm` function to generate the API response based on the user prompt augmented with the retrieved context is shown below. This corresponds to the "Augment" and "Generate" steps in the figure from the beginning of the lesson. 

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
An additional `with_rag` parameter is included as an argument to the method to test the LLM output without the retrieved context. 

Note that we clearly demarcate the retrieved context and the user query with appropriate labels in the API prompt and add the instruction to "use the context to answer the question". 

> If we remove the labels and simply augmented the user query and context, do you think the response would change? Would it improve or worsen?

The code to get the response from the API should be familiar from the chat completions lesson.

```python
response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
response.choices[0].message.content.strip()
```

Now we have all the building blocks in our keyword-based RAG framwork. Lets put it all together.

### Putting everything together

The main code for the keyword-based RAG Q&A loop is given below. 

```python
use_rag = True # Set to True to use keyword RAG, False otherwise

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
We first set the `use_rag` variable depending on the case we're investigating, here it is set to `True` implying that we want to use the keyword-based RAG framework. Here are the broad steps that take place in the execution of this script:
- The script begins with loading and extracting data from the requisite company documents (using the `extract_text_from_pdf` function). 
- Then the conversation scripting is done in the form of a simple infinite while loop (note the use of `while True:`). For each user query, the relevant context from the documents is retrieved using a simple-keyword matching approach (in the `simple_keyword_retrieval` function). 
- This context is then augmented to the user query and input to the API to generate the response (in the `ask_llm` function). This loop continues until the user inputs either "quit" or "exit."

```python
if query.lower() in {"quit", "exit"}:
    print("Goodbye.")
    break
```
Note that this is not a continuous conversation as the LLM is not aware of the previous conversation so far. So this cannot be classified as a chatbot, however it can be trivially extended to include past context to the LLM (refer to the Chatbots lesson). 

That's all there is to this! Hopefully this rather crude and basic example helps you to see the RAG process in the most barebones fashion. To drive home the power of RAG (even at its most basic), lets look at an example. 

### Example - Keyword RAG in action

Lets first set `use_rag` to `False` and run the script. If you now ask the query: "Explain the different products offered by Brightleaf Solar" you will find that the API confidently hallucinates a generic answer. We expect the answer to be taken from the product_specs pdf document but this is not what we get. An example response is given below:

<code>
Brightleaf Solar offers a range of products and services focused on solar energy solutions. Here are the primary offerings:

1. **Solar Energy Systems**: Brightleaf Solar designs and installs custom solar energy systems for residential, commercial, and industrial clients. These systems are tailored to meet the specific energy needs and preferences of each customer.
2. **Solar Panel Installation**: The company provides professional installation services for solar panels, ensuring that the systems are set up efficiently and safely. This includes site assessment, design, and installation.
3. **Energy Storage Solutions**: Brightleaf Solar offers energy storage options, such as battery systems, that allow customers to store excess solar energy for use during non-sunny periods or power outages.
4. **Maintenance and Monitoring Services**: To ensure optimal performance, Brightleaf Solar provides ongoing maintenance and monitoring services for installed solar systems. This includes regular inspections, cleaning, and performance tracking.
5. **Financing Options**: Brightleaf Solar offers various financing solutions to make solar energy more accessible. This may include options like solar loans, leases, or power purchase agreements (PPAs) that allow customers to pay for solar energy over time.
6. **Consultation and Energy Audits**: The company provides consultation services to help customers understand their energy needs and the potential benefits of solar energy. This may include energy audits to assess current energy usage and identify opportunities for savings.
7. **Community Solar Programs**: Brightleaf Solar may also participate in community solar initiatives, allowing customers to benefit from solar energy without needing to install panels on their own property.

These products and services are designed to promote the adoption of solar energy, reduce energy costs, and contribute to a more sustainable future.
</code>

This type of response is usually obtained from averaging the information on similar solar companies from the internet, recall that the OpenAI GPT is trained on the information available on the internet. Now, set the `use_rag` flag to `True`, rerun the script and ask the same question. Now we expect the response to be richer based on the context obtained from the keyword-based retrieval process. You will notice that the response comes directly from the product_specs pdf:

<code>
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
</code>

Congratulations!! Now you have a simple Q&A loop that can generate responses to user queries with additional context from an external database! Feel free to explore with different prompts and test the responses with and without the retrieved context to understand and appreciate the utility of RAG. This basic implementation was simply meant to get your hands wet on a simple RAG framework and focus on the main mechanisms behind RAG. In the next lesson, we will look at a more elaborate framework that improves both the context search efficiency and quality of the retrieved context.

## Food for thought - Evaluating the efficacy of your chatbot

Now you have a RAG-enabled framework that can generate responses to user queries. But can you guarantee that the responses are better suited to the user queries or more useful? How would you assess this? In a subsequent lesson, we will introduce the deepeval tool which is designed to do just this.

Also, it is useful to note that any perceived improvements to the chatbot's responses are heavily dependent on the external database you provide and the user query. In the example provided here, we provide some documents for Brightleaf Solar Solutions. So it is reasonable to assume that if you ask queries directly related to the content in the documents, you will get a richer, more accurate response from the chatbot. 

Another interesting observation is that for every user query, the content in the documents is searched for relevant context. Depending on the number and size of the documents this can be a computationally expensive process. The scanning approach is also not very scalable. The keyword matching based approach is very brittle. Since you are comparing the query and document text for the **same** words, you lose the nuance of text that is similar in content even if the same words are not used. This  brings up the need to search the external database more efficiently while accounting for this nuance. This led to the usage of vector dataspaces, which we will look at in a subsequent lesson.

## Check for Understanding

### Question 1

In this keyword-based RAG implementation, on what basis is the most relevant context to the user query obtained?

Choices:
- A. Semantic similarity
- B. Highest number of keyword matches
- C. Highest number of keywords
- D. Random search


<details>
<summary> View Answer </summary>
<strong>Answer:</strong> B. Highest number of keyword matches <br>
The context retrieval implementation outputs the document text with the most number of keyword matches to the user query.
</details>

### Question 2

Which of the following is **NOT** a drawback of the keyword RAG process? 

Choices:
- A. It appends the entire text of the relevant document as context for the user query
- B. It uses relevant context from documents to improve the accuracy of the response
- C. It is not able to understand semantic similarity between keywords in the query and text in the documents
- D. It is inefficient in terms of token usage


<details>
<summary> View Answer </summary>
<strong>Answer:</strong> B. It uses relevant context from documents to improve the accuracy of the response <br>
The fact that keyowrd RAG uses the additional context from relevant documents to improve response accuracy is a plus and a key reason to use RAG in the first place. While it does retrieve the relevant document to the query, appending the entire text of the document to the query results in high token usage. This can get expensive very quickly. Also, since it is merely matching keywords exactly it is not able to understand and leverage semantic similarity (similar meaning words or similar context words) between the user query and documents text. This can result in the omission of particularly relevant context that is not worded similarly.
</details>