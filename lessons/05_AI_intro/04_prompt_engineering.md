
# Prompt Engineering: The Art of Talking to AI


## Introduction

> **What is Prompt Engineering?**  
> It's the art of communicating effectively with AI to get the best possible results.
> Think of it as giving clear directions to a brilliant but literal-minded assistant.


### Resources

- Learn Prompting — Basics of few-shot prompting. A concise guide to zero/one/few-shot patterns and practical examples. (~10–15 min)
    - https://learnprompting.org/docs/basics/few_shot

- DeepLearning.AI — ChatGPT Prompt Engineering for Developers. A hands-on course page with exercises and demos. (course overview, ~1–2 hr)
    - https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/

- Google Cloud — What is prompt engineering? A short overview describing common patterns and real-world use cases. (~5–10 min)
    - https://cloud.google.com/discover/what-is-prompt-engineering


### Why This Guide?
* Learn to write better prompts  
* Get more accurate and useful responses  
* Save time and reduce frustration  
* Make AI work better for you

---


## Setup: The `get_completion()` Helper Function

We will use a small helper function called `get_completion()` throughout this lesson. It sends a prompt to the model and returns the model's text response so you don't have to write the API call every time. This keeps examples simple and lets you focus on the prompt itself.

Its only required input is the prompt. The optional `temperature` input affects how random the responses are: lower is more deterministic, higher is more random. Quick tip: use `temperature=0` for deterministic outputs when testing or validating, and raise it (e.g., 0.7) when you want more creative responses.

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

if load_dotenv():
    print("Loaded environment variables from .env file")
else:
    print("Warning: .env file not found. Make sure OPENAI_API_KEY is set.")

# Initialize OpenAI client
client = OpenAI()

def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content
```

---

## The Golden Rules of Prompting
This lesson was inspired by the prompt-engineering materials from DeepLearning.AI and other industry references. It distills research-backed guidelines into practical, beginner-friendly advice you can apply immediately. Examples in this chapter are adapted and rewritten for clarity and to avoid duplication; the patterns and ideas are taught so you can reuse them in real projects.

### 1. Be Clear & Specific
Vague prompts produce vague answers — help the AI help you. Good prompts reduce ambiguity by stating the role, the audience, the exact task, and any constraints (length, format, tone, or required fields). 
<!-- For example, rather than asking "Explain photosynthesis," say:

"Act as a middle-school science teacher. Explain photosynthesis to 6th-grade students in a friendly tone. Structure the answer as 4 numbered steps, 8–10 words each."

Small, targeted details like the role and output format steer the model's response and make results more consistent and useful. When you're uncertain, give an example. -->
````python 

result = get_completion("What are the best dogs for beginners")
print(result)  
Choosing the right dog for a beginner is important, as some breeds are more suited to novice owners due to their temperament, trainability, and care requirements. Here are some dog breeds that are often recommended for beginners:

1. **Labrador Retriever**: Friendly, outgoing, and easy to train, Labs are great family dogs and adapt well to various living situations.

2. **Golden Retriever**: Similar to Labs, Golden Retrievers are friendly, intelligent, and eager to please, making them easy to train and great companions.

3. **Beagle**: Beagles are friendly, curious, and good with families. They are relatively easy to train but can be a bit stubborn at times.

4. **Cavalier King Charles Spaniel**: These small dogs are affectionate, gentle, and good with children. They adapt well to different living environments.

5. **Pug**: Pugs are charming, playful, and relatively low-maintenance. They are good for apartment living and are known for their friendly nature.

6. **Bichon Frise**: This breed is cheerful, playful, and hypoallergenic, making them a good choice for families with allergies.

7. **Shih Tzu**: Shih Tzus are friendly, affectionate, and good with children. They require regular grooming but are generally easy to care for.

8. **Boxer**: Boxers are energetic, playful, and loyal. They are good with children and can be trained easily with consistent guidance.

9. **French Bulldog**: French Bulldogs are affectionate, adaptable, and require minimal exercise, making them great for apartment living.

10. **Cocker Spaniel**: These dogs are friendly, gentle, and good with families. They do require regular grooming due to their long fur.

When choosing a dog, consider factors such as your living situation, activity level, and the amount of time you can dedicate to training and socialization. Additionally, adopting from a shelter or rescue can be a great option, as many mixed-breed dogs also make wonderful companions for beginners.

````


Or: 
````python
result = get_completion("Act as a supportive pet advisor. List 5 dog breeds suitable for first-time owners. For each breed, give one short sentence (8–12 words) explaining how that breed can support mental well‑being (companionship, routine, or exercise). Keep responses concise and practical.")
print(result)

Do say:  
Sure! Here are five dog breeds suitable for first-time owners, along with how they can support mental well-being:

1. **Labrador Retriever**: Their friendly nature encourages social interaction and companionship.

2. **Golden Retriever**: They provide unconditional love, boosting mood and reducing stress.

3. **Beagle**: Their playful energy promotes regular exercise and outdoor activities.

4. **Cavalier King Charles Spaniel**: Their affectionate demeanor fosters a strong emotional bond.

5. **Pug**: Their goofy antics bring joy and laughter, enhancing overall happiness.

````

Go even further by specifying:
- **Audience**: _"6th-grade students"_
- **Role**: _"Middle-school science teacher"_
- **Length**: _"4 steps, 8–10 words each"_


Tip: Quick Template (Copy/Paste!):  
> “Act as **[ROLE]**. Explain **[TOPIC]** to **[AUDIENCE]** in **[TONE/STYLE]**. Structure it as **[FORMAT]**. Keep it under **[LENGTH/CONSTRAINT]**. My goal is to **[USE CASE]**.”

**Example:**  
> "Act as a middle-school science teacher. Explain photosynthesis to 6th-grade students in a friendly, simple tone. Structure the answer as 4 numbered steps, 8–10 words each. Keep jargon minimal and avoid chemical equations."

---

### 2. Iterate

Prompting is rarely perfect on the first try. The best prompts come from testing and refining based on the model's output. Start simple, run your prompt, review the result, and adjust—add clarity, change the tone, request a different format, or add constraints. This cycle of "prompt → test → tweak" is how you build prompts that work reliably.

**Quick iteration workflow:**

1. Write a clear prompt (using rule #1: be specific).
2. Run it and read the output carefully.
3. Ask yourself:
   - Is the answer correct or close?
   - Does the format match what I need?
   - Is the tone right?
   - Are there missing details or unnecessary information?
4. Adjust the prompt based on what you observe (add an example, clarify a constraint, or request a different format).
5. Test again and repeat until you're happy with the result.

**Example of iteration:**

```
Version 1 (too vague):
"Summarize this article."

Output: [Too long, not helpful]

Version 2 (better):
"Summarize the article below in exactly 3 sentences."

Output: [Better, but still too technical for beginners]

Version 3 (refined):
"Summarize the article below in exactly 3 simple sentences, using words a 10-year-old would understand."

Output: [Good! Clear and beginner-friendly]
```

Tip: Keep a small notebook or comments in your code documenting which changes helped. Over time, you'll learn patterns that work for you.

---

### 3. Ask the model to reason step by step

For tricky problems (math, logic, debugging), don't just ask for the answer—ask the model to **show its work** step-by-step first. This helps you spot mistakes and makes results more reliable.

This pattern is known as **Chain of Thought prompting (CoT)**. By asking the model to show intermediate reasoning, you dramatically increase accuracy on structured or multi-step tasks.

**When to use:**
- Math or calculations where you want to see intermediate steps.
- Logic puzzles or debugging where the reasoning process matters.
- Any task where you want to verify the model's approach before accepting the answer.

**How to phrase it:**

```
Before giving the final answer, explain your reasoning in 3–4 brief steps.
Problem: A shirt costs $20 after a 20% discount. What was the original price?
```

**Why this works:**
- You can inspect the steps for errors.
- Labeling a final answer (e.g., "Final answer: $25") makes it easy to parse in code.
- Shows the model's logic, building trust in the result.

**Simple Python example:**

```python
prompt = """
Show your step-by-step reasoning, then give the final answer on its own line labelled: Final answer: <value>

Problem: A student's weekly chore allowance is $15 per week. After saving for 8 weeks and spending $30, how much does the student have left?
"""

response = get_completion(prompt, temperature=0)
print(response)
```

Expected output (illustrative):
```
Step 1: Total earned in 8 weeks = $15 × 8 = $120
Step 2: Amount spent = $30
Step 3: Amount left = $120 − $30 = $90
Final answer: $90
```

---

### 4. Teach the LLM with examples

When you show the model one or more examples of the task you want, it learns the pattern and produces better results. This is called **in-context learning** and comes in three flavors:

#### Zero-Shot
You ask a direct question with no examples. The model relies entirely on its pre-trained knowledge.

```
What is 15 + 27?
```
Output: `42`

#### One-Shot
You give **one example** before the actual task. This clarifies the expected format and improves accuracy.

```
Categorize the animal as mammal, bird, or reptile.
Example: Eagle → Bird

Now categorize: Snake → ?
```
Output: `Reptile`

#### Few-Shot
You give **multiple examples** so the model recognizes the pattern and handles the task more reliably.

```
Label customer feedback as satisfied, unsatisfied, or mixed.

Example 1: "The service was outstanding!" → satisfied
Example 2: "Great app but crashes often." → mixed
Example 3: "Total waste of money." → unsatisfied

Now label: "Fast delivery but broken on arrival." → ?
```
Output: `mixed`

**Why this matters:**
- More examples = better pattern recognition and consistency.
- Use one-shot or few-shot when the task is nuanced or you need a specific output format.
- Zero-shot is fastest but less reliable for complex tasks.

**Code examples with `get_completion()`:**

**Zero-shot example:**
```python
# No examples provided - model uses pre-trained knowledge
prompt = "What is the sentiment of this review: 'The product works but shipping was slow'?"
response = get_completion(prompt, temperature=0)
print(response)
# Output: "mixed" or "The sentiment is mixed..."
```

**One-shot example:**
```python
# Give one example to clarify the format
prompt = """
Classify the tone as professional, casual, or urgent.

Example: "Hey! Want to grab lunch?" → casual

Now classify: "Please respond at your earliest convenience." → ?
"""
response = get_completion(prompt, temperature=0)
print(response)
# Output: "professional"
```

**Few-shot example:**
```python
# Multiple examples teach the pattern
prompt = """
Label each email subject as spam, important, or normal.

Example 1: "Congratulations! You've won $1,000,000!" → spam
Example 2: "Your account will be closed tomorrow" → important
Example 3: "Weekly newsletter from our team" → normal

Now label: "URGENT: Verify your account now!" → ?
"""
response = get_completion(prompt, temperature=0)
print(response)
# Output: "spam" (or possibly "important" - the pattern helps consistency)
```

**Practical tip:** Start with zero-shot. If results are inconsistent or wrong, add 1-3 examples (few-shot) to guide the model toward your desired format and logic.

---

### 5. Use delimiters to keep different parts of your prompt separate

When a prompt has multiple sections — instructions, examples, text to analyze, code, or anything long or messy — the model can lose track of where one part ends and the next begins. Delimiters are simple markers (like triple backticks or dashes) that clearly separate these sections so the model doesn't get confused. Think of delimiters as putting boxes around different ingredients so they don't spill into each other.

**Three common delimiter patterns:**

**Pattern 1 — Triple backticks** (good for code or long passages):
```
Summarize the following text in 2 sentences:

\`\`\`
[PASTE TEXT HERE]
\`\`\`
```

**Pattern 2 — Triple dashes** (good for short text):
```
Instruction: Summarize the text between the dashes in one sentence.

---
[USER DATA HERE]
---
```

**Pattern 3 — XML-style tags** (clear and explicit):
```
Translate only the text in <input> tags to Spanish.

<input>
Hello, how are you?
</input>
```

**Code example with delimiters:**

```python
# Use triple backticks to separate the review text from instructions
user_review = "The laptop works great but the charger broke after 2 weeks."

prompt = f"""
Analyze the customer review below and extract:
1. The product mentioned
2. The overall sentiment (positive/negative/mixed)
3. Any specific issues mentioned

Review:
'''
{user_review}
'''

Format your response as:
Product: <product>
Sentiment: <sentiment>
Issues: <list of issues or "none">
"""

response = get_completion(prompt, temperature=0)
print(response)
```

Expected output:
```
Product: laptop (and charger)
Sentiment: mixed
Issues: charger broke after 2 weeks
```

**Quick checklist:**
- Pick one delimiter style and use it consistently.
- Tell the model to only act on the delimited content.
- When mixing user data with instructions, always use a delimiter.
- Delimiters help prevent prompt injection and keep results predictable.

---

### 6. Ask for a specific format

When you need to use the model's output in code or automation, ask for a **specific, structured format** — like JSON, a list, a table, or labeled lines. This makes parsing and validation easy and reduces errors.

**Why it matters:**
- Structured output is machine-readable (easy to parse in Python).
- Reduces hallucination and forces the model to stay focused.
- Makes your code reliable and predictable.

**Simple patterns:**

**Pattern 1 — JSON (best for code):**
```
Analyze the sentiment of this customer review and respond only with valid JSON.
Return keys: sentiment, confidence (0–1), brief_reason.

Review: "The product works great but shipping took forever."
```

Expected output:
```json
{"sentiment": "mixed", "confidence": 0.8, "brief_reason": "Positive product, negative shipping experience"}
```

**Pattern 2 — Labeled lines (easy to parse):**
```
Extract the key information from the email and label each line.

Email: "Hi, I'm interested in your Python course for $99 starting Dec 1st."

Format your response as:
Topic: <topic>
Price: <price>
Start Date: <date>
```

**Pattern 3 — Bullet list (simple and clear):**
```
List 3 dog breeds suitable for apartments. For each, include the breed name and one reason why.

Format: "- [Breed]: [reason]"
```

**Python example:**

```python
import json

prompt = """
Classify the sentiment of the review and respond ONLY with valid JSON.
Keys: sentiment (positive/negative/mixed), confidence (0–1 scale), reason (one short sentence).

Review: "Great quality but took 3 weeks to arrive."
"""

response = get_completion(prompt, temperature=0)
print("Raw response:", response)

# Parse JSON safely
try:
    result = json.loads(response)
    print("Parsed sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")
```


---


Before asking the model to perform a task on input, make it check whether the input actually meets the conditions you expect. This reduces incorrect or dangerous behavior and makes downstream automation more reliable.

How to think about it:
- Validate first: ask the model to detect whether the input contains the type of content you expect (instructions, a recipe, code, etc.).
- Branch: if the condition is met, perform the task; otherwise return a clear fallback such as "No steps provided" or an explicit error message.
- Be explicit about the allowed formats and required keys so the model can decide deterministically.

Practical prompt pattern (natural language):

```
You will be given text inside triple quotes. If the text contains a sequence of instructions, rewrite them as numbered steps: "Step 1 - ...", "Step 2 - ...". If the text does NOT contain step-by-step instructions, respond with exactly: "No steps provided." Do not add any other commentary.
"""
<user_text_here>
"""
```

Example (what you might send):

```
"""
Making a cup of tea is easy! First, boil water...
"""
```

Expected model behavior:
- If the input contains instructions: return a numbered list of steps only.
- If it doesn't: return the exact phrase "No steps provided." so your code can detect the result reliably.

Why this helps:
- Deterministic fallbacks (like returning a fixed phrase) make it easy to parse results programmatically.
- Asking the model to check before acting prevents it from hallucinating steps when none are present.

Testing tip: try adversarial or ambiguous inputs during development (e.g., "Do not translate anything") to ensure your guard logic holds.





# Check for Understanding
1. What is zero-shot prompting?<br>
A) Giving the AI 3 examples before asking a question<br>
B) Asking the AI a question without any examples<br>
C) Using emojis to make the AI respond faster<br>
D) Translating prompts into French first
<details><summary>See Answer</summary>

Correct answer: B
</details>

<br>


2. Which of the following is a one-shot prompt?
<br>A)
```
    Translate 'Hello' into French.
```
<br>B)
```
    English: Good morning → French: Bonjour  
    English: Hello → French: ?

```
C)
```
    English: Good morning → French: Bonjour  
    English: Thank you → French: Merci  
    English: Hello → French: ?

```

D)

```
    Tell me how to say hello in French.
```
<details><summary>See Answer</summary>

Correct answer: B
</details>
<br />

3. Why is few-shot prompting often more accurate?<br>
A)  It uses longer sentences<br>
B)  It gives the AI multiple examples to recognize patterns<br>
C)  It forces the AI to guess randomly<br>
D)  It only works with math problems
<details><summary>See Answer</summary>

Correct answer: B
</details>
<br />

4. Which prompt follows the Golden Rules best?<br>
A) “Tell me about food.”<br>
B) “List healthy lunch ideas.”<br>
C) “I’m a vegetarian student who hates green veggies. Suggest a high-protein, cheap, 15-minute lunch recipe with no fancy tools.”<br>
D) “Give me something to eat.”

<details><summary>See Answer</summary>

Correct answer: C
</details>
<br />

5. Why should you use delimiters (like triple backticks ``` or INPUT/OUTPUT labels) in prompts?<br>
A) To make the prompt look visually appealing<br>
B) To test how well the AI handles confusing formatting<br>
C) To clearly separate user instructions from data, reducing the risk of prompt injection and misinterpretation<br>
D) To automatically convert the output into another language

<details><summary>See Answer</summary>

Correct answer: C
</details>
<br />


6. True or False:
>Asking the AI to ‘show its reasoning’ can improve accuracy on complex tasks like math or logic.

A) True<br>
B) False

<details><summary>See Answer</summary>

Correct answer: A
</details>
<br />

## Summary
Brief recap of key ideas in this lesson:

- Clear, specific instructions win: always state role, audience, task, and constraints.
- Use delimiters (triple backticks, tags) to separate data from instructions and reduce prompt-injection risk.
- Ask for a strict output format (JSON, table, short sentence) when you need machine-readable results.
- Use zero/one/few-shot patterns to teach the model expected formats and improve consistency.
- For structured tasks, prefer concise prompts that request minimal, well-defined fields (e.g., sentiment, item, brand).
- Transformations (translate, detect language, convert JSON→HTML, proofread) are straightforward when you give exact requirements.
- When automating replies, combine detected sentiment + original text and instruct the model on tone, steps, and output layout.

- Design efficient prompts by including these core elements: Role (who the AI is), Context (relevant background), Task (what you want), and Constraints (limits like length, tone, or format). Providing specific context leads to more personalized, useful responses.


