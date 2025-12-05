## Six Simple Rules for Prompting (Python 200)

Below are six short, high-impact rules to get students started with prompting. Each rule is brief with one or two short examples so learners can try them quickly.

### 1. Be Clear and Specific
State the role, the audience, the task, and any format or length constraints. This is the biggest win for better outputs.

Bad (vague):
```
What are the best dogs for beginners?
```

Better (clear and specific):
```
Act as a supportive pet advisor. List 5 dog breeds suitable for first-time owners. For each breed, give one short sentence (8–12 words) explaining why it's a good fit.
```

Example outputs (show, don't tell):

Output (vague prompt):
```
Labrador Retriever - friendly and good with families.
Poodle - smart and hypoallergenic.
Beagle - energetic and good with kids.
```

Output (clear prompt):
```
1. Labrador Retriever — calm, friendly family dog, easy to train.
2. Cavalier King Charles Spaniel — gentle companion, low exercise needs.
3. Poodle — intelligent, low-shed coat, great for allergy-prone owners.
4. Beagle — friendly, good with kids, needs regular walks.
5. Greyhound — quiet at home, gentle and low-maintenance grooming.
```

(Students: copy both prompts and compare the outputs to see the difference.)

### 2. Iterate — Prompting Is Interactive
Your first prompt rarely needs to be perfect. Try small changes: tweak role, length, or output format and compare results. Encourage students to experiment and note what changed.

Quick tip: change one thing at a time (role → format → constraints) so you can see the effect.

### 3. Ask the Model to Reason Step‑by‑Step (when needed)
For logic, math, or multi-step problems ask the model to show 2–5 brief steps then a labelled final line. Keep the steps short.

Prompt example:
```
Explain your reasoning in 2–4 brief steps, then give the final result on a line labeled exactly: Final answer: <value>
Problem: What is 123 × 45?
```

Sample model output (shortened):
```
Step 1: 123 × 40 = 4920
Step 2: 123 × 5 = 615
Final answer: 5535
```

Note: don't introduce parsing tricks (regex) in this intro — keep the pattern simple and readable.

### 4. Teach the LLM with Examples (0 → 1 → Few)
Introduce one unifying task and show zero-shot, one-shot, and few-shot prompts so students can see how examples change results. Use the same task for all three.

Task: classify tone as `formal`, `casual`, or `urgent`.

Zero-shot (no example):
```
Classify the tone of the text below as formal, casual, or urgent.
Text: "Can you send me the report by Monday?"
```

One-shot (single example):
```
Example:
Text: "Dear Professor, I would like to discuss my grade."
Label: formal

Now classify:
Text: "Can you send me the report by Monday?"
```

Few-shot (multiple examples):
```
Text: "Dear Professor, I would like to discuss my grade." → formal
Text: "Hey, can we grab coffee later?" → casual
Text: "Server is down — fix immediately!" → urgent

Now classify:
Text: "Can you send me the report by Monday?"
```

Students: run each prompt and compare the model's answers. The examples teach the model the format and style you expect.

### 5. Use Delimiters to Separate Instructions from Data
When you mix instructions with user content, wrap the content in a delimiter so the model treats it as data.

Example (triple backticks):
```
Summarize the text between the backticks in one sentence:
```
```
Here is a paragraph to summarize.
```

Short note: delimiter patterns (``` or `<input>...</input>`) are a simple way to reduce accidental instruction mixing. The detailed prompt‑injection checklist is advanced — we can cover that later.

### 6. Ask for a Specific Format (JSON for code)
If you plan to consume the output in code, ask for a strict format such as JSON.

Prompt example:
```
Extract the product name and price from the text below. Respond as JSON with keys: name, price.
Text: "Widget Pro — $19.99"
```

Sample output:
```
{"name": "Widget Pro", "price": 19.99}
```

Keep this last: formats are important for automation but come after students understand clarity and iteration.

---

Now continue with short, hands-on examples below (language detection, JSON→HTML, and other small transformations). The advanced automation patterns (validation-first, deep prompt-injection defenses, heavy parsing/regex) are intentionally omitted from this intro and can be added later in a follow-up lesson.


# Prompt Engineering: The Art of Talking to AI


## Introduction

> **What is Prompt Engineering?**  
> It's the art of communicating effectively with AI to get the best possible results.
print(response)
```

Continue below with the multi-step prompt example.


```
Next example: a multi-step prompt that summarizes delimited text, translates to Spanish, extracts key topics, and returns a compact JSON object with strict keys.
```python
prompt_2 = f"""
Your task is to perform the following actions: 
1 - Summarize the following text delimited by 
  <> with 1 sentence.
2 - Translate the summary into Spanish.
3 - List the main topics mentioned in the Spanish summary.
4 - Output a json object that contains the 
  following keys: spanish_summary, num_topics.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Topics: <list of main topics in summary>
Output JSON: <json with summary and num_topics>

Text: <{text}>
"""
response = get_completion(prompt_2)
print("\nCompletion for prompt 2:")
print(response)

```
A short example: Detect the language of a text and return a compact JSON result.

```python
prompt = f"""
Identify the language and ISO 639-1 code of the text below.
Respond as JSON with keys: language, iso639_1.
Text: ```Bonjour, comment puis-je vous aider aujourd'hui ?```
"""
response = get_completion(prompt)
print(response)


```
Another transformation: JSON → HTML
Tell the AI how to give you the answer—especially if you’re using it in code or spreadsheets.

Do this:
```
Generate 3 made-up book titles with authors and genres.
Provide them in JSON with keys: book_id, title, author, genre.
``` 



## 3. Ask the AI to Check Conditions First
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

### The coding version

Below is a minimal Python example that implements the same idea with the OpenAI API. It assumes your OPENAI_API_KEY is loaded (e.g., via dotenv) and a helper get_completion(prompt) is available.
```python 
prompt = f"""
First, check if the input below contains both a clear 'Question' and a 'Student\'s Solution' with numeric values we can evaluate. Respond ONLY with a JSON object with these keys:
    - can_evaluate: true or false
    - reason: short explanation if can_evaluate is false (e.g., "missing student solution", "ambiguous numbers")
    - If can_evaluate is true, include these additional keys:
        - correct: "yes" or "no"
        - explanation: brief reasoning (2-4 short sentences)
        - correct_answer: numeric total (e.g., 1450)

Now evaluate the input below following that rule.

Input:
Question:
A food truck owner is calculating monthly expenses. \
The truck costs are:
- Truck lease: $800 per month flat fee
- Ingredients: $12 per day
- Gas: $8 per day
- Business license: $150 per month flat fee
If the owner operates 25 days per month, what are the total monthly expenses?

Student's Solution:
Daily variable costs: $12 + $8 = $20 per day
Monthly variable costs: $20 × 25 = $500
Fixed monthly costs: $800 + $150 = $950
Total monthly expenses: $500 + $950 = $1,450
"""
response = get_completion(prompt)
print(response)


```
The AI now checks first, then responds appropriately.

## 4. Give the Model Time to “Think”
For complex or multi-step problems, explicitly ask the model to show its chain of thought — i.e., reason step-by-step — before giving the final result. When the model outlines intermediate steps you can inspect them for mistakes, which reduces silent errors and makes automated checks easier.

When to ask for step-by-step reasoning:
- Math and numeric calculations where intermediate operations should be visible.
- Logic, debugging, or code explanations where the process matters as much as the answer.
- Any task where you want to validate the model's approach before accepting the final output.

How to phrase it (copy-paste):

```
Show your step-by-step reasoning to calculate 123 × 456. After the steps, print a single line labelled: Final answer: <value>
```

Why this helps:
- You can programmatically check the labelled final answer and, if needed, re-run or flag the response when the steps don't match the final value.
- Asking for a labelled final line (e.g., "Final answer:") makes parsing deterministic and reduces ambiguity in automated workflows.

Caveats and tips:
- Keep the requested level of detail bounded (e.g., "brief steps") to avoid overly long outputs.
- Use temperature=0 for deterministic results when testing or running checks.
- Be aware that asking for internal chain-of-thought can increase token usage and latency; use it when the extra safety is worth the cost.

Example prompt you can reuse:

```
Explain your reasoning in 3–6 brief steps, then give the final answer on a single line labelled exactly: Final answer: <value>
Problem: What is 123 × 456?
```

This pattern improves accuracy and builds trust because it exposes the model's reasoning for inspection before you accept the result.
### The coding version

Below is a minimal Python example that implements the same idea with the OpenAI API. It assumes your OPENAI_API_KEY is loaded (e.g., via dotenv) and a helper get_completion(prompt) is available.
```python
prompt = f"""
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem including the final total. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.

Use the following format:
Question:
question here

Student's solution:
student's solution here

Actual solution:
steps to work out the solution and your solution here

Is the student's solution the same as actual solution \
just calculated:
yes or no

Student grade:
correct or incorrect


Question:

A bakery is planning its weekly flour order. They bake \
three types of bread daily:
- Sourdough loaves use 2 pounds of flour each, and they bake 15 loaves per day
- Baguettes use 0.5 pounds of flour each, and they bake 40 per day
- Whole wheat loaves use 1.5 pounds of flour each, and they bake 20 per day
How many pounds of flour does the bakery need for one week (7 days)?


Student's solution:

Daily flour usage:
- Sourdough: 15 × 2 = 30 pounds
- Baguettes: 40 × 0.5 = 20 pounds
- Whole wheat: 20 × 1.5 = 30 pounds
Total per day: 30 + 20 + 30 = 70 pounds
Weekly total: 70 × 7 = 490 pounds


Actual solution:
"""

response = get_completion(prompt)
print(response)


```

## 5. Ask the Model to Reason Through Its Own Solution
Great for math, logic, or debugging.

“A shirt costs $20 after a 20% discount. What was the original price?
First, explain your reasoning. Then give the answer.” 

The AI is more accurate when it “shows its work”!

Tip: Iterate!
Your first prompt doesn’t have to be perfect.
If the answer isn’t quite right, tweak and try again—that’s how you learn what works!


### The coding version

This example asks the model to show its reasoning before giving a final answer, then parses the labelled final line. It's useful when you want to audit the model's steps and programmatically verify results.

```python
import re
from typing import Optional

# Assumes get_completion(prompt, model, temperature) helper is defined earlier in this lesson.

prompt = f"""
Explain your reasoning in 3–6 brief steps, then give the final answer on a single line labelled exactly: Final answer: <value>
Problem: A shirt costs $20 after a 20% discount. What was the original price?
"""

response = get_completion(prompt, model="gpt-3.5-turbo", temperature=0)
print("MODEL RESPONSE:\n", response)

# Find the Final answer line
match = re.search(r"^Final answer:\s*(.+)$", response, flags=re.MULTILINE | re.IGNORECASE)
final_value: Optional[str] = match.group(1).strip() if match else None

if final_value:
    print("Parsed final answer:", final_value)
    # Optional: try to parse numeric value and validate independently
    try:
        numeric = float(re.sub(r"[^0-9.-]", "", final_value))
        # Independent check: original_price = discounted_price / (1 - discount_rate)
        discounted_price = 20.0
        discount_rate = 0.20
        expected = discounted_price / (1 - discount_rate)
        print(f"Independent check: expected {expected:.2f}")
        if abs(numeric - expected) < 0.01:
            print("Validation: PASSED")
        else:
            print("Validation: FAILED (model final value does not match computation)")
    except ValueError:
        print("Could not parse numeric value from final answer for validation.")
else:
    print("No labelled final answer found. Consider asking the model to include 'Final answer:' on its own line.")
```

Notes:
- Keep the steps brief to avoid excessive token use.
- Use deterministic settings (temperature=0) for testing and validation flows.
- If you need higher assurance, parse the intermediate steps and compute the result yourself rather than trusting the final line.




## Zero-Shot, One-Shot, Few Shot

### Zero-Shot

Zero-shot is a direct question to the llm (like ChatGPT). According to the learning prompt article, 
>"No examples are provided, and the model must reply entirely on its pre-trained knowledge."

It is the most direct form of prompting. An example would be. 
#### Example
**Prompt**
```

Identify the primary color in the following item:<br>
Item: A ripe banana<br>
Color:

Yellow

```

**Prompt**

"What is 15 + 27?"

```
42

```

### One-Shot

One-shot prompting is a technique in **in-context learning (ICL)** where the model is given **a single example** before the actual task. This helps clarify the expected format, style, or logic—leading to better performance than zero-shot prompting.

Note: According to the learning prompt article:  
*"One-shot prompting enhances zero-shot prompting by providing a single example before the new task, which helps clarify expectations and improves model performance."*

#### Example 1:
**Prompt:**
Categorize the following animal as mammal, bird, or reptile.
Animal: Eagle
Category: Bird

Animal: Snake
Category: ?
```
Reptile
```

#### Example 2:
**Prompt:**
Convert the temperature from Fahrenheit to Celsius (formula: C = (F - 32) × 5/9).
Temperature: 32°F
Celsius: 0°C

Temperature: 68°F
Celsius: ?
```
20°C

```

### Few-Shot
This is when multiple examples are fed into the llm.
> "provides two or more examples, which helps the model recognize patterns and handle more complex tasks. With more examples, the model gains a better understanding of the task, leading to improved accuracy and consistency."

#### Examples: 

**Example 1:**
```
Label the customer feedback as satisfied, unsatisfied, or mixed.

Feedback: "The service was outstanding!" 
Label: satisfied

Feedback: "Great app but crashes often." 
Label: mixed

Feedback: "Total waste of money." 
Label: unsatisfied
```

**Example 2:**
```
Classify the text tone as formal, casual, or urgent.

Text: "Dear Sir or Madam, I am writing to inquire about..."
Tone: formal

Text: "Hey! Wanna grab coffee later?"
Tone: casual

Text: "URGENT: Server down. Need immediate assistance!"
Tone: urgent
```
Further resources, 
https://medium.com/@mike_onslow/ai-simplified-exploring-the-basics-of-zero-shot-one-shot-and-few-shot-learning-d46248b5072a

https://learnprompting.org/docs/basics/few_shot



[Watch this video on YouTube](https://www.youtube.com/watch?v=YMs-BbNKs0o)





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


