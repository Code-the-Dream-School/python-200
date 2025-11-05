# From completions to chatbots
When you first learn to use the Chat Completions API, it feels like you already have everything you need to build a chatbot. You send a prompt, the model replies - done, right? But if you try to carry on a conversation, you quickly realize something’s missing.

As we've seen previously, every API call is _stateless_: the model only sees what you send currently. If you send only your most recent message, it forgets everything that came before. To have a flowing conversation, you have to keep reminding it of the whole dialogue so far. 

To remedy this, and create a flowing conversation that keeps track of context, you can instead create a chatbot, which has memory built in so the model can respond in context. This can be as simple as a list of previous messages accumulating inside of a while loop. 

Here's some additional resources on chatbots: [Youtube video](https://www.youtube.com/watch?v=1P5Yccy1rRk), [IBM blogpost on chatbots](https://www.ibm.com/think/topics/chatbots). For those interested, [here](https://www.sciencedirect.com/science/article/pii/S2666827020300062) is a paper explaining the history, classification, and applications of chatbots.

<!-- Paper on the history, technology, and applications of chatbots - [Adamopoulou, Eleni, and Lefteris Moussiades. "Chatbots: History, technology, and applications." Machine Learning with applications 2 (2020): 100006.](https://www.sciencedirect.com/science/article/pii/S2666827020300062) -->

We will illustrate this with a simple chatbot example. As before, it is highly recommended that you run this example yourself to gain the most from this lesson. 

## A minimal chatbot 
Let's demonstrate this idea by building a simple `SalesBot`. It's a friendly, competent shoe-store assistant that asks questions to learn about the customer's shoe size, preferences, and needs.

We'll start by specifying a `system` message that defines the Salesbot's personality. Then, we'll enter a while loop in which the user and the assistant take turns exchanging messages. Each time the user types something, we'll add their message to a list called messages (with the `user` role), and the model's responses will be appended with the `assistant` role. 

To ensure that we don't go overboard with token usage, we will limit completion tokens to 300 to make our model succinct and save money, and the back and forth to 10 turns just as a guardrail:

The loop keeps going until the user types `quit` or `exit` (or until one of the limits is hit).

```python
from openai import OpenAI

client = OpenAI()

# 1. Start with the system message that defines the bot's personality
messages = [
    {"role": "system", 
    "content": "You are a friendly, competent sales assistant who helps customers find shoes that fit their needs. Ask questions to learn their shoe size, style preferences, and use case before recommending options."}
]

print("SalesBot: Hi there! I'm here to help you find the perfect pair of shoes.")
print("Please let me know what kind of shoes you're looking for -- running, casual, hiking, etc.")
print("(Type 'quit' anytime to end the chat.)")

# 2. Begin the conversation loop
exchange_count = 0
max_exchanges = 10

while exchange_count < max_exchanges:
    user_input = input("You: ")

    if user_input.lower() in {"quit", "exit"}:
        print("SalesBot: Thanks for visiting! Have a great day.")
        break

    # Add the user's message to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Send the full conversation so far to the API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_completion_tokens=300
    )

    # Extract and print the assistant's reply
    reply = response.choices[0].message.content
    print(f"SalesBot: {reply}")
    print("(Type 'quit' anytime to end the chat.)\n")

    # Add the assistant's reply to the conversation history
    messages.append({"role": "assistant", "content": reply})

    exchange_count += 1

if exchange_count >= max_exchanges:
    print("SalesBot: It was great chatting with you! Let's continue another time.")
```
Congratulations — you've just built your first AI chatbot!

Each loop through this code expands the messages list, allowing the chatbot to stay aware of the conversation so far. The `max_completion_tokens` setting limits how long the model's responses can be, and the `max_exchanges` counter keeps the chat from running forever.

In a real web-based chatbot, such core logic would be wrapped in quite a bit more structure. For instance, a production chatbot would include guardrails to prevent prompt injection (so users can't trick the bot into revealing system instructions or do something malicious). It would also include a more fancy user interface. However, those features wouldn't change the core chatbot architecture. 

### Check for Understanding
```
You are an employee in the HR department of GreatAI, an AI service company's. Alex is an employee in the company's operations department. You would like to use the OpenaI API to evaluate her performance for appraisal purposes. The company's evaluation policy for appriasal states that only percentage of projects delivered, number of projects participated in, and value improvement to the company are considered. The following are two approaches to pass Alex's work highlights to the API before asking it to evaluate her performance. Which of the two approaches would you expect to give you the most accurate evaluation based on the company's policy?
```

```python
# Approach 1:
messages = [{"role": "system", "content": "You are a helpful HR assistant for an AI service company, proficient in evaluating exmployees' performance for appraisal. You will consider percentage of projects delivered, number of projects participated in, and value improvement provided to the company in your calculations."},
    {"role": "user", "content": "Hello! We have an employee Alex who recently joined our team. She worked on research and development on the newest feature of analyzing data through pdf files. She delivered the product and has created 12% impact. She was quick in adapting the MCP and delivered the product on time."},
    {"role": "assistant", "content": "Hello! It sounds like Alex is a valuable addition to your team. It's great to hear that she has been successful in both the research and development aspects of the project, especially in creating a feature for analyzing data through PDF files. Delivering a product with a 12% impact is impressive, and her ability to adapt quickly to new tools like MCP is commendable. It seems like Alex is proactive and efficient in her work, which is beneficial for the team's productivity and success. If you need any more assistance or advice regarding Alex or any other team member, feel free to ask!"}.
    {"role":"user", "content":"Evaluate Alex's performance for appraisal."}]
response = client.chat.completions.create(model='gpt-4o-mini', 
                                          messages=messages)

# Approach 2:
personality_message = [{"role": "user", "content": "You are a helpful HR assistant for an AI company, proficient in evaluating exmployees' performance for appraisal. You will consider percentage of projects delivered, number of projects participated in, and value improvement provided to the company in your calculations."}]
personality_response = client.chat.completions.create(model='gpt-4o-mini', 
                                          messages=personality_message)

evaluation_message = [{"role": "user",
             "content": "Hello! We have an employee Alex who recently joined our team. She worked on research and development on the newest feature of analyzing data through pdf files. She delivered the product and has created 12% impact. She was quick in adapting the MCP and delivered the product on time. Evaluate Alex's performance for appraisal."}]
evaluation_response = client.chat.completions.create(model='gpt-4o-mini', 
                                          messages=evaluation_message)
print(evaluation_response.choices[0].message.content)
```

```
A: Approach 1, because the API has context about the company's appraisal evalaution policy
B: Approach 2, because the evaluation message contains the information about the employee along with the request to evalaute.
C: Approach 1, because the API can draw from the appraisal policy's of different AI service companies.
```

<details>
<summary> View Answer </summary>
<strong>Answer:</strong> A. Approach 1, because the API has context about the company's appraisal evalaution policy <br>
The system message serves two purposes. It sets the API's personality and provides the company's policy regarding performance evaluations for appraisal. Approach 2 will give a response that will resemble a professional appraisal evaluation but it will not be according to the company's policy as the API does not remember previous user message assigning the personality and providing the company's policy guidelines. 
</details>

