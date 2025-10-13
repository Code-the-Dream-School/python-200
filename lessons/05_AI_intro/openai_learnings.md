# Hello OpenAI

This is a standalone intro to the OpenAI API to help get people started working with OpenAI's ChatGPT models. 

You will need an OpenAI API key. Running the code in this notebook will cost less than a penny. 

To invoke the completions API, we use `client.chat.completions.create(...)`. We'll step through this below, but for now note that a  *completion* refers to the output the model generates in response to your input. In the context of the *chat* API, that output is always text: the model is continuing or trying to *complete* the chat or conversation. 

If you dig into the online docs, you will see that there is a newer *responses* API from OpenAI that is geared toward building agents, but the completions API is important to learn about as other companies mimic it, and it is sort of the gold standard. 

To run this, you will need to create a virtual environment with the following packages installed: `OpenAI`, `dotenv`, and `jupyterlab`. Also, as discussed in the next section, I recommend creating a `.env` file with your openai api key (this is discussed in the next section). Be sure to put `.env` in `.gitignore`!

## Handling your API Key
There are a couple of main ways to securely handle API keys. One, you can set it as an environment variable, and the OpenAI client (instatiated with `client = OpenAI()`) will see it. A second convention is to store it in a local `.env` file in the same directory as the project (so in this case, in the same directory as this notebook).

The content of the `.env` file would be:

    OPENAI_API_KEY=<api key here>

> **Security consideration**: When using .env to handle api keys, it is essential that `.env` be in `.gitignore`, otherwise you will end up publicly sharing your private api keys. 

Once it is stored in `.env` there is a function you can import that will extract the contents of this secret dot env file called...`load_dotenv`, as follows:


```python
from dotenv import load_dotenv
if load_dotenv():
    print("Successfully loaded api key")
```

    Successfully loaded api key
    


```python
#To check if your Python variable path is set to the right path
import sys
print(sys.executable)
```

    C:\Users\macha\OneDrive - Cal State Fullerton\Documents\Practice\Concepts\python-200\venv\Scripts\python.exe
    

## Connecting to ChatGPT
The standard way to connect to the Open AI API is to create a `client`, which is an instance of the `OpenAI` class. The `client` is an object that stores your API key and connects you to the OpenAI API server so you don't have to write raw HTTP requests yourself.


```python
from openai import OpenAI
from pprint import pprint
```

Create the client to interface with their server:


```python
client = OpenAI()
```

To generate chat responses, we use the `chat.completions.create()` method. This has only two required arguments:

`model` and `messages`

- **`model`** :  the string name of the model you want to use (`"gpt-4"`, `"gpt-3.5-turbo"`).
- **`messages`** : a list of messages:  each message is a dictionary with `role` and `content` keys. 

There are lots of other optional parameters but let's not focus on those now. 

Let's look at an example.

Note that we'll use `gpt-3.5-turbo`, as it is really cost-efficient. 


```python
response = client.chat.completions.create(model="gpt-3.5-turbo",
                                          messages = [{"role": "user",
                                                      "content": "Hello World"}],
                                          n=1,
                                          temperature=1.3)
```

Let's see it's response


```python
print(response.choices[0].message.content)
```

Congratulations, you've successfully build your first interface with a ChatGPT model!!!

Let's look at what just happened. What is this `response`?

The `response` is its own special type of object, a `ChatCompletion` object: `.choices`, `.usage`, and `.model` are its key attributes.

Unpacking the attributes a little bit:

- **`.choices`** : a list of chat responses , or completions (default length one). Each has a `.message.content` field that contains the model’s reply. So to get the chat response: `response.choices[0].message.content`, which is what we did above.
- **`.usage`** : object that tells you about token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`). This can be useful for monitoring costs.   
- **`.model`** : the name of the model that generated the response (e.g. `gpt-3.5-turbo`)

Feel free to play with these attributes.


```python
type(response)
```


```python
response.model
```

### Other parameters for `completions.create()`
There are a few other important parameters for the completions API you might want to play with.
- `temperature`: controls randomness. Lower (0 is min) is more deterministic. `0` means pick the most likely token. 0.7 is a standard default. 1.0 and great adds a great deal of randomness in selection. If you want deterministic outputs, set it to 0. 
- `top_p`: enables *nucleus sampling* — the model restricts the set of tokens to the smallest set whose probabilities is `p`, so it limits the model outputs. Instead of using all tokens, the model samples only from the smallest group of tokens whose probabilities sum to p.
- `n`: sets number of responses returned in `.choices`. It defaults to 1. If you have temperature set to 0, you are wasting tokens.
- `max_tokens`: limits the lenght of the response. This can be a useful way to keep costs under control. You can also just *tell* the model to keep the response under 50 words in your prompt. 



```python
Quick Question? 

A language model is trying to pick the next word after “I ate a”.
The possible next words and their probabilities are:

Word	Probability
apple	0.6
banana	0.25
cherry	0.1
donut	0.05

Now consider two different parameter settings:

Setting A: temperature = 0.0, top_p = 1.0

Setting B: temperature = 1.0, top_p = 0.6

Which outcome is most likely?

A.
Both A and B will always choose apple, because it has the highest probability.

B.
Setting A will always choose apple, while Setting B will randomly choose between apple and banana, since together they make up the top 60% of probability.

C.
Setting A will randomly choose between all four words, while Setting B will always choose apple.

D.
Setting A and B both randomly choose among all four options.
```


```python
#Let's play around more with the completions API. 
# We want to give kudos to an employee who did a great job at their current project. 
#First, we will give information about the employee to the model and then ask for suggestions on what to say about his/her work.
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! We have an employee Alex who recently joined our team. She worked on reaserch and also development part on the newest feature of analyzing data through pdf files. She develiedred product and having created 12% impact. She got fast in adapting the MCP and delievered the product on time."}
  ],
    store = True
)

#store parameter(bool value) - helps us store the chat completion output so that we can use it later as required.

print(completion.choices[0].message)
```

    ChatCompletionMessage(content="Hello! It sounds like Alex is a valuable addition to your team. It's great to hear that she has been successful in both the research and development aspects of the project, especially in creating a feature for analyzing data through PDF files. Delivering a product with a 12% impact is impressive, and her ability to adapt quickly to new tools like MCP is commendable. It seems like Alex is proactive and efficient in her work, which is beneficial for the team's productivity and success. If you need any more assistance or advice regarding Alex or any other team member, feel free to ask!", refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None)
    


```python
completions = client.chat.completions.list()
#print(completions)
if completions.first_id: 
    print("First_ID",completions.first_id)
    first_id = completions.first_id
else:
    print("No data")
if first_id: 
    first_completion = client.chat.completions.retrieve(completion_id=first_id)
    print("Data", first_completion.choices[0].message.content)
    employee_info = first_completion.choices[0].message.content
kudos_message = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": employee_info },
      {"role": "user", "content": "You have to give kudos to the employee for their hard work and align it to their work to feel personal." }
  ],
    store = True
)
print("Kudos",kudos_message.choices[0].message.content)

    
```

    First_ID chatcmpl-CPnj8rO3x2TNXfihsXXaVlHwZemcf
    Data Alex, your commitment to excellence and passion for innovation really came through on the new feature. You took it from concept to a polished release—grounded in thoughtful research, careful testing, and crisp cross-team collaboration—and the results show it. Your work is advancing our roadmap, elevating the user experience, and energizing the team. Congratulations on a job brilliantly done, and thank you for the dedication you bring every day. Keep up the fantastic work!
    Kudos Alex, your dedication and commitment to excellence truly shone through in your work on the new feature. Your innovative approach, attention to detail, and collaborative spirit were instrumental in bringing this project to life. Your tireless efforts have not only enhanced our roadmap but have also elevated the overall user experience and inspired the entire team. Your passion and hard work do not go unnoticed, and we truly appreciate the value you bring each day. Congratulations on a job well done, Alex. Your contributions make a real difference, and we are grateful for your outstanding work. Keep up the fantastic work!
    

# Generate audio


```python
from pathlib import Path
from IPython.display import Audio
```

The OpenAI API is not just built for generating written text. 

Just for fun, try running the following, and then click on the play button below (you might need to install some additional libraries).

If you know a language other than English, try it out. It can speak many different languages. 


```python
voice = "alloy"  #shimmer is higher pitch, onyx lower pitch

speech_file_path = Path("speech.mp3")

input_text1 = "Python is an amazing programming language."
input_text2 = "But it can't take my dog for a walk.... Or feed my fish."

input_text = input_text1 + input_text2

try:
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=input_text
    ) as speech_response:
        speech_response.stream_to_file(speech_file_path)
except openai.BadRequestError:
    print("Invalid voice selected: {voice}")

print(f"Audio saved as {speech_file_path} with voice of {voice}")

Audio(str(speech_file_path))
```

# Moderations

Moderations endpoint is used for identifying harmful content(text/image) and to take corrective measures with the users or filter content. 
Model  -  omni-moderation-latest: This model supports more categorization options and multi-modal inputs.

params of output response: 
flagged(bool): it will help us identify if the content is harmful. true if harmful else false.
categories(dict): this will have sub-categories with bool values to identify the content is in which category. eg: hate, hate/threatening, self-harm violence
 


```python
# Example: moderation check for several test strings

examples = [
    "The kid got raged in school from his seniors and they also attacked him was beaten up for not following their rules",
    "If we don’t act in 24 hours, all coastlines will be underwater.",
]

for text in examples:
    resp = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    print("INPUT:", text)
    print("MODERATION RESPONSE:")
    print("flag", resp.results[0].flagged)
    print("categories", resp.results[0].categories)
    #print("category_applied_input_types", resp.results[0].category_applied_input_types)
    print("---")

```

    INPUT: The kid got raged in school from his seniors and they also attacked him was beaten up for not following their rules
    MODERATION RESPONSE:
    flag True
    categories Categories(harassment=False, harassment_threatening=False, hate=False, hate_threatening=False, illicit=False, illicit_violent=False, self_harm=False, self_harm_instructions=False, self_harm_intent=False, sexual=False, sexual_minors=False, violence=True, violence_graphic=False, harassment/threatening=False, hate/threatening=False, illicit/violent=False, self-harm/intent=False, self-harm/instructions=False, self-harm=False, sexual/minors=False, violence/graphic=False)
    category_applied_input_types CategoryAppliedInputTypes(harassment=['text'], harassment_threatening=['text'], hate=['text'], hate_threatening=['text'], illicit=['text'], illicit_violent=['text'], self_harm=['text'], self_harm_instructions=['text'], self_harm_intent=['text'], sexual=['text'], sexual_minors=['text'], violence=['text'], violence_graphic=['text'], harassment/threatening=['text'], hate/threatening=['text'], illicit/violent=['text'], self-harm/intent=['text'], self-harm/instructions=['text'], self-harm=['text'], sexual/minors=['text'], violence/graphic=['text'])
    ---
    INPUT: If we don’t act in 24 hours, all coastlines will be underwater.
    MODERATION RESPONSE:
    flag False
    categories Categories(harassment=False, harassment_threatening=False, hate=False, hate_threatening=False, illicit=False, illicit_violent=False, self_harm=False, self_harm_instructions=False, self_harm_intent=False, sexual=False, sexual_minors=False, violence=False, violence_graphic=False, harassment/threatening=False, hate/threatening=False, illicit/violent=False, self-harm/intent=False, self-harm/instructions=False, self-harm=False, sexual/minors=False, violence/graphic=False)
    category_applied_input_types CategoryAppliedInputTypes(harassment=['text'], harassment_threatening=['text'], hate=['text'], hate_threatening=['text'], illicit=['text'], illicit_violent=['text'], self_harm=['text'], self_harm_instructions=['text'], self_harm_intent=['text'], sexual=['text'], sexual_minors=['text'], violence=['text'], violence_graphic=['text'], harassment/threatening=['text'], hate/threatening=['text'], illicit/violent=['text'], self-harm/intent=['text'], self-harm/instructions=['text'], self-harm=['text'], sexual/minors=['text'], violence/graphic=['text'])
    ---
    

## A little more about messages
We said above that the messages parameter the chat completions endpoint is a *list of messages* is a dictionary with `role` and `content` keys. Before finishing this up, let's look at this in a little more detail. The `role` describes where the content is coming from: the `user` or the model (`assistant`). The `content` is the text being sent. There is a third special role, the `system`, which is typically sent first, that sets up the model's personality (e.g., tell it to act like a kindly grandmother talking to a bunch of young children). 

In practice, once the `system` role is set, the roles tend to switch between `user` (query from person) and `assistant` (answer from the model). 

Sending a list of messages is important because the chat completions endpoint *has no memory of previous conversations*, so if you want it to have context, *you have to send it*.

Here is an example:


```python
messages = [{"role": "system",
             "content": "You are a helpful teacher. You explain things at a level that a beginning Python programmer can understand."},
            {"role": "user",
             "content": "Are there any other measures of complexity besides time complexity for an algorithm?"},
            {"role": "assistant",
             "content": "Yes there are other measures, such as space complexity."},
            {"role": "user",
             "content": "What is that?"}]
```


```python
current_model = "gpt-3.5-turbo"
```


```python
response = client.chat.completions.create(model=current_model,
                                          messages=messages)
```


```python
print(response.choices[0].message.content)
```

## Explore on your own!
Congrats you know the basics of the completions API! There really isn't that much more to it. From here you could build a simple chatbot with a personality, build in memory of previous conversations, and build a simple application. 

I'd encourage you to explore more below. See what it can do. Explore in different languages to see where it suceeds, fails, etc. 


```python
messages = [{"role": "system",
             "content": "<add your own> create your own personality here"},
            {"role": "user",
             "content": "<add your own> create your own content here"}]

# Feel free to expand the list of messages with assistant/user interactions
```


```python
response = client.chat.completions.create(model=current_model,
                                          messages=messages)
```


```python
print(response.choices[0].message.content)
```


```python

```
