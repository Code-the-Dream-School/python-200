# OpenAI Chat Completions API

In the previous lesson, we learned about the basic theory of LLMs. Now we'll see how to actually use them through OpenAI's chat completion API. These are the models that power most real world AI applications. This lesson can be used as a foundation to understand the chat completions APIs from other similar models (like Google's Gemini, Anthropic's Claude etc.). If you dig into the online docs, you will see that there is a newer *responses* API from OpenAI that is geared toward building agents, but the completions API is important to learn about as other companies mimic it, and it is sort of the gold standard. 

Here are some additional resources you can check out on chat completions: [Medium blog](https://medium.com/the-ai-archives/getting-started-with-openais-chat-completions-api-in-2024-462aae00bf0a), [Youtube video](https://www.youtube.com/watch?v=zTa97AOi61w)

> **NOTE:** To get the most out of this lesson, it is recommended that you run the examples yourself. Before beginning with the exercises, ensure that you have your OpenAI API key. If you don't have this, please reach out to your mentor. To run the exercises, you will need to create a virtual environment with the following packages installed: `OpenAI` and `dotenv`.

## What is a completion?

A *completion* refers to the output the model generates in response to your input. In the context of the *chat* API, that output is always text: the model is continuing or trying to *complete* the chat or conversation. It is useful to note that current chatbots used by Amazon, Microsoft, IBM, etc. now work with multimodal inputs (any combinations of text,  images and audio). Although there are a few models (GPT-4o for example) claimed to be *truly multimodal* (i.e. they can handle video inputs along with text, images and audio), there is still considerable work going on in this direction.

## Preliminary Steps

The following sections will guide you through the process of connecting to ChatGPT using your API key.

### Handling your API Key
There are a couple of main ways to securely handle API keys. One, you can set it as an environment variable, and the OpenAI client (instatiated with `client = OpenAI()`) will see it. A second convention is to store it in a local `.env` file in the same directory as the project (so in this case, in the same directory as this notebook). I recommend using this approach!

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
    

<!-- To check if your Python variable path is set to the right path, you can use the *executable* function.

```python
import sys
print(sys.executable)
```
```
    C:\Users\macha\OneDrive - Cal State Fullerton\Documents\Practice\Concepts\python-200\venv\Scripts\python.exe
```
-->    

### Connecting to ChatGPT
The standard way to connect to the Open AI API is to create a `client`, which is an instance of the `OpenAI` class. The `client` is an object that stores your API key and connects you to the OpenAI API server so you don't have to write raw HTTP requests yourself.


```python
from openai import OpenAI
from pprint import pprint
```

Create the client to interface with their server:


```python
client = OpenAI()
```

If you've followed the instructions in the previous section, the client should load successfully.

## "Chatting" with the API

To generate chat responses (i.e. *completions*), we use the `chat.completions.create()` method. You can find out more about it [here](https://platform.openai.com/docs/api-reference/chat/create). This has only two required arguments:

`model` and `messages`

- **`model`** :  the string name of the model you want to use (`"gpt-4"`, `"gpt-3.5-turbo"`).
- **`messages`** : a list of messages:  each message is a dictionary with `role` and `content` keys. We will explore this argument in detail in the next section.

There are lots of other optional parameters but let's not focus on those now. 

Let's look at an example.

Note that we'll use `gpt-4o-mini`, as it is really cost-efficient. Let's create a message and see it's response


```python
response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages = [{"role": "user",
                                                      "content": "Hello World"}],
                                          n=1,
                                          temperature=1.3)
print(response.choices[0].message.content)
```

Let's look at what just happened. What is this `response`?

The `response` is its own special type of object, a `ChatCompletion` object: `.choices`, `.usage`, and `.model` are its key attributes.

Unpacking the attributes a little bit:

- **`.choices`** : a list of chat responses , or completions (default length one). Each has a `.message.content` field that contains the model’s reply. So to get the chat response: `response.choices[0].message.content`, which is what we did above.
- **`.usage`** : object that tells you about token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`). This can be useful for monitoring costs.   
- **`.model`** : the name of the model that generated the response (e.g. `gpt-4o-mini`)

Feel free to play with these attributes. 


```python
type(response)
```


```python
response.model
```

### Other parameters for `completions.create()`
There are a few other important parameters for the completions API you might want to play with. You can learn more from the documentation [here](https://platform.openai.com/docs/api-reference/chat/create). Feel free to play with the optional arguments and analyse their effect on the output message. 
- `temperature`: controls randomness. Lower (0 is min) is more deterministic. `0` means pick the most likely token. 0.7 is a standard default. 1.0 and great adds a great deal of randomness in selection. If you want deterministic outputs, set it to 0. 
- `top_p`: enables *nucleus sampling* — the model restricts the set of tokens to the smallest set whose probabilities is `p`, so it limits the model outputs. Instead of using all tokens, the model samples only from the smallest group of tokens whose probabilities sum to p.
- `n`: sets number of responses returned in `.choices`. It defaults to 1. If you have temperature set to 0, you are wasting tokens.
- `max_tokens`: limits the lenght of the response. This can be a useful way to keep costs under control. You can also just *tell* the model to keep the response under 50 words in your prompt. 

Congratulations, you've successfully built your first interface with a ChatGPT model!!!

Our initial exmaple was literally a "hello world" built to demonstrate how the API works. We will take a much deeper look at how to design effective prompts in the prompt engineering lesson. This lesson focuses on giving an overview of the chat completions API. Also one feature of the API that is important to understand is that unlike web interfaces like ChatGPT, the API does not remember previous conversations - it has no memory i.e. it is *stateless*. If you want the model to recall previous messages, you need to build that memory yourself by wrapping prior messages in each request. We will go into this in more detail in the chatbot lesson.

### Check for Understanding

#### Question 1

A language model is trying to pick the next word after “I ate a”.
The possible next words and their probabilities are:

| Word | Probability |
| --- | ------------ |
| apple | 0.6 |
| banana | 0.25 |
| cherry | 0.1 |
| donut | 0.05 |

Now consider two different parameter settings:

Setting A: temperature = 0.0, top_p = 1.0

Setting B: temperature = 1.0, top_p = 0.6

Which outcome is most likely?

Choices:
- A. Both A and B will always choose apple, because it is the most/only likely option.
- B. Setting A will always choose apple, while Setting B will randomly choose between apple and banana, since together they make up the top 60% of probability.
- C. Setting A will randomly choose between all four words, while Setting B will always choose apple.
- D. Setting A and B both randomly choose among all four options.


<details>
<summary> View Answer </summary>
<strong>Answer:</strong> A. Both A and B will always choose apple. <br>
In setting A, temperature is 0.0 so top_p doesn't matter, the output will always be the word with the highest probability (i.e. apple).
In setting B, temperature is 1.0 so we're allowing for randomness in the output, top_p is 0.6 so the output is restricted to a random choice amongst the smallest set of tokens with probabilities totalling to 0.6. Here, the smallest set that satisfies this condition is {apple}, so the output is always going to be apple.
</details>

## Using built-in moderation and guardrails

The moderations endpoint is used for identifying harmful content(text/image) and to take corrective measures with the users or filter content. It is a really cool *free* service that you can incorporate in your back-end to moderate the responses from your chatbots. OpenAI has instituted a [moderation policy](https://platform.openai.com/docs/guides/moderation) and enforces it using [omni-moderation](https://platform.openai.com/docs/models/omni-moderation-latest) which supports more categorization options and multi-modal inputs.

The moderations endpoint will flag if the message falls in any of the following categories:
- hate — content that expresses hate toward a protected group
- hate/threatening — hateful content that also includes threats of violence *
- harassment — insulting, bullying, or harassing content
- harassment/threatening — harassing content that also includes threats of violence *
- self-harm — discussion of self-injury or suicide
- self-harm/intent — indicates intent to self-harm *
- self-harm/instructions — provides instructions or methods for self-harm *
- sexual — sexually explicit content
- sexual/minors — sexual content involving children/minors *
- violence — violent content, including descriptions of harming others
- violence/graphic — graphic or gory descriptions of violence *

*Asterisk* means automatic block, the openai chat completions models themselves (gpt5) will not answer, even if you the developer want it to. The others allow some flexibility and judgment on your part (note that we have seen news cases where users can be clever and figure out ways around these blocks, but the idea is that you should not provide answers if you are building a tool). 

Note you can check the moderation bits, for free, using the `moderations` endpoint which takes in `model` and `input` (text):. The model is not one of the standard models, but `omni-moderation-latest`: 


```python
mod_response1 = client.moderations.create(model="omni-moderation-latest",
                                     input="I want to kill my neighbor.")
mod_response2 = client.moderations.create(model="omni-moderation-latest",
                                     input="I want to water my neighbor's plant.")
 ```

 You can check to see if they were flagged generally with: 

     print(mod_response1.results[0].flagged, mod_response2.results[0].flagged)  # True/False

To see things in more detail about the flagged message:

    pprint(mod_response1.results[0].categories.model_dump())

In much more detail (with the scores):

    pprint(mod_response1.results[0].model_dump())

The output response has the following parameters: 
 - flagged(bool): it will help us identify if the content is harmful. true if harmful else false.
 - categories(dict): this will have sub-categories with bool values to identify the content is in which category. eg: hate, hate/threatening, self-harm violence

You should see that the first message will be flagged while the second is not.

It is important to note that the moderations endpoint is not just checking for the presence of words that can fall into the aforementioned categories, but its looking at the context of the entire message. To illustrate this, try the following example:

```python
mod_response1 = client.moderations.create(model="omni-moderation-latest",
                                     input="I want to kill my presentation.")
 ```

When you print the response, you will notice that it is not flagged. 

<!-- Let's look at an example.

```python
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
Here, we simply iterate over the example messages and pass them through the moderations endpoint to check whether they are flagged. 
```python
    INPUT: The kid got raged in school from his seniors and they also attacked him was beaten up for not following their rules
    MODERATION RESPONSE:
    flag True
    categories Categories(harassment=False, harassment_threatening=False, hate=False, hate_threatening=False, illicit=False, illicit_violent=False, self_harm=False, self_harm_instructions=False, self_harm_intent=False, sexual=False, sexual_minors=False, violence=True, violence_graphic=False, harassment/threatening=False, hate/threatening=False, illicit/violent=False, self-harm/intent=False, self-harm/instructions=False, self-harm=False, sexual/minors=False, violence/graphic=False)
    #category_applied_input_types CategoryAppliedInputTypes(harassment=['text'], harassment_threatening=['text'], hate=['text'], hate_threatening=['text'], illicit=['text'], illicit_violent=['text'], self_harm=['text'], self_harm_instructions=['text'], self_harm_intent=['text'], sexual=['text'], sexual_minors=['text'], violence=['text'], violence_graphic=['text'], harassment/threatening=['text'], hate/threatening=['text'], illicit/violent=['text'], self-harm/intent=['text'], self-harm/instructions=['text'], self-harm=['text'], sexual/minors=['text'], violence/graphic=['text'])
    ---
    INPUT: If we don’t act in 24 hours, all coastlines will be underwater.
    MODERATION RESPONSE:
    flag False
    categories Categories(harassment=False, harassment_threatening=False, hate=False, hate_threatening=False, illicit=False, illicit_violent=False, self_harm=False, self_harm_instructions=False, self_harm_intent=False, sexual=False, sexual_minors=False, violence=False, violence_graphic=False, harassment/threatening=False, hate/threatening=False, illicit/violent=False, self-harm/intent=False, self-harm/instructions=False, self-harm=False, sexual/minors=False, violence/graphic=False)
    #category_applied_input_types CategoryAppliedInputTypes(harassment=['text'], harassment_threatening=['text'], hate=['text'], hate_threatening=['text'], illicit=['text'], illicit_violent=['text'], self_harm=['text'], self_harm_instructions=['text'], self_harm_intent=['text'], sexual=['text'], sexual_minors=['text'], violence=['text'], violence_graphic=['text'], harassment/threatening=['text'], hate/threatening=['text'], illicit/violent=['text'], self-harm/intent=['text'], self-harm/instructions=['text'], self-harm=['text'], sexual/minors=['text'], violence/graphic=['text'])
    --- 
```
As expected, the first messaged is flagged as violent content and the second is not flagged at all. The `category_applied_input_types` argument becomes relevant in the case of multimodal inputs (text, images etc.) We won't go over that here, but you're free to learn more on your own! -->

## Generate audio
Interestingly, the OpenAI API is not just built for generating written text. You can generate audio completions as well!

Just for fun, try running the following, and then click on the play button in the output (you might need to install some additional libraries).

If you know a language other than English, try it out. It can speak many different languages. 

```python
from pathlib import Path
from IPython.display import Audio

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

## A Quick Introduction to Abstraction Libraries

So far, we have exclusively been using the OpenAI chat completions API. In the wider world, there are many different providers with similar APIs: OpenAI, Anthropic (Claude), Google Gemini, and others. Each one has its own format, parameters, and authentication.

Abstraction libraries exist to smooth over those differences. Instead of writing a separate application for OpenAI, Anthropic, and Gemini, you can work with a single package that lets you interface with any one of these AI libraries. That library then knows how to call each provider behind the scenes. The key benefit is that you get one unified interface to many model providers. You can switch providers (for cost, quality, or availability reasons) rather than rewriting all your application code. This also helps prevent vendor lock-in: you are not tied to a single API forever.

Examples of these abstraction tools include [langchain](https://www.langchain.com/langchain), [litellm](https://www.litellm.ai/), [any-llm](https://github.com/mozilla-ai/any-llm). All of them let you write code once and route requests to different LLM backends. Many of them can also talk to local models, such as models served by Ollama, which is an open-source package that lets you run models locally.

### Running a local model with Ollama

[Ollama](https://ollama.com/) is a local model manager and server. It lets you download language models to your own machine and run them without paying per-token API fees. Ollama also has an [API](https://github.com/ollama/ollama-python) that can be used to incorporate your local model into your workflow. While many abstraction libraries can talk to Ollama as just another backend, you can also use it directly from the command line. Let's check it out!

First, follow instructions in [website](https://ollama.com/download) to install Ollama. Then, open a terminal and check that the CLI is available:
```bash
ollama --version
```
Next, we need to download a model. Check out the list of available models [here](https://ollama.com/search). We will use a small model called qwen3:0.6b. It is not the smartest model in the world, but it is small enough to run on most laptops, even without an NVIDIA GPU. This only needs to happen once per model:
```bash
ollama pull qwen3:0.6b
```
The `pull` command downloads a full runnable model artifact to your local machine. <!-- So you may end up with the error: `Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted`. -->

Sometimes the downloads can be slow (the model is about 500MB). After the pull finishes, you can chat with the model locally:
```bash
ollama run qwen3:0.6b
```
This opens an interactive prompt in your terminal. Type a message and press Enter, just like in a chat window. For example:
```
Explain how to define a function in python
```
Ollama will stream back a response. When you are done, type `/bye` to finish the chat.

You can see which models are installed locally with:
```bash
ollama list
```
While a model is running, you can open a second terminal and check whether it is using CPU or GPU:
```bash
ollama ps
```
Look at the PROCESSOR column in the output. On machines with a supported GPU, you may see GPU. If not you will see CPU.

<!--
LLM Arena - https://lmarena.ai/leaderboard
Mozilla AI's [Any-Agent](https://github.com/mozilla-ai/any-agent).You can read more about the abstraction framework and its usage [here](https://blog.mozilla.ai/introducing-any-agent-an-abstraction-layer-between-your-code-and-the-many-agentic-frameworks/). -->

## Explore on your own!
Congratulations! You've learned the basics of the completions API! There really isn't that much more to it. From here you could build a simple chatbot with a personality, build in memory of previous conversations, and build a simple application. 

I'd encourage you to explore more below. See what it can do. Explore in different languages to see where it suceeds, fails, etc. Feel free to expand the list of messages with assistant/user interactions. You will be given the opportunity for a more hands-on experience in the assignments.

```python
messages = [{"role": "system",
             "content": "<add your own> create your own personality here"},
            {"role": "user",
             "content": "<add your own> create your own content here"}]

response = client.chat.completions.create(model='gpt-4o-mini',
                                          messages=messages)
print(response.choices[0].message.content)
```

<!-- ## More to learn for the curious

### Providing more context to the API

So far, we've looked at simple text inputs to the API. Let's look at providing additional context and information to the Completions API to enable it to provide more informed outputs.

Recall that a message is a dictionary with `role` and `content` keys. The `role` describes where the content is coming from: the `user` or the model (`assistant`). The `content` is the text being sent. There is a third special role, the `system`, which is typically sent first, that sets up the model's personality (e.g., tell it to act like a kindly grandmother talking to a bunch of young children).

In practice, once the `system` role is set, the roles tend to switch between `user` (query from person) and `assistant` (answer from the model). 

Let's look at an example where we want to give kudos to an employee who did a great job at their current project. First, we will give information about the employee to the model and then ask for suggestions on what to say about his/her work.

```python
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! We have an employee Alex who recently joined our team. She worked on research and development on the newest feature of analyzing data through pdf files. She delivered the product and has created 12% impact. She got fast in adapting the MCP and delivered the product on time."}
  ],
    store = True
)

print(completion.choices[0].message)
```
Here, we provide an additional message in the system role, instructing the API to act as a helpful assistant. The responses from the API will now be as an assistant. Note that you can provide multiple messages as context for the API completion. As you will learn in a later lesson, chatbots provide all previous messages in a conversation as context to generate appropriate responses. Interestingly, in the later versions of the API you can add names to different messages with the same role, so you can provide a complete conversation as context as well. Check out [this](https://community.openai.com/t/role-management-in-the-chat-completions-api/929112) forum post.

The store parameter is a boolean value that helps us store the chat completion output so that we can use it later as required. We will later used this stored  completion to continue the conversation. The stored completions can also be viewed on your OpenAI platform dashboard.

Let's see what the response look like.

```python
    ChatCompletionMessage(content="Hello! It sounds like Alex is a valuable addition to your team. It's great to hear that she has been successful in both the research and development aspects of the project, especially in creating a feature for analyzing data through PDF files. Delivering a product with a 12% impact is impressive, and her ability to adapt quickly to new tools like MCP is commendable. It seems like Alex is proactive and efficient in her work, which is beneficial for the team's productivity and success. If you need any more assistance or advice regarding Alex or any other team member, feel free to ask!", refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None) 
```
As you can see, the response looks like an assistant assessing and welcome the new team member Alex. Let's use the stored completion to continue the chat. 

```python
completions = client.chat.completions.list()
if completions.first_id: 
    print("First_ID:",completions.first_id)
    first_id = completions.first_id
else:
    print("No data")
if first_id: 
    first_completion = client.chat.completions.retrieve(completion_id=first_id)
    print("Data:", first_completion.choices[0].message.content)
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
print("Kudos:",kudos_message.choices[0].message.content)
```
Once you've stored the previous API response, it will be available using the `chat.completions.list()` function. You can learn more about it [here](https://platform.openai.com/docs/api-reference/chat/object).

Let's look at the outputs.
```python
    First_ID: chatcmpl-CPnj8rO3x2TNXfihsXXaVlHwZemcf
    Data: Alex, your commitment to excellence and passion for innovation really came through on the new feature. You took it from concept to a polished release—grounded in thoughtful research, careful testing, and crisp cross-team collaboration—and the results show it. Your work is advancing our roadmap, elevating the user experience, and energizing the team. Congratulations on a job brilliantly done, and thank you for the dedication you bring every day. Keep up the fantastic work!
    Kudos: Alex, your dedication and commitment to excellence truly shone through in your work on the new feature. Your innovative approach, attention to detail, and collaborative spirit were instrumental in bringing this project to life. Your tireless efforts have not only enhanced our roadmap but have also elevated the overall user experience and inspired the entire team. Your passion and hard work do not go unnoticed, and we truly appreciate the value you bring each day. Congratulations on a job well done, Alex. Your contributions make a real difference, and we are grateful for your outstanding work. Keep up the fantastic work!
```
As you can see, the previously provided context of the system message and its first response has been used to provide the Kudos message. 

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
response = client.chat.completions.create(model='gpt-4o-mini', 
                                          messages=messages)
print(response.choices[0].message.content)
```

We will look at how chatbots append the previous messages continuously into the current input to add context in a subsequent module. -->