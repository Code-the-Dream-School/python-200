While using jupyter notebook and a virtual environment created in folder. To let know jupyter notebook. 
step 1: pip install jupyter ipykernel
Step 2: register your venv as jupyter kernel - python -m ipykernel install --user --name=myproject-venv --display-name "Python (myproject venv)"
then go to jupter:
step1: Click the kernel name (top-right of the editor — says something like “Python 3.12 (base)”).
step2: Choose “Select Another Kernel”.
step 3: Go to “Jupyter Kernel” → you should now see:Python (myproject venv). Select it.

TO verify:
import sys, os
print(sys.executable)
output - you should see the venv folder path

Let's look at what just happened. What is this OpenAI response?

The response is its own special type of object, a ChatCompletion object: .choices, .usage, and .model are its key attributes.

Unpacking the attributes a little bit:

.choices : a list of chat responses , or completions (default length one). Each has a .message.content field that contains the model’s reply. So to get the chat response: response.choices[0].message.content, which is what we did above.
.usage : object that tells you about token usage (prompt_tokens, completion_tokens, total_tokens). This can be useful for monitoring costs.
.model : the name of the model that generated the response (e.g. gpt-3.5-turbo)

Other parameters for completions.create()¶
There are a few other important parameters for the completions API you might want to play with.

temperature: controls randomness. Lower (0 is min) is more deterministic. 0 means pick the most likely token. 0.7 is a standard default. 1.0 and great adds a great deal of randomness in selection. If you want deterministic outputs, set it to 0.
top_p: enables nucleus sampling — the model restricts the set of tokens to the smallest set whose probabilities is p, so it limits the model outputs.
n: sets number of responses returned in .choices. It defaults to 1. If you have temperature set to 0, you are wasting tokens.
max_tokens: limits the lenght of the response. This can be a useful way to keep costs under control. You can also just tell the model to keep the response under 50 words in your prompt.