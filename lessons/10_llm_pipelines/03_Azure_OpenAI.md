# Azure OpenAI: A Note for Production

The previous lesson used the OpenAI API directly -- the same API you have been using since Week 5. In most learning environments and personal projects, that is the right choice. But if you take a data engineering role at a company running on Azure, you will almost certainly encounter something different: Azure OpenAI.

This is a short reading lesson. There is no hands-on exercise -- you just need to understand the pattern so it is not a surprise when you see it on the job.

For reference:
- [Azure OpenAI Service overview](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview)
- [Azure AI Foundry portal](https://ai.azure.com/)

## What Azure OpenAI Is

Azure OpenAI is not a different model or a different API. It is the same GPT-4o, GPT-4o-mini, and other OpenAI models, hosted inside Microsoft's Azure infrastructure under your organization's subscription.

From the perspective of the Python code you write, the difference is almost invisible: you use the same `openai` Python package, the same `chat.completions.create()` call, the same message structure, and the same response parsing. Two lines change. Everything else stays the same.

## Why Organizations Use It

The reason most enterprises use Azure OpenAI instead of calling OpenAI directly comes down to three things.

*Data residency and compliance.* When you call the OpenAI API, your data leaves your organization's infrastructure and travels to OpenAI's servers. For companies in regulated industries -- healthcare, finance, government -- or companies with strict data governance policies, that is often not acceptable. With Azure OpenAI, requests stay inside Azure's infrastructure, subject to the same compliance controls as the rest of your cloud environment.

*No training data use.* OpenAI's enterprise terms already prohibit using API data for training, but Azure OpenAI makes this contractually explicit under Microsoft's enterprise agreements, which some organizations require.

*Unified billing and support.* Azure OpenAI costs appear on the same bill as the rest of your Azure infrastructure. Support goes through Microsoft rather than a separate vendor relationship. For large organizations, this simplifies procurement and reduces the number of vendor relationships to manage.

## What Changes in the Code

Two lines. The import changes from `OpenAI` to `AzureOpenAI`, and the client initialization takes three Azure-specific parameters instead of an API key:

```python
# Using OpenAI directly (Lesson 2 approach)
from openai import OpenAI

client = OpenAI(api_key="<openai-api-key>")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

```python
# Using Azure OpenAI (enterprise / production approach)
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://<resource-name>.openai.azure.com",
    api_key="<azure-api-key>",
    api_version="2024-02-01"
)

response = client.chat.completions.create(
    model="my-gpt4o-deployment",   # deployment name, not model name
    messages=[{"role": "user", "content": "Hello"}]
)
```

Everything from `chat.completions.create()` onward is identical. The same prompt, the same response parsing, the same error handling all carry over without changes.

The one non-obvious difference: `model` takes a *deployment name*, not a model name. In Azure OpenAI, you do not call a model directly -- you call a named deployment that your organization's admin created and configured. The deployment name is chosen by whoever set up the resource and might be something like `"gpt4o-mini-prod"` or `"my-gpt4o-deployment"` rather than `"gpt-4o-mini"`.

## Finding Your Endpoint and Deployment Name

When you start a job and need to connect to Azure OpenAI, you will need two pieces of information: the endpoint URL and the deployment name. Both are found in [Azure AI Foundry](https://ai.azure.com/).

Navigate to your Azure OpenAI resource in AI Foundry, then open the *Deployments* section. You will see a list of deployed models -- the deployment name is in the first column and the endpoint URL is visible in the resource overview. Your platform or infrastructure team may also provide these directly.

That is all there is to it. When you encounter Azure OpenAI on the job, the swap is straightforward -- and now you know why it is there.
