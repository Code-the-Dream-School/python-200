# Azure OpenAI: A Note for Production

*Short lesson -- closer to a reading than a hands-on exercise. Shows the code change and explains the why, without requiring students to connect to Azure OpenAI.*

Sections:

  * What Azure OpenAI is: the same models hosted under your organization's Azure subscription
  * Why organizations use it: data doesn't leave your tenant, compliance requirements, enterprise SLA, no data used for OpenAI training
What changes in the code: two lines -- the import and the client initialization
What stays identical: every call after that (chat.completions.create, prompt structure, response parsing)
How to make the switch when you encounter it in a job context: find your endpoint and deployment name in Azure AI Foundry, swap the client

**Before/After**

```python
# Using OpenAI directly (what we built in Lesson 2)
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...]
)

# Using Azure OpenAI (what you'll see in most enterprise environments)
from openai import AzureOpenAI
client = AzureOpenAI(
    azure_endpoint="https://<resource>.openai.azure.com",
    api_key="<azure-api-key>",       # or use DefaultAzureCredential
    api_version="2024-02-01"
)
response = client.chat.completions.create(
    model="my-gpt4o-deployment",     # deployment name, not model name
    messages=[...]                   # identical from here on
)
```

**No hands-on exercise in this lesson**. Students read it, understand the pattern, and move on.

**External resources:** Azure OpenAI vs. OpenAI API (Microsoft doc); Azure AI Foundry portal overview
