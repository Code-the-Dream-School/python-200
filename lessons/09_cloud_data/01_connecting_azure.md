# Connecting to Azure from Python

*Covers how Python code authenticates to Azure. Bridges the az login from Week 8 into programmatic access.*

Sections:

* Why authentication works differently in the cloud (not just a password -- identity and credentials)
* The azure-identity package and DefaultAzureCredential
* How DefaultAzureCredential uses the az login session students already have
* Quick check: a 5-line script that prints the current subscription name
* Brief note: what changes in production (managed identity, service principals) -- name them but don't go deep

Representative code sketch:

```python
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import SubscriptionClient

credential = DefaultAzureCredential()
client = SubscriptionClient(credential)
for sub in client.subscriptions.list():
    print(sub.display_name)
```

**External resources:** Microsoft: DefaultAzureCredential explained; Azure Identity client library overview
