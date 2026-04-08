# Connecting to Azure from Python

In Week 8, you logged into the Azure Portal and ran commands in Cloud Shell. Everything you did required a human in the loop -- you clicked a button, typed a password, or ran a command yourself. This week, you are going to write Python scripts that interact with Azure directly, without any human involved.

That shift creates an immediate problem: how does a script prove to Azure that it has permission to do something? You cannot type a password into an automated process, and hardcoding credentials in source code is a serious security risk. Cloud platforms solve this with a different model entirely -- one based on *identity* rather than passwords.

> TODO: add short video on Azure authentication and the identity model (Azure Entra ID overview, ~5-10 min)

For background reading:
- [Azure Identity client library for Python](https://learn.microsoft.com/en-us/python/api/overview/azure/identity-readme)
- [Authenticate Python apps to Azure services](https://learn.microsoft.com/en-us/azure/developer/python/sdk/authentication/overview)

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain how cloud authentication differs from password-based login
- Install and import the azure-identity package
- Use DefaultAzureCredential to authenticate a Python script using your existing az login session
- Describe how authentication changes in a deployed (production) environment

## Authentication in the Cloud

When you log into a website, the site checks your username and password. Simple enough. But a Python script running in a data pipeline does not have a user sitting at a keyboard -- it needs to prove its identity automatically, often without any secrets baked into the code at all.

Cloud platforms handle this through an *identity-based* model. Instead of a password, every resource and user in Azure is assigned an *identity* -- a unique principal registered in a directory called Microsoft Entra ID (formerly Azure Active Directory). When your code wants to access an Azure resource, it presents credentials that prove it is a recognized identity, and Azure checks whether that identity has the required permissions.

There are a few types of principals you will encounter:

- *User accounts* are the identities tied to human users -- the one you use to log into the portal.
- *Service principals* are application identities: non-human accounts created for scripts and services.
- *Managed identities* are a special type of service principal that Azure manages for you, attached automatically to a compute resource like a VM or container.

You do not need to memorize these distinctions right now. The key insight is that scripts are treated as *principals with permissions*, just like users -- they just use different mechanisms to prove who they are.

## DefaultAzureCredential

Writing code that handles every credential type separately would be tedious and fragile. The azure-identity library provides a single object that solves this elegantly: `DefaultAzureCredential`.

When you create a `DefaultAzureCredential` instance, it tries a sequence of authentication methods in order, stopping at the first one that works. The sequence includes environment variables, managed identity (for VMs and containers running in Azure), your `az login` session, and a few other developer tools.

For local development -- which is what you are doing right now -- the `az login` session from Week 8 is what matters. Because you already ran `az login`, `DefaultAzureCredential` can pick up that session automatically. You do not need to write any credential-handling code yourself.

## Installing the Packages

Install the azure-identity and azure-mgmt-resource packages:

```bash
uv pip install azure-identity azure-mgmt-resource
```

## Verifying Your Connection

The quickest way to confirm that your credentials are working is to list the subscriptions your account has access to. With `az login` in place, the following script should run without errors:

```python
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import SubscriptionClient

credential = DefaultAzureCredential()
client = SubscriptionClient(credential)

for sub in client.subscriptions.list():
    print(sub.display_name)
```

Run this script locally. You should see the Code the Dream subscription name printed to the terminal. If you get an authentication error, make sure you have run `az login` recently -- sessions do expire after a period of inactivity.

## A Note on Production Authentication

In local development, `DefaultAzureCredential` relies on your `az login` session. But when you deploy a pipeline to run on an Azure VM or in a container, there is no human around to run `az login`. What then?

This is where managed identities come in. When a VM or container has a managed identity assigned to it, `DefaultAzureCredential` detects and uses it automatically -- no extra configuration required. The same Python code that works locally with `az login` will work in the cloud with managed identity. You do not need to change anything.

This is the main reason `DefaultAzureCredential` is preferred over hardcoding credentials or managing environment variables yourself: it works correctly in both development and production without any code changes. You will see this pay off in Week 11 when you deploy the capstone pipeline.
