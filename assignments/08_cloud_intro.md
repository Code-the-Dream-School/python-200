# Week 8 Assignments

This week's assignments cover the cloud computing concepts from the two Week 8 lessons:

- Core cloud concepts: what cloud computing is, how services are delivered, and when it makes sense
- Getting oriented in Azure: the portal, Cloud Shell, the Azure CLI, and SSH keys

The warmup is a check for understanding -- short written answers, no code. The project introduces something new to the cloud weeks: a short video. More on that below.

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_08/`. Inside that folder, create two files:

1. `warmup_08.md` : for the warmup questions
2. `project_08.md` : for the project (cost analysis write-up and video link)

When finished, commit and open a PR as described in the [assignments README](README.md).

# Part 1: Warmup -- Check for Understanding

Answer each question in your own words in `warmup_08.md`. A sentence or two is enough for most questions -- you are demonstrating that you understood the concept, not writing an essay. Try to do this without AI assistance.

## Cloud Concepts

These questions are based on the [Cloud Overview](../lessons/08_cloud_intro/01_cloud_overview.md) lesson.

### Cloud Concepts Question 1

What is the core economic model of cloud computing, and how does it differ from owning your own servers?

### Cloud Concepts Question 2

What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

### Cloud Concepts Question 3

Describe IaaS, PaaS, and SaaS in your own words. For each, give one example from the lesson and describe what you, as the developer, are responsible for managing.

### Cloud Concepts Question 4

What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?

### Cloud Concepts Question 5

The lesson names two situations where the cloud is probably not the right choice. What are they?

## Azure Basics

These questions are based on the [Getting Started with Azure](../lessons/08_cloud_intro/02_azure_intro.md) lesson.

### Azure Basics Question 1

What is the difference between an Azure *subscription* and a *resource group*? Which one is yours alone, and which one does CTD share?

### Azure Basics Question 2

Azure Cloud Shell is ephemeral by default. What does that mean in practice, and what does your course setup use to make it persistent?

### Azure Basics Question 3

What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?

### Azure Basics Question 4

Run the following command in Cloud Shell *without* the `--output table` flag:

```bash
az account show
```

Paste the output into your answer. Then describe in one sentence what changes when you add `--output table`.

# Part 2: Project -- Cloud Orientation

This week's project is intentionally light. The goal is simply to get you logged in, oriented, and set up in Azure -- and to give you some breathing room to catch up on anything from previous weeks if needed. Enjoy! :smile: 

## A Note on Cloud Assignments

Starting this week, each cloud assignment includes a short video. If you did Python 100, you've done this before -- same idea.

Cloud proficiency is different from Python proficiency. It is not primarily about writing code -- it is about navigating an ecosystem: finding the right resources, understanding what things cost, knowing which lever to pull. That is genuinely harder to demonstrate in a `.py` file than it is on screen. Videos let us see that you can actually find your way around.

Keep it concise. The target is 3 minutes; the hard limit is 5. Brevity is part of the skill -- if a professional cloud engineer can't show you something clearly in a few minutes, that's a problem. Your mentors will thank you.

## The Video 

Record a single video (target: 3 minutes, max: 5) with two parts. Part one is a portal walkthrough; part two is a cost analysis on screen. Details below.

Post your video somewhere accessible (whatever you used in Python 100) and paste the link in `project_08.md`.

### Part 1: Portal Walkthrough

Show the following on screen:

1. Navigate to your resource group in the portal. Point out the storage account inside it.
2. Open Cloud Shell. Run `ls ~/clouddrive` and show that your `test.txt` from the persistence exercise is still there.
3. Run `az group list --output table` and briefly say what it shows.
4. Explain anything else you find interesting or curious. 

### Part 2: Cost Analysis

The [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) is a standalone tool -- separate from the Azure portal, no login needed. You search for a service, and click "Add to estimate" -- you can configure it and add it as the running total accumulates in the *Estimate* section at the bottom of the page. Before recording, build estimates for the two scenarios below, then feel free to keep exploring. There are hundreds of services in there -- throw in whatever looks interesting and see what happens to the total. This is a sandbox, not a test.

Imagine you are scoping infrastructure for a data pipeline. Start with these two scenarios (East US, Linux):

**Scenario A -- Lightweight ETL:** A Standard_B1s VM (1 vCPU, 1 GB RAM) running 8 hours a day, 5 days a week (about 160 hours a month).

**Scenario B -- Heavy analytics workload:** A GPU-enabled VM (Standard_NC6s_v3: 6 vCPU, 1 V100 GPU) running 24/7 for the full month (730 hours), an Azure SQL Database (General Purpose tier, 4 vCores), and an Azure Blob Storage account with 1 TB of data.

From there, go wherever curiosity takes you. In your video, pull up the completed estimates and briefly walk through what each scenario costs. In `project_08.md`, write up a summary about the costs and call out anything surprising or interesting you found while poking around.




