# Week 8 Assignments

This week's assignments cover the cloud computing concepts from the two Week 8 lessons:

- Core cloud concepts: what cloud computing is, how services are delivered, and when it makes sense
- Getting oriented in Azure: the portal, Cloud Shell, the Azure CLI, and SSH keys

The warmup is basically a check for understanding -- short written answers, no code. The mini-project introduces something new to the cloud weeks: a short video. More on that below.

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_08/`. Inside that folder, create two files:

1. `warmup_08.md` : for the warmup questions
2. `project_08.md` : for the project (cost analysis write-up and video link)
3. `project_08.py` : a short script you will run in Cloud Shell (details in Part 2)

When finished, commit and open a PR as described in the [assignments README](README.md).

# Part 1: Warmup -- Check for Understanding

Answer each question in your own words in `warmup_08.md`. A sentence or two is enough for most questions -- you are demonstrating that you understood the concept, not writing an essay. Try to do this without AI assistance.

## Cloud Concepts

These questions are based on the [Cloud Overview](../lessons/08_cloud_intro/01_cloud_overview.md) lesson.

### Cloud Concepts Question 1

What is the core economic model of cloud computing, and how does it differ from owning your own servers?

### Cloud Concepts Question 2

What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

- A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
- A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
- A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

### Cloud Concepts Question 3

Before writing your definitions, classify each item in the list below as **IaaS**, **PaaS**, or **SaaS**. One sentence of reasoning is enough for each.

- Gmail
- Azure Virtual Machines
- Azure App Service
- AWS S3 (Simple Storage Service)
- GitHub Codespaces
- Snowflake

Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

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

# Part 2: Project -- Intro and Cost Analysis

This week's project is intentionally light. The goal is simply to get you logged in, oriented, and set up in Azure -- and to give you some breathing room to catch up on anything from previous weeks if needed. Enjoy! :smile: 

## A Note on Cloud Assignments

Starting this week, each cloud assignment includes a short video. If you took Python 100, you've done this before. 

Cloud proficiency is different from Python proficiency. It is not *primarily* about writing code -- it is about navigating an ecosystem: finding the right resources, understanding what things cost, knowing which lever to pull. In many cases, videos are a much easier way for you to demonstrate that you know your way around.

Please keep it short. The target is 3 minutes; the **hard limit** is 5. Clearly explaining complex concepts in a short amount of time is an important skill in the job force, so this is good practice.  

## The Video 

Record a single video (target: 3 minutes, max: 5) with audio. Briefly narrate what you are showing as you go through each part.

Post your video somewhere accessible (whatever you used in Python 100) and paste the link in `project_08.md`.

### Part 1: Portal Walkthrough

Show the following on screen, narrating briefly as you go:

1. Your Azure portal, logged in under the Code the Dream tenant ("Code the Dream" should be visible in the account/directory selector at the top right).
2. Navigate to your personal resource group (`p200-year-<yourname>-rg`). Point out the storage account inside it.
3. Open Cloud Shell. Run `ls ~/clouddrive` and show that your `test.txt` from the persistence exercise is still there.
4. Run `ls ~/.ssh` and show that both your private and public key files are present.
5. Run `az group list --output table` and briefly explain what it shows.
6. Feel free to click around and point out anything else you find interesting or puzzling.

### Part 2: Cost Analysis

The [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) is a standalone tool -- separate from the Azure portal (no login needed). 

The way it works is that you search for a service, and click "Add to estimate" -- you configure the particular service (e.g., compute) and add it as the running total accumulates in the *Estimate* section at the bottom of the page. 

Before recording, build estimates for the two scenarios below, then feel free to keep exploring. There are hundreds of services in there -- throw in whatever looks interesting and see what happens to the total. This is meant to be a fun exercise in cost exploration.

In business contexts, you will often be asked to give a low and a high-end estimage of a budget for a workflow, so spinning out cost scenarios like this is a very useful practical skill. 

For your project, imaging you are scoping infrastructure for a data pipeline. Start with these two scenarios (East US, Linux), which are meant to be low and high end estimates for the infrastructure costs:

**Scenario A -- Lightweight Compute:** A Standard_B1s VM (1 vCPU, 1 GB RAM) running 8 hours a day, 5 days a week (about 160 hours a month).

**Scenario B -- Heavy analytics workload:** A GPU-enabled VM (Standard_NC6s_v3: 6 vCPU, 1 V100 GPU) running 24/7 for the full month (730 hours), an Azure SQL Database (General Purpose tier, 4 vCores), and an Azure Blob Storage account with 1 TB of data.

In your video, pull up the completed estimates and briefly walk through what each scenario costs. In `project_08.md`, write up a summary about the costs and discuss anything surprising or interesting you found in your exploration. 

### Part 3: Python in Cloud Shell

Write `project_08.py` using the hourly rates you found in the Pricing Calculator. The script is short -- just fill in your two rates and run it.

```python
# project_08.py
# Run this in Azure Cloud Shell after completing the Cost Analysis above.

# Fill in the hourly rates from your two Pricing Calculator estimates.
rate_a = 0.0    # Standard_B1s hourly rate (Scenario A)
rate_b = 0.0    # Standard_NC6s_v3 hourly rate (Scenario B, VM only)

hours_a = 160   # Scenario A: 8h/day, 5 days/week, ~4 weeks
hours_b = 730   # Scenario B: always on

cost_a = rate_a * hours_a
cost_b = rate_b * hours_b

print("=== Monthly Cost Estimates ===")
print(f"Scenario A (lightweight):       ${cost_a:.2f}")
print(f"Scenario B (GPU VM only):       ${cost_b:.2f}")

if cost_a > 0:
    print(f"Scenario B VM costs {cost_b / cost_a:.1f}x more than Scenario A")
```

To run it in Cloud Shell, you have two options:

**Option A (recommended): Pull from GitHub**

Commit `project_08.py` to your repo first, then in Cloud Shell:

```bash
# First time
git clone https://github.com/<your-username>/python200-homework.git

# If you have cloned it before
cd python200-homework
git pull
```

Then run the script:

```bash
cd assignments_08
python3 project_08.py
```

**Option B (backup): Upload via the toolbar**

Cloud Shell has a file upload button in the top-right corner of the shell panel. Click it, upload `project_08.py`, and run:

```bash
python3 project_08.py
```

Show the terminal output in your video -- this is the main thing to capture for Part 3.

## Write-Up

In `project_08.md`, write a short summary (a few sentences to a paragraph) covering:

- What each scenario costs, and whether the numbers surprised you.
- Anything interesting you found while exploring the Pricing Calculator beyond the two required scenarios.
- What the script printed, and whether the calculated costs matched what you saw in the Pricing Calculator. (They should -- if they don't, note the discrepancy.)
