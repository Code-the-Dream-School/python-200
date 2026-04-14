# Week 8: Introduction to Cloud Computing

## Overview

Students got their first hands-on experience with cloud infrastructure. The first lesson covered cloud computing concepts — the economic model, service tiers (IaaS/PaaS/SaaS), and scaling strategies. The second was a practical walkthrough of the Azure portal: logging in under the Code the Dream tenant, navigating Cloud Shell, setting up persistent storage, and generating SSH keys. The assignment is intentionally light and includes a cost exploration exercise using Azure's Pricing Calculator.

## Key Concepts

**The cloud's core economic model** — Instead of buying servers, you rent compute on demand. You pay for what you use, scale up or down instantly, and let the cloud provider handle maintenance. The tradeoff: you're dependent on the provider, and costs can surprise you if you're not careful.

**IaaS / PaaS / SaaS** — Infrastructure as a Service (you manage the OS and up — e.g., Azure VMs), Platform as a Service (provider manages the runtime, you manage your app — e.g., Azure App Service), Software as a Service (provider manages everything — e.g., Gmail). Most data work lives in IaaS and PaaS territory.

**Horizontal vs. vertical scaling** — Vertical: give a single machine more resources (bigger VM). Horizontal: add more machines and distribute work across them. Horizontal scaling is generally preferred for resilience and cost, but requires software designed to run in parallel.

**Azure portal and Cloud Shell** — The portal is a web UI for managing Azure resources. Cloud Shell is a browser-based terminal (Bash or PowerShell) that runs in Azure's infrastructure. It's ephemeral by default — the course setup uses persistent storage so files survive between sessions.

**SSH keys** — Public-key authentication for remote servers. The private key stays on your machine (never share it). The public key goes on the remote server. Knowing how to generate and use SSH keys is a baseline cloud skill.

## Common Questions

- **"What's the difference between a subscription and a resource group?"** — A subscription is the billing account (CTD's). A resource group is a logical container for resources within that subscription. Each student has their own resource group.
- **"Why is Cloud Shell ephemeral?"** — The shell runs in a temporary container that gets destroyed between sessions. The course sets up a mounted storage account (`clouddrive`) so files written there persist. Files written elsewhere in the shell do not.
- **"How do I know which service tier to use?"** — IaaS for maximum control and custom environments. PaaS when you want to focus on your application and not the infrastructure. SaaS when you just need software to work.
- **"How do I keep my Azure costs from spiraling?"** — Always delete resources you're not using. Set up cost alerts. The course uses a shared CTD subscription — students should not be creating expensive resources without checking with the instructor.

## Watch Out For

- **Tenant and directory confusion** — Students must be logged into the Code the Dream tenant, not their personal Azure account. The top-right corner of the portal shows which tenant is active. If they're in the wrong one, they won't see their resource group.
- **MFA setup** — Some students may not have MFA configured for the CTD account. This needs to be done before they can log in. If someone can't get into the portal at all, this is the first thing to check.
- **Cloud Shell session timeout** — Cloud Shell times out after a period of inactivity. Students who step away and come back may find their session reset. Files in `~/clouddrive/` will still be there, but any in-memory state (running processes, environment variables) will be gone.
- **This week is intentionally light** — The assignment is low-stakes and designed to give students time to catch up on previous weeks if needed. Don't add pressure; reinforce that this is a setup and orientation week.

## Suggested Activities

1. **IaaS/PaaS/SaaS classification exercise:** Go around the group and ask each student to classify one service they use daily (Spotify, GitHub, Heroku, their university's email, etc.) as IaaS, PaaS, or SaaS, and explain why. This makes the abstract tiers concrete.

2. **Cost exploration:** Ask students to share what surprised them most from the Pricing Calculator exercise. Ask: "What would you change about your pipeline design if you knew a GPU VM costs 30x more per hour than a standard VM?"

3. **Cloud Shell live demo:** Ask a student to share their screen with Cloud Shell open. Have them run `az group list --output table` and walk the group through what each column means. Then run `ls ~/clouddrive` — do they still have their `test.txt` from the persistence exercise?
