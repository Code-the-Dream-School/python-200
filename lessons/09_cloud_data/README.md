# Week 9: Data in the Cloud

In Week 8 you got your Azure environment set up -- portal access, Cloud Shell, the Azure CLI, and SSH keys. This week you take the next step: writing Python that talks directly to Azure.

The goal is to add cloud storage to the pipeline toolkit you have been building all course. By the end of the week, you will be able to authenticate a Python script to Azure, read and write files in Blob Storage, and build a simple pipeline that extracts data from a public API and loads it into the cloud. That Extract + Load pattern is the foundation of the capstone pipeline you will build in Week 11.

> For an introduction to the course, and a discussion of how to set up your environment, please see the [Welcome](../README.md) page.

## Topics

1. [Connecting to Azure from Python](01_connecting_azure.md)
Python scripts cannot type a password -- they need a different authentication strategy. This lesson introduces the azure-identity package and DefaultAzureCredential, which handles authentication automatically based on your environment. You will verify the connection by printing your subscription name.

2. [Azure Blob Storage](02_blob_storage.md)
Blob Storage is Azure's service for storing files in the cloud -- CSVs, JSON exports, images, model artifacts, anything a pipeline might produce or consume. This lesson covers the account to container to blob hierarchy and walks through the full set of CRUD operations using the azure-storage-blob SDK.

3. [Loading Pipeline Data to Blob Storage](03_loading_pipeline.md)
Pulls everything together into a realistic pipeline step: extract data from a REST API, serialize it, upload it to Blob Storage at a structured path, and read it back into a pandas DataFrame. This is the Extract + Load skeleton you will build on in Weeks 10 and 11.

## Week 9 Assignments

Once you finish the lessons, head on over to the [assignments](../../assignments/README.md) to get more hands-on practice with the material.
