# Course Objectives: Python for Cloud & AI

### Week 1: Introduction to Data Analysis

- Students will be able to manipulate, clean, and summarize tabular datasets using
  pandas and NumPy at a level sufficient for exploratory data analysis.
- Students will be able to interpret basic statistical measures: including central
  tendency, variance, distributions, and correlation: and use them to draw
  preliminary conclusions from a dataset.
- Students will be able to construct a reproducible data analysis pipeline using
  Prefect, organizing data loading, transformation, and output steps as discrete,
  reusable tasks.

### Week 2: Introduction to Machine Learning

- Students will be able to distinguish between supervised and unsupervised learning
  and between regression and classification problems, selecting the appropriate
  approach for a given task.
- Students will be able to train and evaluate a linear regression model using
  scikit-learn, interpreting performance using MSE, MAE, and R².
- Students will be able to explain the risk of overfitting and demonstrate the
  train/test split as a practical mitigation strategy.

### Week 3: Machine Learning Classification

- Students will be able to preprocess tabular data for machine learning, including
  feature scaling, missing value handling, categorical encoding, and dimensionality
  reduction with PCA.
- Students will be able to train and compare classification models using scikit-learn
  and evaluate their performance using accuracy, precision, recall, F1-score, and
  confusion matrices.
- Students will be able to interpret classifier evaluation results and identify when
  reported performance may be misleading, particularly in the presence of class
  imbalance.

### Week 4: Deep Learning

- Students will be able to explain how neural networks learn through forward passes,
  backpropagation, and gradient descent at a conceptual level sufficient to interpret
  training results and recognize common failure modes.
- Students will be able to use PyTorch to execute the standard training loop :
  defining a model architecture, loss function, and optimizer: on a provided
  dataset, using Kaggle for GPU-accelerated compute.
- Students will be able to apply pretrained convolutional neural networks for image
  inference and adapt them to new classification tasks using transfer learning,
  distinguishing between feature extraction and fine-tuning approaches.

### Week 5: Introduction to AI and LLMs

- Students will be able to explain how large language models work at a conceptual
  level, including the roles of tokenization, embeddings, and training, without
  requiring mathematical depth.
- Students will be able to use the OpenAI API to build chat-based applications,
  managing conversation history, system prompts, and model parameters to control
  output behavior.
- Students will be able to apply prompt engineering techniques: including zero-shot,
  few-shot, and chain-of-thought prompting: to improve the reliability and
  specificity of LLM outputs for a defined task.

### Week 6: AI Augmentation (RAG)

- Students will be able to explain retrieval-augmented generation (RAG), contrast it
  with fine-tuning, and select the appropriate knowledge-augmentation strategy for a
  given use case and budget.
- Students will be able to build a semantic RAG pipeline from scratch using vector
  embeddings and FAISS, implementing document chunking, retrieval, and
  response generation without a framework.
- Students will be able to implement a production-ready RAG system using LlamaIndex,
  connecting a vector store, retriever, and LLM into a coherent pipeline with
  significantly less custom code than a from-scratch implementation.

### Week 7: AI Agents

- Students will be able to explain the ReAct (Reason + Act) loop and differentiate
  tool-based from code-based agent architectures, identifying which pattern is
  appropriate for a given task.
- Students will be able to construct a functional AI agent from scratch in Python
  that uses defined tools to complete a multi-step data analysis task.
- Students will be able to implement equivalent agent behavior using the smolagents
  framework and compare the tradeoffs between custom and framework-based
  approaches in terms of flexibility, debuggability, and development time.

### Week 8: Introduction to Cloud Computing

- Students will be able to explain core cloud computing concepts: including IaaS,
  PaaS, SaaS, horizontal and vertical scaling, and the role of managed data platforms
 : and identify which model is appropriate for a described use case.
- Students will be able to navigate the Azure Portal, configure a persistent Cloud
  Shell environment with mounted storage, generate SSH keys, and run basic Azure CLI
  commands to inspect their cloud environment.

### Week 9: Data in the Cloud

- Students will be able to authenticate a Python script to Azure using
  DefaultAzureCredential and explain, at a conceptual level, how the credential
  chain operates differently in development versus production contexts.
- Students will be able to perform create, read, list, and delete operations on
  Azure Blob Storage using the azure-storage-blob Python SDK.
- Students will be able to build a script that extracts data from a REST API and
  loads it to Azure Blob Storage in a structured path, forming the extract and load
  steps of a cloud ETL pipeline.

### Week 10: LLMs in Pipelines

- Students will be able to identify data transformation tasks that are well-suited
  to LLMs: such as classification, extraction, and summarization: and distinguish
  them from tasks better handled by deterministic code, justifying the choice.
- Students will be able to implement an LLM-assisted transform step that reads
  records from Blob Storage, enriches them using the OpenAI API with a
  structured prompt, and writes results back to a new storage path.
- Students will be able to describe the code changes required to migrate a pipeline
  from the OpenAI API to Azure OpenAI and explain why organizations use Azure
  OpenAI in production environments.

### Week 11: Cloud ETL

- Students will be able to construct a complete cloud ETL pipeline as a Prefect
  flow, decomposing extract (REST API), transform (LLM enrichment), and load
  (Blob Storage) operations into discrete, reusable tasks.
- Students will be able to apply production pipeline patterns: including task
  retries, raise-on-failure error handling, and structured logging: to make a
  Prefect flow resilient to common failures such as transient API errors.
- Students will be able to run and monitor a pipeline end-to-end using the Prefect
  UI, interpreting run states, task logs, and failure traces to verify correct
  execution or diagnose problems.
