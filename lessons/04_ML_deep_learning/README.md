# Week 4: Machine Learning: Deep Learning
> Draft readme to be completed once material is finished.
> 
Welcome to Week 4 of Python 200! This week we will explore the world of deep learning, which is a powerful subset of machine learning that uses neural networks to model complex patterns in data. This is probably the most ambitious week of the course, but also one of the most exciting! This is the basis of modern AI systems like ChatGPT, DALL-E, and many others. By understanding deep learning, you will gain insights into how these systems work, and how to help build pipelines and applications that leverage them. Most modern machine learning applications use deep learning in some form, so this knowledge is very useful in modern data engineering, data science, and cloud computing roles.

After exploring the fundamentals of deep learning, we will use the pytorch library to explore basics of neural network models. The focus will be on understanding how deep learning models work, how to implement them using pytorch, and how to evaluate their performance. 

Our goal is to equip you with the skills needed to build and deploy deep learning models for various applications, not to become deep learning researchers or academics. As usual, the focus is on building intuition and understanding, not on deep mathematical theory. 

> To fill in later: brief motivational preview here. Briefly explain why this lesson matters, what students will be able to do by the end, and what topics will be covered. Keep it tight and motivating.

> For an introduction to the course, and a discussion of how to set up your environment, please see the [Welcome](../README.md) page.  

## Topics
1. [Introduction to neural networks and deep learning](01_ann_intro.md)  
A brief introduction to neural networks and deep learning, including biological inspiration, architecture, and common use cases.

2. [Introduction to pytorch](02_pytorch_intro.md)  
An overview of the pytorch library, mainly focusing on installation, and tensor operations: we will direct you to the excellent tutorial at learnpytorch.io to learn the basics of pytorch.

1. [Training your first neural network](03_xor.md)  
Before diving into more complex models, we will start with a simple example with a neural network. This will help us understand the basic workflow for training a model in pytorch.

1. [Machine vision](03_machine_vision_cnn.md)  
For this section, we will use a pre-trained convolutional neural network (CNN) to classify images, to show how deep neural networks can be used for computer vision tasks.

1. [Transfer learning](04_transfer_learning.md)
Here, we will explore transfer learning, which is a technique that allows us to leverage pre-trained models for new tasks with limited data. We will fine-tune a pre-trained model on a new dataset to demonstrate this concept.