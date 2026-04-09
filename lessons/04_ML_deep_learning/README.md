# Week 4: Machine Learning: Deep Learning
Welcome to Week 4 of Python 200! This week we will explore the world of deep learning, a powerful subset of machine learning that uses neural networks to recognize complex patterns in data. This is probably the most ambitious week of the course, but also one of the most exciting.

Deep learning is the foundation of all modern AI systems like ChatGPT (text-based LLMs), DALL-E (image-generating AI), and many others. By understanding deep learning, you will gain insights into how these systems work and how to help build pipelines and applications that leverage them. The focus is on building intuition and understanding -- not deep mathematical theory.

Lessons are structured a bit differently this week. The standard markdown page introduces the lesson, but the main work is happening in Juptyer notebooks. Each lesson landing page has an "Open in Kaggle" button that loads the notebook directly into your account -- the notebooks live in `resources/notebooks/` and are named to match their lesson. This lets us sidestep the complex installation issues that arise with GPU-based computation. Kaggle provides 30 hours of free GPU time per week, which is more than enough for this course.

> If you want to install PyTorch on your own machine for practice, you can find out how at their [official installation page](https://pytorch.org/get-started/locally/). Just bear in mind that your mentors won't be able to provide support for debugging local PyTorch issues. 

## Topics
1. [Introduction to neural networks and deep learning](https://github.com/Code-the-Dream-School/python-200/blob/main/lessons/04_ML_deep_learning/01_deep_intro.md)
A brief introduction to neural networks and deep learning. What are they, how do they work, how do they learn?

2. [Introduction to pytorch](https://github.com/Code-the-Dream-School/python-200/blob/main/lessons/04_ML_deep_learning/02_pytorch_intro.md)
An overview of the PyTorch library, mainly focusing on tensor operations (in the world of deep learning, they use the word "tensor" instead of "array", but it's still just arrays of numbers like we saw with NumPy).

3. [Training your first neural network](https://github.com/Code-the-Dream-School/python-200/blob/main/lessons/04_ML_deep_learning/03_first_neural_network.md)
Before diving into more complex models, we will start by training a simple neural network in PyTorch to gain an understanding of the basic workflow for training and evaluating a model -- a workflow that carries over to much more complex architectures.

4. [Machine vision intro](https://github.com/Code-the-Dream-School/python-200/blob/main/lessons/04_ML_deep_learning/04_cnn_inference.md)
For this section, we will use a pre-trained convolutional neural network (CNN) to classify images. We use this to show how deep neural networks can be used for computer vision tasks.

5. [Transfer learning](https://github.com/Code-the-Dream-School/python-200/blob/main/lessons/04_ML_deep_learning/05_transfer_learning.md)
You rarely train a complex neural network from scratch. Here, we will explore *transfer learning*, in which a neural network trained on one task is fine-tuned on a similar task, but requires much less data. Many deep-learning pipelines are really just transfer learning pipelines under the hood.

## Week 4 Assignments
Once you finish the lessons, continue below to the assignment for this week.
