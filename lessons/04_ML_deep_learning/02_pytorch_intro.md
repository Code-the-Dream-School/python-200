# Introduction to PyTorch
PyTorch is an open-source tool for building neural network models, originally developed by Facebook's AI Research lab. Because of its intuitive design, it has become one of the most popular frameworks for deep learning and artificial intelligence research.

Aside from its intuitive interface, PyTorch has many features that make it a great choice for building deep learning models. It includes a basic library for tensor computations, very similar to NumPy, but with strong GPU acceleration support. It also provides built-in tools to build neural networks and calculate gradients, which is required for backpropagation. Also, unlike the other major frameworks like Tensorflow, PyTorch works very hard to ensure it is easy to install and integrate with your GPU on multiple platforms (including Windows).

For this first lesson, we want to just build some familiarity with PyTorch independently of neural networks, and focus on its ability to do numerical computing similar to NumPy but with GPU acceleration. 

For this lesson, we will use the official PyTorch tutorial notebook on tensors. While you can run it locally if you install PyTorch on your machine, we recommend using Kaggle to avoid installation issues:

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/Code-the-Dream-School/python-200/blob/main/lessons/04_ML_deep_learning/resources/pytorch_tensors.ipynb)

Click the "Open in Kaggle" button above to open the notebook in Kaggle. 

In that notebook, focus on these sections:
- Creating tensors
- Random tensors and seeding
- Shapes and dtypes
- Basic mathematical operations

What to skim for now:
- Broadcasting
- requires_grad / autograd-related details

For now, we are just treating PyTorch as a library for tensor operations, similar to NumPy. In future lessons, we will build on this foundation to create neural networks and train them using backpropagation (as discussed in the introduction to neural networks and deep learning). 

## Additional Resources
If you want additional material on learning basic PyTorch:
- [learnpytorch.io](https://www.learnpytorch.io/00_pytorch_fundamentals/)
- [YouTube Video](https://www.youtube.com/watch?v=v43SlgBcZ5Y)

In that video the author assumes you are using conda to manage your Python environment, but you can just import torch in Kaggle without worrying about that. Also, if you want to install PyTorch on your own machine, you can find out how at their [official installation page](https://pytorch.org/get-started/locally/). It is very easy to install compared to other deep learning frameworks.






