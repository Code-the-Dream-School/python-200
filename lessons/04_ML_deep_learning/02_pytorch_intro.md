# Introduction to PyTorch
As discussed in the [Intro to Deep Learning](01_deep_intro.md) lesson, PyTorch is an open-source tool for building neural network models, originally developed by Facebook's AI Research lab. Because of its intuitive design, it has become one of the most popular frameworks for deep learning and artificial intelligence research. It is used to train many of the state-of-the-art models in natural language processing (NLP) and computer vision (CV), including many of the models developed by OpenAI such as ChatGPT. 

For more background on PyTorch, a bit on its usage and history, you can check out the first few sections of this lesson at learnpytorch.io: [https://www.learnpytorch.io/00_pytorch_fundamentals/](https://www.learnpytorch.io/00_pytorch_fundamentals/). 

Aside from its intuitive interface, PyTorch has many features that make it a great choice for building deep learning models. It includes a basic library for tensor computations, very similar to NumPy, but with strong GPU acceleration support. It also provides built-in tools to build neural networks and perform backpropagation. Also, unlike the other major frameworks like Tensorflow, PyTorch works very hard to ensure it is easy to install and integrate with your GPU on multiple platforms (including Windows).

## Pytorch lesson
For this first lesson, we want to just build some familiarity with PyTorch independently of neural networks, and focus on its ability to do numerical computing similar to NumPy but with GPU acceleration. 

Toward that end, we will work through an official PyTorch tutorial notebook on tensors. While you can run it locally if you install PyTorch on your machine, we recommend using Kaggle to avoid installation issues (this will let us avoid any potential headaches with setting up PyTorch and GPU support on your local machine).

You can open the notebook directly in Kaggle using the button below.
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/Code-the-Dream-School/python-200/blob/ml/pytorch_intro/lessons/04_ML_deep_learning/resources/pytorch_tensors.ipynb)

Click the "Open in Kaggle" button above to open the notebook in Kaggle. You should already have a Kaggle account from Python 100, but if not please reach out to us for help! Once you open the notebook there, you will have a full working notebook where you can run the code, interact and tinker to optimize your learning experience. 

In that notebook, focus on these sections:
- Creating tensors
- Random tensors and seeding
- Shapes and dtypes
- Basic mathematical operations

What to skim for now (more advanced topics)::
- Broadcasting
- Requires_grad / autograd-related details

For this lesson, we are *just* treating PyTorch as a library for tensor operations, similar to NumPy. In subsequent lessons we will build on this foundation to create neural networks and train them using backpropagation (the learning rule discussed in the [Intro to Deep Learning](01_deep_intro.md) lesson).

## Additional Resources

For additional material on learning basic PyTorch:
- [learnpytorch.io](https://www.learnpytorch.io/00_pytorch_fundamentals/)
- [YouTube Video](https://www.youtube.com/watch?v=v43SlgBcZ5Y)

To install PyTorch locally, you can find out how at their [official installation page](https://pytorch.org/get-started/locally/). It is very easy to install compared to other deep learning frameworks.


## Check for Understanding
After working through the lesson above, you can work through the following questions as a quick check for understanding of some of the concepts you learned in the leasson.

### Question 1
Which statement best describes how PyTorch tensors compare to NumPy arrays?

Choices:

    - A) Tensors are only for images and cannot store numbers.
    - B) Tensors are like NumPy arrays, but can also run efficiently on a GPU (when available).
    - C) Tensors are always Python lists under the hood.
    - D) Tensors can only store integers.

<details>
<summary>View answer</summary>
**Answer:** B) Tensors are like NumPy arrays, but can also run efficiently on a GPU (when available).
</details>


### Question 2
You run this code twice:

```python
torch.manual_seed(42)
x = torch.rand(3)
```

What should you expect?

Choices:

    - A) `x` will be different each time because random is always unpredictable.
    - B) `x` will be the same each time because the seed resets the random number generator.
    - C) `x` will always be all zeros.
    - D) This will raise an error because `manual_seed` only works for integers.

<details>
<summary>View answer</summary>
**Answer:** B) `x` will be the same each time because the seed resets the random number generator.
</details>


### Question 3
You create a tensor with shape `(4, 3)`.

What does that shape mean?

Choices:

    - A) The tensor has 12 dimensions.
    - B) The tensor has 4 rows and 3 columns (12 total elements).
    - C) The tensor has 4 total elements and each element has 3 bytes.
    - D) The tensor must store strings.

<details>
<summary>View answer</summary>
**Answer:** B) The tensor has 4 rows and 3 columns (12 total elements and two dimensions).
</details>


### Question 4
You have:

```python
a = torch.tensor([1, 2, 3])       # dtype is an integer type
b = torch.tensor([1.0, 2.0, 3.0]) # dtype is a floating type
c = a + b
```

Which statement is most accurate?

Choices:

    - A) This will fail because you cannot add tensors with different dtypes.
    - B) `c` will be an integer tensor because integers are "more strict."
    - C) `c` will be a floating tensor because PyTorch promotes the result to a type that can represent decimals.
    - D) `c` will be a Python list.

<details>
<summary>View answer</summary>
**Answer:** C) `c` will be a floating tensor because PyTorch promotes the result to a type that can represent decimals.
</details>





