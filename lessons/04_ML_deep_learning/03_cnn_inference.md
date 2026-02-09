# CNN Inference with a Pretrained Model

In this lesson, you’ll learn how to use a pretrained Convolutional Neural Network (CNN) to perform inference on new images using PyTorch and TorchVision.

Convolutional Neural Networks (CNNs) are a core architecture for working with image data. In practice, however, we often do **not** train CNNs from scratch. Instead, we use **pretrained models** that have already learned rich visual features from large datasets.

In this lesson, we focus on **inference**, using an already trained CNN to make predictions on new, unseen images.

We will cover:
- What inference means in the context of deep learning
- Why pretrained CNNs are commonly used in real-world applications
- How to run inference using PyTorch and TorchVision without training a model

## CNN Inference Lesson

For this lesson, we are **not training** a neural network. Instead, we will use a pretrained CNN provided by TorchVision and run it in inference mode to classify images.

To avoid installation and environment issues, we recommend running the notebook on **Kaggle**, which provides PyTorch and GPU support out of the box.

You can open the lesson notebook directly in Kaggle using the link below.

[Open in Kaggle](https://www.kaggle.com/code/niharikamatcha/lesson-3-cnn-inference-pretrained)

Once opened in Kaggle, you will be able to run the notebook cells, experiment with different images, and observe how a pretrained CNN produces predictions.

## What to Focus On

While working through the notebook, focus on these concepts:

- What it means for a model to be in **evaluation (inference) mode**
- Loading pretrained CNN models from TorchVision
- Image preprocessing and transformations
- Understanding model outputs as **class probabilities**

## Why Pretrained Models Matter

Pretrained CNNs are trained on massive datasets and have already learned useful visual features such as edges, shapes, and textures. Using these models allows us to:
- Save training time and compute resources
- Achieve strong performance with minimal setup
- Apply deep learning models to real-world problems quickly

This approach is widely used in applications such as image classification, medical imaging, facial recognition, and autonomous systems.

## Additional Resources

If you want a deeper intuition for how CNNs work:
- [A Visual Guide to Convolutional Neural Networks](https://saturncloud.io/blog/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way/)

For additional background on image classification using pretrained models in PyTorch, see this tutorial from LearnOpenCV:
- [PyTorch Image Classification with Pretrained Models](https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/)


## Check for Understanding

After completing the lesson notebook, try answering the questions below.

### Question 1
What does **inference** mean in deep learning?

Choices:

- A) Training a neural network by updating its weights
- B) Using a trained model to make predictions on new data
- C) Designing the architecture of a neural network
- D) Initializing model parameters randomly

<details>
<summary>View answer</summary>
**Answer:** B) Using a trained model to make predictions on new data
</details>

---

### Question 2
Why are pretrained CNN models commonly used?

Choices:

- A) They require no data at all
- B) They always outperform custom-trained models
- C) They reuse features learned from large datasets, saving time and compute
- D) They only work for small images

<details>
<summary>View answer</summary>
**Answer:** C) They reuse features learned from large datasets, saving time and compute
</details>

---

### Question 3
Why do we set a model to `eval()` mode during inference?

Choices:

- A) To enable weight updates
- B) To disable layers like dropout and batch normalization training behavior
- C) To increase the learning rate
- D) To convert the model into NumPy arrays

<details>
<summary>View answer</summary>
**Answer:** B) To disable layers like dropout and batch normalization training behavior
</details>

---

### Question 4
What does the output of a CNN during inference usually represent?

Choices:

- A) Raw image pixels
- B) Updated model weights
- C) Probabilities or scores for each possible class
- D) Training loss values

<details>
<summary>View answer</summary>
**Answer:** C) Probabilities or scores for each possible class
</details>
