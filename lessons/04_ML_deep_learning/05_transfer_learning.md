# CNN Transfer Learning

## Overview

In this lesson, you’ll learn how to use transfer learning to adapt a pretrained Convolutional Neural Network (CNN) to a new image classification task using PyTorch and TorchVision.

Training deep neural networks from scratch requires large datasets, significant computational power, and long training times. Transfer learning solves this problem by reusing knowledge learned from large datasets and adapting it to a new task.

In this lesson, we focus on **fine-tuning a pretrained CNN** for a new classification problem using the Fashion MNIST dataset.

## We will cover:
- What transfer learning means in deep learning
- Why transfer learning is effective for image classification
- How to fine-tune a pretrained CNN using PyTorch
- The difference between feature extraction and full fine-tuning  

## Transfer Learning Lesson

Unlike the CNN inference lesson, here we will train part of a pretrained model on a new dataset.

We will:
- Load a pretrained CNN from TorchVision
- Replace the final classification layer
- Train the modified model on Fashion MNIST
- Evaluate its performance

To avoid installation and environment issues, we recommend running the notebook on Kaggle, which provides PyTorch and GPU support out of the box.

You can open the lesson notebook directly in Kaggle using the link below.
[Open in Kaggle](https://www.kaggle.com/code/niharikamatcha/04-transfer-learning-fashion-mnist)


## What is Transfer Learning?

Transfer learning is a technique where a model trained on one task is reused for a different but related task.

For example:
- A CNN trained on ImageNet has already learned to detect edges, textures, shapes, and patterns.
- We can reuse those learned features to classify a new dataset (e.g., cats vs dogs, medical images, etc.).

This reduces:
- Training time
- Required dataset size
- Risk of overfitting

## Why Use Transfer Learning?

Training deep neural networks from scratch requires:
- Large labeled datasets
- High computational power
- Long training time

Transfer learning allows us to:
- Use pretrained weights
- Replace the final classification layer
- Fine-tune the model for our specific problem

## Additional Resources

For a deeper walkthrough of transfer learning in PyTorch, see:
[Image Classification using Transfer Learning in PyTorch](https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/)

## Check for Understanding:

After completing the lesson notebook, try answering the questions below.

Question 1:
Why is transfer learning effective for image classification?
Choices:
A) CNNs only work on one dataset
B) Early layers learn general visual features useful across tasks
C) It removes the need for labeled data
D) It eliminates the need for GPUs
<details> <summary>View answer</summary> **Answer:** B) Early layers learn general visual features useful across tasks </details>

Question 2:
What does it mean to “freeze” layers in transfer learning?
Choices:
A) Delete the layers from the model
B) Prevent their weights from being updated during training
C) Convert them to NumPy arrays
D) Increase their learning rate
<details> <summary>View answer</summary> **Answer:** B) Prevent their weights from being updated during training </details>

Question 3:
What is typically replaced when adapting a pretrained CNN to a new classification task?
Choices:
A) The convolutional layers
B) The input image
C) The final fully connected classification layer
D) The loss function only
<details> <summary>View answer</summary> **Answer:** C) The final fully connected classification layer </details>

Question 4:
What would likely happen if you trained a very deep CNN from scratch on a small dataset?
Choices:
A) The model would generalize perfectly
B) The model would train instantly
C) The model might overfit
D) The model would not require labeled data
<details> <summary>View answer</summary> **Answer:** C) The model might overfit </details>

Question 5:
What would likely happen if you trained a very deep CNN from scratch on a small dataset?
Choices:
A) The model would generalize perfectly
B) The model would train instantly
C) The model might overfit
D) The model would not require labeled data
<details> <summary>View answer</summary> **Answer:** C) The model might overfit </details>

Question 6:
What is the main difference between **feature extraction** and **fine-tuning** in transfer learning?
Choices:
A) Feature extraction trains all layers, fine-tuning trains none
B) Feature extraction freezes pretrained layers, fine-tuning updates some or all of them
C) Feature extraction requires no data
D) Fine-tuning removes convolutional layers
<details> <summary>View answer</summary> **Answer:** B) Feature extraction freezes pretrained layers, fine-tuning updates some or all of them </details>