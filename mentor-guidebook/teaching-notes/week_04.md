# Week 4: Deep Learning

## Overview

Students made the conceptual leap from classical ML to neural networks. The week started with how neural networks actually work (forward pass, backpropagation, gradient descent) and then moved into hands-on PyTorch work on Kaggle, covering a full training loop, convolutional neural networks for images, and transfer learning. Most of the practical work happens in Kaggle notebooks rather than local files — this is intentional given GPU requirements.

## Key Concepts

**The neuron and the network** — A single neuron takes weighted inputs, sums them, applies an activation function, and produces an output. Stack layers of neurons and you get a neural network. The network learns by adjusting those weights.

**Backpropagation and gradient descent** — Training a neural network means finding weights that minimize a loss function. Gradient descent does this iteratively: compute the loss, calculate gradients (which direction to adjust each weight), take a small step. Backpropagation is the algorithm that efficiently computes those gradients.

**PyTorch basics** — Tensors are PyTorch's equivalent of NumPy arrays, with GPU support built in. The training loop follows a consistent pattern: forward pass → compute loss → backward pass → update weights. Students don't need to memorize this, but they should recognize it when they see it.

**CNNs for images** — Convolutional layers learn spatial features (edges, textures, shapes) by sliding small filters across an image. Pooling layers reduce spatial dimensions. Fully connected layers at the end make the final prediction.

**Transfer learning** — Rather than training from scratch, you take a model pre-trained on a massive dataset (like ImageNet) and fine-tune it for your specific task. This is how almost all practical deep learning is done — students with limited data and compute can get strong results this way.

## Common Questions

- **"Why use Kaggle instead of running this locally?"** — Training neural networks requires a GPU. A model that takes 10 minutes on Kaggle's free T4 GPU would take hours on a typical laptop CPU. Kaggle provides free GPU access, which makes the exercises practical.
- **"What is an epoch?"** — One full pass through the training data. You typically train for multiple epochs, and the loss should generally decrease as training progresses.
- **"Why does my loss go down but my validation loss goes up?"** — Overfitting. The model is memorizing the training data rather than learning generalizable patterns. Common remedies: more data, dropout layers, early stopping, or a simpler architecture.
- **"What's the difference between fine-tuning and feature extraction?"** — Feature extraction: freeze all the pre-trained weights and only train a new classifier head. Fine-tuning: unfreeze some or all of the pre-trained layers and let them continue training on your data.

## Watch Out For

- **Kaggle notebook access** — Students need a Kaggle account and need to enable GPU for their notebooks. Walk anyone through this who gets stuck. The GPU is only available for a limited number of hours per week on the free tier.
- **Slow training expectations** — Even on GPU, some experiments take several minutes. Students may think something is broken when it's just running. Encourage them to watch the loss output for progress.
- **Treating deep learning as a magic box** — The conceptual lesson (`01_deep_intro.md`) is important groundwork. If a student jumped straight to the notebooks without reading it, they may not understand what they're doing. It's worth spending group time on the concepts.
- **Week 4 markdown files are minimal** — The `.md` lesson files for lessons 2–5 are brief descriptions pointing to Kaggle notebooks. The real content is in the notebooks themselves.

## Suggested Activities

1. **Concept check — backprop in plain English:** Ask a student to explain (without math) why the model gets better during training. If they can articulate "the model calculates how wrong it was, figures out which weights contributed most to the error, and adjusts them," they understand the core idea.

2. **Transfer learning intuition:** Ask: "If you wanted to classify X-ray images to detect pneumonia, would you start training from scratch or use a model pre-trained on ImageNet (photos of dogs, cars, etc.)? Why?" The answer (yes, use ImageNet) surprises students — the low-level features a model learns from natural photos (edges, textures) transfer well to medical images.

3. **Loss curve reading:** Have students share their training loss curves from the assignment. Ask: at what epoch did the model converge? Did anyone see validation loss increase while training loss decreased? What does that tell you?
