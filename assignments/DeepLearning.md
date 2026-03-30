# Week 4 Assignments

This week's assignments build on your introduction to machine learning and take a step into **deep learning with PyTorch**. You’ll work with real tools used in industry and begin to think about models not just as algorithms, but as systems that must run efficiently in real-world environments.

Over the course of this assignment, you will explore:

- PyTorch tensors and the fundamentals of GPU-accelerated computing  
- Loading and inspecting pretrained convolutional neural networks  
- Running inference with pretrained CNNs using TorchVision  
- Comparing multiple pretrained models and understanding their real-world tradeoffs  

Unlike previous weeks, this work happens inside **Kaggle notebooks** instead of local `.py` scripts. This is intentional. Deep learning benefits heavily from GPU acceleration, and Kaggle provides free access to GPUs without any setup overhead. This allows you to focus on learning the concepts rather than configuring environments.

As before, the assignment is split into two parts:

- The **warmup** builds core intuition and familiarity with PyTorch  
- The **mini-project** asks you to apply these ideas in a more open-ended, production-style setting  

---

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_04/`. Inside that folder, include:

1. `warmup_04.ipynb` — your completed warmup notebook (downloaded from Kaggle)  
2. `project_04.ipynb` — your completed project notebook (downloaded from Kaggle)  
3. `outputs/` — any figures or images your code generates  

To download a notebook from Kaggle:  
**File → Download Notebook** will save a `.ipynb` file that you can commit directly.

When finished, commit your work and open a PR as described in the assignments README.

- **Primary submission**: your GitHub PR link  
- **Optional**: a short Loom video (2–5 minutes) walking through your notebook and explaining your results  

The Loom is not required, but explaining technical work out loud is excellent practice and often leads to clearer understanding.

---

# Part 1: Warmup Exercises

All warmup work should live in a single Kaggle notebook: `warmup_04.ipynb`.

Use markdown cells to clearly label sections (e.g. `## PyTorch Tensors`, `### Q1`).  
Use `print()` statements to show outputs so your results are easy to review.

---

## Getting Set Up (Kaggle)

Before writing any code, take a moment to configure your environment properly.

- **Enable GPU**  
  Go to **Settings → Accelerator → GPU**  
  If your device prints `"cpu"` later, this is the first thing to check.

- **Add dataset**  
  Search for **Intel Image Classification (Puneet Bansal)** and add it  
  You will use this dataset later in both warmup and project

---

### A quick note on Kaggle GPUs

Kaggle provides free GPU access, but there are a few practical limitations:

- Sessions can time out if idle  
- Weekly GPU usage is limited  
- You may occasionally need to restart your session  

To avoid losing work:

- Save versions frequently (**File → Save Version**)  
- If disconnected, reopen and rerun cells  
- If GPU runs out, your code will still work on CPU (just slower)  

If setup issues block your progress, reach out — don’t stay stuck.

---

### Output Directory

Add this near the top of your notebook:

```python
import os
os.makedirs("outputs", exist_ok=True)
````

---

### Setup Block

You’ll use this pattern in nearly every PyTorch workflow:

```python
import torch
import torchvision
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"PyTorch version:     {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
```

Take a moment to notice what device you’re on — this will matter throughout the assignment.

---

## PyTorch Tensors

Tensors are the foundation of everything in deep learning.
They represent inputs, model weights, predictions, and losses.

If NumPy arrays feel familiar, tensors will too — but with one key difference:

> Tensors can run on GPUs, enabling massive parallel computation.

---

### Tensor Question 1

Create a few simple tensors and inspect them:

```python
a = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

b = torch.zeros(2, 3)
c = torch.ones(4)
```

Print:

* values
* shape
* dtype
* device

Then reflect:

> Where are these tensors stored right now?
> Why would it matter if your model and your data are on different devices?

---

### Tensor Question 2

Work with a simple vector:

```python
x = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])
```

Compute:

* square root
* sum
* mean
* index of the maximum value

Think about this:

> In a classifier with 1,000 outputs, what does `.argmax()` represent?

---

### Tensor Question 3

Move data between devices:

```python
a_gpu = a.to(device)
a_numpy = a_gpu.cpu().numpy()
```

Consider:

> Why does PyTorch require moving data back to CPU before converting to NumPy?
> What does that tell you about how NumPy works?

---

### Tensor Question 4

Reshape and manipulate dimensions:

```python
t = torch.arange(24).float()
```

Try different shapes and add a new dimension.

Focus on this idea:

> Neural networks expect **batches of data**, not single examples.
> What role does `.unsqueeze(0)` play in making that possible?

---

### Tensor Question 5

Compare NumPy and PyTorch matrix multiplication.

This is more than a syntax exercise:

> Matrix multiplication is the core operation inside neural networks.
> Every layer transforms data using these operations.

---

## Pretrained Models

Training a neural network from scratch is expensive — often requiring:

* millions of labeled images
* long training times
* significant compute resources

Pretrained models solve this by starting from networks that already learned useful visual features.

TorchVision provides these models out of the box.

---

### Model Question 1

Load ResNet18 and inspect its size.

Reflect:

> What does it mean that this model already has millions of learned parameters?
> Why is starting from pretrained weights so valuable?

---

### Model Question 2

Print the architecture and explore it.

Focus less on memorizing details, more on understanding structure:

> What does it mean for a model to be “deep”?
> Where does the final prediction happen?

---

### Model Question 3

Prepare the model for inference:

```python
model = model.to(device)
model.eval()
```

Think carefully:

> Why must the model and input live on the same device?
> What changes when a model switches to evaluation mode?

---

### Model Question 4

Inspect preprocessing:

```python
preprocess = weights.transforms()
```

This step is critical.

> Why do models expect inputs in a very specific format?
> What happens if preprocessing is incorrect?

---

## Running Inference

You now have everything needed to make predictions on real images.

This is where the pieces come together:

* preprocessing
* model forward pass
* interpreting outputs

---

### Inference Question 1

Run the model on a single image.

Focus on interpretation:

> The model wasn’t trained on your dataset specifically.
> Do the predictions still make sense?

---

### Inference Question 2

Run across all scene types.

Look for patterns:

> Which images are easy?
> Which are confusing?
> Why might that be?

---

### Inference Question 3

Compare logits vs probabilities.

Key idea:

> Neural networks output raw scores first — probabilities are derived afterward.

---

### Inference Question 4

Visualize predictions.

Think beyond code:

> How would this look in a real product?
> What would a non-technical user need to see?

---

# Part 2: Mini-Project — Model Comparison

At this point, the focus shifts.

Instead of just running a model, you will now evaluate and compare multiple models — exactly what happens in real-world ML workflows.

The goal is not to find “the best model,” but to understand tradeoffs:

* speed vs accuracy
* simplicity vs performance
* practicality vs theoretical strength

---

## Task 1 — Data Setup

Load a consistent set of images.

Important idea:

> Every model should be tested on the same data to ensure fair comparison.

---

## Task 2 — Baseline Model (ResNet18)

Start with a single model and understand its behavior deeply.

Pay attention to:

* confidence levels
* differences across classes

> A model can be highly confident and still be wrong.

---

## Task 3 — Multi-Model Comparison

Introduce additional models.

Think like an engineer:

> Do models agree?
> If not, what does that tell you?

---

## Task 4 — Speed vs Accuracy

Measure inference time.

This is where ML meets reality:

> A model is only useful if it meets system constraints.

---

## Task 5 — Feature Extraction

Use the model as a feature extractor.

This reveals something deeper:

> Even without retraining, pretrained models already organize visual information meaningfully.

---

## Task 6 — Final Recommendation

Bring everything together.

You are no longer just running code — you are making a decision.

Consider:

* performance
* speed
* reliability
* real-world constraints

---

## Final Thought

Deep learning can feel complex at first, but what you are doing here is exactly what happens in practice:

* loading real models
* running real data
* evaluating tradeoffs
* making decisions

That’s the core of applied machine learning.
