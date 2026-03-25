# Week 4 Assignments

This week's assignments cover the Week 4 material, including:

- PyTorch tensors and the fundamentals of GPU-accelerated computing
- Loading and inspecting pretrained convolutional neural networks
- Running inference with pretrained CNNs using TorchVision
- Comparing multiple pretrained models and understanding their real-world tradeoffs

Unlike previous weeks, this week's assignments live in **Kaggle notebooks** rather than local `.py` scripts. Neural network inference benefits significantly from GPU acceleration, and Kaggle's free GPU tier is the most frictionless way to access it. You will download your completed notebooks and submit them as part of a PR, exactly as described below.

As with previous weeks, the warmup exercises are meant to build muscle memory for the core mechanics — try to work through them without AI assistance. The mini-project will ask you to apply these tools in a more open-ended, production-flavored context.

# Submission Instructions

In your `python200-homework` repository, create a folder called `assignments_04/`. Inside that folder, place:

1. `warmup_04.ipynb`  : your completed warmup notebook, downloaded from Kaggle
2. `project_04.ipynb` : your completed project notebook, downloaded from Kaggle
3. `outputs/`         : any figures your code saves

To download a notebook from Kaggle: **File → Download Notebook** saves a `.ipynb` file you can commit directly.

When finished, commit and open a PR as described in the [assignments README](README.md).

**Primary submission**: A link to your open GitHub PR.

**Optional second link**: A short Loom walkthrough (2–5 minutes) where you run a few cells from your project notebook and explain what the model is doing and what you found. This is not required, but reviewers find it helpful for richer feedback, and explaining technical work out loud is excellent practice for interviews and team standups.

Submit your link(s) using the submission form.

# Part 1: Warmup Exercises

Put all warmup exercises in a single Kaggle notebook saved as `warmup_04.ipynb`. Use markdown cells to label each section and question (e.g. a cell containing `## PyTorch Tensors` and another containing `### Q1`). Use `print()` to display all outputs.

Before you start, configure your Kaggle notebook:

**Enable GPU**: Click **Settings** in the right-hand sidebar → **Accelerator** → select **GPU T4 x2** (or any available GPU option). If `device` prints `cpu` when you run the setup block below, double-check this setting before continuing.

**Add the dataset**: Click **Add Data** (top right) → search for **Intel Image Classification** by Puneet Bansal → click **Add**. The dataset will be available at `/kaggle/input/intel-image-classification/`. You will use it again in the project notebook, so note the steps.

> **A note on Kaggle's free GPU tier**: Kaggle provides free GPU access, but the weekly quota is limited and sessions can time out after a period of inactivity. A few practical tips to avoid losing work:
> - Save your notebook frequently using **File → Save Version** so you have a checkpoint to return to. Kaggle autosaves, but an explicit version is more reliable.
> - If your session disconnects, your code is still there — just reopen the notebook and re-run from the top.
> - If you exhaust your weekly GPU quota, everything in this assignment will still run on CPU — just more slowly. The benchmarking numbers in Project Task 4 will look different on CPU, which is fine to note in your comments.
> - If you run into persistent barriers — quota limits, session errors, or anything that's blocking your progress — reach out to your CIL (Cohort Instructional Leader). They can help you work through environment issues or adjust your timeline as needed. Don't spend hours stuck on a setup problem when one message can get you unstuck.

**Create your output directory**: Add this to a cell near the top of the notebook:

```python
import os
os.makedirs("outputs", exist_ok=True)
```

Run this setup block once at the top of your notebook:

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

# The standard device check — you'll use this pattern in every PyTorch notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version:     {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
```

## PyTorch Tensors

A PyTorch tensor is the fundamental data type for everything in deep learning — model weights, input images, predictions, and loss values are all tensors. If NumPy arrays feel familiar at this point, tensors will too. They support the same slicing, shapes, and arithmetic. The key difference is that tensors can live on a GPU, which is what makes the massively parallel computation behind neural networks practical.

### Tensor Question 1

Create the following tensors and, for each one, print its value, `shape`, `dtype`, and `device`:

```python
a = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

b = torch.zeros(2, 3)
c = torch.ones(4)
```

Add a comment: what device are these tensors on right now? If you were running a training loop on the GPU, why would it matter that your model weights and your input tensors are on the *same* device?

### Tensor Question 2

Starting from:

```python
x = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])
```

1. Compute and print the element-wise square root using `torch.sqrt()`.
2. Compute and print the sum using `.sum()`.
3. Compute and print the mean using `.mean()`.
4. Find and print the index of the maximum value using `.argmax()`.

Add a comment: `.argmax()` appears in nearly every inference example you'll encounter. In the context of a classifier that outputs scores for 1,000 classes, what does `.argmax()` give you?

### Tensor Question 3

Move tensor `a` from Question 1 to the GPU, then bring it back to CPU and convert it to a NumPy array:

```python
a_gpu   = a.to(device)
print(f"a_gpu device: {a_gpu.device}")

a_back  = a_gpu.cpu()
a_numpy = a_back.numpy()
print(f"numpy type: {type(a_numpy)}")
print(f"numpy values:\n{a_numpy}")
```

Add a comment: why does PyTorch require `.cpu()` before you can call `.numpy()`? What does this tell you about where NumPy arrays live?

### Tensor Question 4

Shape manipulation appears constantly when preparing images for neural networks. Starting from:

```python
t = torch.arange(24).float()
```

1. Reshape to `(4, 6)`. Print the shape.
2. Reshape to `(2, 3, 4)`. Print the shape.
3. Take the result from step 1 and add a new dimension at position 0 using `.unsqueeze(0)`. Print the new shape.

Add a comment: a single image tensor typically has shape `(channels, height, width)`. Neural networks expect batches with shape `(batch_size, channels, height, width)`. What does `.unsqueeze(0)` accomplish when you are processing one image at a time?

### Tensor Question 5

Compare PyTorch and NumPy for a small matrix multiplication:

```python
np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
np_b = np.array([[5.0, 6.0], [7.0, 8.0]])

t_a  = torch.tensor(np_a, dtype=torch.float32)
t_b  = torch.tensor(np_b, dtype=torch.float32)
```

1. Compute `np_a @ np_b` and print the result.
2. Compute `t_a @ t_b` and print the result.
3. Confirm the outputs match.

Add a comment: at a high level, what role does matrix multiplication play as data passes through a single layer of a neural network?

## Pretrained Models

Training a CNN from scratch requires millions of labeled images and days of GPU time. Pretrained models skip all of that — TorchVision ships with CNNs that were already trained on ImageNet, a dataset of over one million images across 1,000 categories. These models have learned to recognize edges, textures, shapes, and high-level objects. Loading one takes two lines of code. In practice, pretrained models are the default starting point for almost every real-world computer vision project.

### Model Question 1

Load ResNet18 with pretrained weights and count its parameters:

```python
weights = ResNet18_Weights.DEFAULT
model   = models.resnet18(weights=weights)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

Add a comment: ResNet18 has roughly 11 million parameters. Training it from scratch required approximately 1.2 million labeled ImageNet images and days of multi-GPU compute. What does that tell you about the practical value of starting from pretrained weights when you're on a deadline or a budget?

### Model Question 2

Print the full model architecture by running `print(model)`. Read through the output — it shows every layer and its configuration.

Then answer the following in comments:

1. What is the name of the final layer in ResNet18, and what is its output size? (This number is the total count of ImageNet categories the model can predict.)
2. Can you identify the blocks named `layer1` through `layer4`? These are the "deep" part of the network — the feature extractor. In plain terms, what does it mean for a network to be "deep"?

### Model Question 3

Before running inference, move the model to the GPU and set it to evaluation mode:

```python
model = model.to(device)
model.eval()
print("Model ready for inference.")
```

Add a comment explaining each line:

1. What does `.to(device)` do, and why does it need to match the device your input tensors will be on?
2. What does `model.eval()` change about the model's behavior? Name at least one layer type that behaves differently in training mode vs. evaluation mode.

### Model Question 4

TorchVision model weights include the exact preprocessing pipeline the model was trained with. Use it directly:

```python
preprocess = weights.transforms()
print(preprocess)
```

This prints the full transform chain. Add a comment describing in plain English what each step does and why it matters. Address:

1. What does the resize/crop step accomplish?
2. What does `ToTensor()` do to the pixel value range?
3. What is normalization doing, and why does it use ImageNet's specific mean and standard deviation values rather than, say, `mean=0.5, std=0.5`?

## Running Inference

With a model loaded and preprocessing defined, running inference on a new image takes about five lines of code. This section walks you through each step using real images from the Intel Image Classification dataset.

Add this image-loading helper to your notebook. It picks a random image from a given scene class:

```python
import random
random.seed(42)

DATA_DIR = Path("/kaggle/input/intel-image-classification/seg_test/seg_test")
LABELS   = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def load_sample_image(label):
    """Load a random image file from the given class folder."""
    class_dir = DATA_DIR / label
    img_path  = random.choice(list(class_dir.glob("*.jpg")))
    return Image.open(img_path).convert("RGB"), img_path.name
```

Get the ImageNet class labels from the weights metadata — no separate download needed:

```python
imagenet_classes = weights.meta["categories"]
print(f"Number of classes: {len(imagenet_classes)}")
print(f"First 5 labels: {imagenet_classes[:5]}")
```

### Inference Question 1

Write a function that runs inference on a single PIL image and returns the top-5 predicted class names and their probabilities:

```python
def get_top5_predictions(model, preprocess, image, device, class_labels):
    """
    Run inference on a PIL image and return the top-5 predictions.
    Returns a list of (class_name, probability) tuples.
    """
    # Step 1: Preprocess the image and add a batch dimension
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Step 2: Run inference — torch.no_grad() disables gradient tracking
    #         since we are not training; this saves memory and speeds things up
    with torch.no_grad():
        output = model(input_tensor)        # shape: (1, 1000) — one score per class

    # Step 3: Convert raw scores (logits) to probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Step 4: Get the top 5 predictions
    top5_probs, top5_indices = torch.topk(probabilities, 5)

    results = []
    for prob, idx in zip(top5_probs, top5_indices):
        results.append((class_labels[idx.item()], prob.item()))
    return results
```

Test it on one mountain image:

```python
img, img_name = load_sample_image("mountain")
preds         = get_top5_predictions(model, preprocess, img, device, imagenet_classes)

print(f"\nTop-5 predictions for '{img_name}':")
for class_name, prob in preds:
    print(f"  {class_name:30s}  {prob:.4f}")
```

Add a comment: does the top prediction make sense? Remember that the model was trained on ImageNet's 1,000 categories, which include things like `"alp"`, `"valley"`, and `"lakeside"` rather than simply `"mountain"`. Do any of the top-5 labels map onto what you'd describe as a mountain scene?

### Inference Question 2

Run inference on one image from each of the six scene classes. For each, print the top-3 predictions:

```python
for label in LABELS:
    img, img_name = load_sample_image(label)
    preds = get_top5_predictions(model, preprocess, img, device, imagenet_classes)[:3]
    print(f"\n[{label}]  {img_name}")
    for class_name, prob in preds:
        print(f"  {class_name:30s}  {prob:.4f}")
```

Add a comment: which classes does the model seem most confident about (high top-1 probability)? Which does it seem least confident about? Is there a pattern?

### Inference Question 3

The raw output of the model before `softmax` is called *logits* — unconstrained scores that can be any real number. After `softmax` they become probabilities that sum to 1. Observe the difference:

```python
img, _ = load_sample_image("forest")
input_tensor = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(input_tensor)

probs = torch.nn.functional.softmax(logits[0], dim=0)

print(f"Logit  range: min={logits.min():.2f}, max={logits.max():.2f}")
print(f"Prob   range: min={probs.min():.6f}, max={probs.max():.4f}")
print(f"Probs sum to: {probs.sum():.6f}")
print(f"Top prediction: {imagenet_classes[probs.argmax().item()]}  ({probs.max():.4f})")
```

Add a comment: why do neural networks output logits internally rather than probabilities? In a production pipeline that needs to filter out low-confidence predictions, which representation would you work with — logits or probabilities — and why?

### Inference Question 4

Create a visualization that shows an image alongside a horizontal bar chart of its top-5 predictions. Save it to `outputs/`:

```python
def visualize_predictions(img, predictions, title="", save_path=None):
    """Display an image alongside its top-5 predictions as a bar chart."""
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 4))

    ax_img.imshow(img)
    ax_img.set_title(title)
    ax_img.axis("off")

    class_names = [p[0] for p in predictions]
    probs       = [p[1] for p in predictions]

    ax_bar.barh(range(len(class_names)), probs, color="steelblue")
    ax_bar.set_yticks(range(len(class_names)))
    ax_bar.set_yticklabels(class_names)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Probability")
    ax_bar.set_xlim(0, 1)
    ax_bar.set_title("Top-5 Predictions (ResNet18)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

img, img_name = load_sample_image("sea")
preds = get_top5_predictions(model, preprocess, img, device, imagenet_classes)
visualize_predictions(img, preds, title=f"sea / {img_name}",
                      save_path="outputs/warmup_inference_viz.png")
```

Add a comment: how would you adapt this kind of visualization for a dashboard that a non-technical team member needs to review flagged predictions? What threshold on the top-1 probability might you use to decide when a prediction is "confident enough" to act on?

---

# Part 2: Mini-Project — The Model Lineup: Putting Pretrained CNNs to Work

In most real-world situations you will not be asked which model is theoretically best. You will be asked: *"Given our latency budget, our GPU constraints, and the kind of images we're dealing with — which model should we actually deploy?"* Answering that question requires running the candidates side by side and comparing them on the same data.

This project gives you a hands-on version of that process. You will run three pretrained CNN architectures on the same set of images, compare their predictions, measure their inference speed, explore what the pretrained features actually capture, and write a brief production recommendation — the kind of summary a data engineer might hand to a team lead when evaluating model options.

Put all code in a new Kaggle notebook: `project_04.ipynb`. Add the Intel Image Classification dataset to this notebook as well (same steps as the warmup). Enable GPU before you begin.

## Task 1: Environment Setup and Data Loading

Add this setup block at the top of your project notebook:

```python
import torch
import torchvision
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    MobileNet_V3_Small_Weights,
    EfficientNet_B0_Weights,
)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import copy
import os
from pathlib import Path
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATA_DIR = Path("/kaggle/input/intel-image-classification/seg_test/seg_test")
LABELS   = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

random.seed(42)
```

Load a fixed sample of images from the test split. You'll use this same sample throughout the project so that every model sees the same inputs:

```python
def load_images(n_per_class=10):
    """Load n images per class. Returns a list of (PIL.Image, label_string) tuples."""
    image_set = []
    for label in LABELS:
        class_dir = DATA_DIR / label
        paths = random.sample(list(class_dir.glob("*.jpg")), n_per_class)
        for path in paths:
            img = Image.open(path).convert("RGB")
            image_set.append((img, label))
    random.shuffle(image_set)
    return image_set

image_set = load_images(n_per_class=10)
print(f"Total images loaded: {len(image_set)}")
```

Display a 2×3 grid showing one sample image per class, labeled with the ground-truth class name. Save to `outputs/dataset_sample.png`.

Add a comment: the six scene types in this dataset are buildings, forest, glacier, mountain, sea, and street. The model you'll be using was trained on ImageNet, whose 1,000 classes include labels like `"alp"`, `"lakeside"`, `"valley"`, and `"barn"` — not a direct match to these six categories. Does that mean a pretrained ImageNet model is a poor fit for this data, or a reasonable starting point? Why?

## Task 2: Baseline Inference with ResNet18

Before comparing models, make sure you understand what a single model is doing end to end.

Load ResNet18 and prepare it for inference:

```python
resnet_weights  = ResNet18_Weights.DEFAULT
resnet          = models.resnet18(weights=resnet_weights).to(device).eval()
resnet_preproc  = resnet_weights.transforms()
imagenet_classes = resnet_weights.meta["categories"]

print(f"ResNet18 parameters: {sum(p.numel() for p in resnet.parameters()):,}")
```

Write a general-purpose inference function that you will reuse for all three models:

```python
def run_inference(model, preprocess, image, device, class_labels, top_k=5):
    """
    Run inference on a PIL image.
    Returns a list of (class_name, probability) tuples for the top k predictions.
    """
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_indices = torch.topk(probs, top_k)
    return [(class_labels[idx.item()], prob.item())
            for prob, idx in zip(top_probs, top_indices)]
```

Run inference on every image in `image_set` and store the results. Use a list of dictionaries:

```python
resnet_results = []
for img, true_label in image_set:
    preds = run_inference(resnet, resnet_preproc, img, device, imagenet_classes)
    resnet_results.append({
        "true_label":   true_label,
        "top1_class":   preds[0][0],
        "top1_prob":    preds[0][1],
        "top5_classes": [p[0] for p in preds],
        "top5_probs":   [p[1] for p in preds],
    })

print(f"Processed {len(resnet_results)} images.")
```

Compute and print:

- Overall mean top-1 probability across all images
- Mean top-1 probability broken down by true class (which classes does the model feel most and least confident about?)

Create a boxplot showing the distribution of top-1 probabilities across the six classes. Label each box with the class name, add a title, and save to `outputs/resnet18_confidence_by_class.png`.

Add a comment: high confidence and high accuracy are not the same thing. A model can be confidently wrong. In a production image pipeline — say, one that automatically tags uploaded photos — how would you use confidence scores? What threshold might trigger a "send to human reviewer" flag?

## Task 3: Multi-Model Comparison

Load the two additional pretrained models. Each model has its own preprocessing pipeline — this is easy to get wrong, and using the wrong transforms will silently produce bad predictions:

```python
# MobileNetV3-Small — designed for mobile and edge deployment
mobile_weights = MobileNet_V3_Small_Weights.DEFAULT
mobilenet      = models.mobilenet_v3_small(weights=mobile_weights).to(device).eval()
mobile_preproc = mobile_weights.transforms()

# EfficientNet-B0 — designed to maximize accuracy per unit of compute
effnet_weights = EfficientNet_B0_Weights.DEFAULT
efficientnet   = models.efficientnet_b0(weights=effnet_weights).to(device).eval()
effnet_preproc = effnet_weights.transforms()

# Print parameter counts for all three
for name, m in [("ResNet18",          resnet),
                ("MobileNetV3-Small", mobilenet),
                ("EfficientNet-B0",   efficientnet)]:
    params = sum(p.numel() for p in m.parameters())
    print(f"{name:22s}  {params:>12,} parameters")
```

Add a comment after reading the parameter counts: what does a smaller parameter count imply about a model's capacity? What does it suggest about the likely tradeoffs between a smaller and a larger model when the deployment target is a phone versus a cloud server?

Run inference on the full `image_set` with MobileNet and EfficientNet, using the same `run_inference` function and the same dictionary structure as Task 2. Store results in `mobilenet_results` and `effnet_results`.

Build a comparison grid for 6 images — one from each class. For each image, display:

- The image (left column)
- Top-3 predictions from ResNet18, MobileNetV3, and EfficientNet (right columns)

```python
# Suggested layout: pick one representative image per class.
# Rows = one image per class, columns = [Image, ResNet18, MobileNet, EfficientNet]
# Use plt.subplots and annotate each column header clearly.
```

Save the grid to `outputs/model_comparison_grid.png`.

After building the grid, add a comment addressing:

1. Do the three models generally agree on their top-1 prediction?
2. Are there cases where they disagree significantly? What might that tell you about whether combining model predictions (an ensemble) could help?
3. For this particular dataset — outdoor scenes — which model's top-5 predictions feel most semantically sensible, even when the ImageNet label isn't an exact match?

## Task 4: Speed vs. Accuracy Tradeoff

Latency is one of the most important practical constraints when deploying a model. It determines whether a model is viable for real-time applications, batch pipelines with throughput requirements, or edge devices with limited compute.

Benchmark all three models using the function below. Note the `torch.cuda.synchronize()` calls — without them, timing on a GPU is unreliable because GPU operations are asynchronous and may not have completed by the time you stop the clock:

```python
def benchmark_model(model, preprocess, image_set, device, n_warmup=5):
    """
    Benchmark single-image inference speed.
    Returns mean latency in milliseconds per image.
    """
    # Warm up the GPU — the first few calls are slower due to CUDA initialization
    for img, _ in image_set[:n_warmup]:
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(tensor)

    # Timed run — synchronize before and after to get accurate GPU timing
    torch.cuda.synchronize()
    start = time.time()

    for img, _ in image_set:
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(tensor)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    return (elapsed / len(image_set)) * 1000  # milliseconds per image

resnet_ms  = benchmark_model(resnet,       resnet_preproc,  image_set, device)
mobile_ms  = benchmark_model(mobilenet,    mobile_preproc,  image_set, device)
effnet_ms  = benchmark_model(efficientnet, effnet_preproc,  image_set, device)

print(f"ResNet18:           {resnet_ms:.2f} ms/image")
print(f"MobileNetV3-Small:  {mobile_ms:.2f} ms/image")
print(f"EfficientNet-B0:    {effnet_ms:.2f} ms/image")
```

Create a bar chart showing inference latency (ms/image) for all three models. Add a title, label the axes, and save to `outputs/inference_speed.png`.

Then build a summary table — you can use formatted `print()` statements — comparing each model on three dimensions:

| Model | Parameters | ms / image |
|---|---|---|
| ResNet18 | ? | ? |
| MobileNetV3-Small | ? | ? |
| EfficientNet-B0 | ? | ? |

Add a comment addressing this scenario: your team needs to classify images in near-real-time at a rate of 50 images per second. What is the maximum tolerable latency per image in milliseconds? Based on your results, which models can meet that bar?

Then add a second comment: which model would you choose if the deployment target is (a) a high-throughput cloud pipeline, (b) an on-device mobile app, (c) a safety-critical quality-control system where getting it right matters more than speed? Briefly justify each choice.

## Task 5: Pretrained Features as a Window into Transfer Learning

One of the key ideas from the transfer learning lesson is that the early layers of a pretrained CNN learn general visual features — edges, textures, patterns — that are useful across many tasks. The final layer is the only part that is specific to ImageNet's 1,000 categories.

This means you can use a pretrained CNN as a *feature extractor*: remove the final classification layer, run images through the rest of the network, and get a dense vector (an embedding) that represents the visual content of the image. This is entirely inference-based — you are not updating any weights.

Here is how to do it with ResNet18. We replace the final fully connected layer with an identity operation, which passes the input through unchanged:

```python
import copy

feature_extractor = copy.deepcopy(resnet)
feature_extractor.fc = torch.nn.Identity()   # remove the classification head
feature_extractor    = feature_extractor.to(device).eval()

def extract_features(model, preprocess, image, device):
    """Extract a feature vector from an image using the truncated CNN."""
    tensor   = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze().cpu().numpy()

# Extract features for all images
feature_vectors = []
true_labels     = []

for img, label in image_set:
    feat = extract_features(feature_extractor, resnet_preproc, img, device)
    feature_vectors.append(feat)
    true_labels.append(label)

feature_matrix = np.array(feature_vectors)
print(f"Feature matrix shape: {feature_matrix.shape}")
# Expected: (60, 512) — 60 images, 512-dimensional feature vector each
```

Reduce the 512-dimensional feature vectors to 2D using PCA and plot them, colored by true class label. The `PCA` import is already in your setup block:

```python
pca          = PCA(n_components=2)
features_2d  = pca.fit_transform(feature_matrix)

fig, ax = plt.subplots(figsize=(8, 6))
colors  = plt.cm.tab10(np.linspace(0, 1, len(LABELS)))

for i, label in enumerate(LABELS):
    mask = [l == label for l in true_labels]
    ax.scatter(
        features_2d[mask, 0],
        features_2d[mask, 1],
        label=label, color=colors[i], s=60, alpha=0.75
    )

ax.legend()
ax.set_title("ResNet18 Feature Embeddings (PCA to 2D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig("outputs/feature_embeddings.png")
plt.show()
```

Add a comment addressing:

1. Do images from the same class tend to cluster together in the 2D feature space? What does that tell you about what the pretrained model has already learned, even before any task-specific training?
2. The transfer learning lesson described two strategies: *feature extraction* (freeze all pretrained layers, train only a new final layer) and *fine-tuning* (allow some or all pretrained weights to update during training). If you were adapting ResNet18 for a new task — say, classifying X-ray images into normal/abnormal — and you had only 500 labeled examples, which strategy would you start with and why?

### Stretch Goal: Fine-Tuning the Classification Head (Optional)

*Complete Tasks 1–6 before attempting this. This section is optional and worth 3 bonus points on the rubric.*

So far you have used ResNet18 exactly as it was trained: outputting scores for 1,000 ImageNet categories. The transfer learning lesson described a more powerful adaptation: replace the final layer with one that predicts *your specific classes*, then train just that layer while keeping everything else frozen. The pretrained visual features stay intact; only a small new head learns.

This is called *feature extraction* transfer learning. Because only the final layer has trainable weights, it converges in a few epochs even on a small dataset — you will see meaningful results in under a minute on Kaggle's GPU.

```python
# --- Stretch Goal: Fine-Tuning the Classification Head ---

import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

NUM_CLASSES = len(LABELS)
TRAIN_DIR   = Path("/kaggle/input/intel-image-classification/seg_train/seg_train")

# Step 1: Build the fine-tuning model
#   Start from the trained ResNet18, freeze all layers,
#   then replace only the final fc layer with one that outputs 6 classes.

ft_model = copy.deepcopy(resnet)

for param in ft_model.parameters():       # freeze everything
    param.requires_grad = False

ft_model.fc = nn.Linear(ft_model.fc.in_features, NUM_CLASSES)  # new trainable head
ft_model    = ft_model.to(device)

trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in ft_model.parameters())
print(f"Trainable: {trainable:,} of {total:,} total parameters ({100*trainable/total:.2f}%)")
```

```python
# Step 2: Load a small, balanced training set — 50 images per class

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),       # simple augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_train = ImageFolder(TRAIN_DIR, transform=train_transforms)
print(f"Classes (alphabetical): {full_train.classes}")

# Sample 50 images per class for a balanced mini-training set
random.seed(42)
imgs_per_class    = 50
balanced_indices  = []
for class_idx in range(NUM_CLASSES):
    indices = [i for i, (_, lbl) in enumerate(full_train.samples) if lbl == class_idx]
    balanced_indices.extend(random.sample(indices, min(imgs_per_class, len(indices))))

train_subset = Subset(full_train, balanced_indices)
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

print(f"Training on {len(train_subset)} images across {NUM_CLASSES} classes")
```

```python
# Step 3: Fine-tune for 3 epochs
#   Only ft_model.fc has requires_grad=True, so the optimizer only updates that layer.

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ft_model.fc.parameters(), lr=1e-3)

for epoch in range(3):
    ft_model.train()
    running_loss = 0.0
    correct      = 0
    total_seen   = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = ft_model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct      += (outputs.argmax(dim=1) == labels).sum().item()
        total_seen   += images.size(0)

    print(f"Epoch {epoch+1}/3 — loss: {running_loss/total_seen:.4f},"
          f" train acc: {correct/total_seen:.3f}")
```

```python
# Step 4: Compare original vs. fine-tuned predictions on 3 test images

ft_model.eval()
idx_to_label = full_train.classes      # alphabetical class names

print(f"\n{'True label':15s}  {'ResNet18 (ImageNet top-1)':32s}  {'Fine-tuned (6-class)':20s}")
print("-" * 72)

for test_label in ["forest", "sea", "buildings"]:
    img, _ = load_sample_image(test_label)

    # Original ResNet18 — outputs an ImageNet class name
    original_top1 = run_inference(resnet, resnet_preproc, img, device, imagenet_classes)[0][0]

    # Fine-tuned model — outputs one of our 6 scene classes
    tensor = resnet_preproc(img).unsqueeze(0).to(device)
    with torch.no_grad():
        ft_out = ft_model(tensor)
    ft_prediction = idx_to_label[ft_out.argmax(dim=1).item()]

    print(f"{test_label:15s}  {original_top1:32s}  {ft_prediction:20s}")
```

Add a comment addressing:

1. What fraction of ResNet18's total parameters were actually updated during fine-tuning? What does that tell you about where the learned "knowledge" in the network lives?
2. Did the fine-tuned model predict the correct scene category? Given only 300 training images and 3 epochs, what would be the next step if you wanted better results?
3. The original ResNet18 outputs labels like `"alp"` or `"lakeside"`. The fine-tuned model outputs `"mountain"` or `"sea"`. Which output format is more useful in a real application, and what does this illustrate about the practical value of fine-tuning even a single layer?

## Task 6: Summary and Recommendation

Write a brief summary in a markdown cell or as a comment block addressing the following. Clear technical bullets are fine — you do not need polished prose.

**Model Comparison**: Based on Tasks 3 and 4, which of the three models performed best for this dataset? Reference specific observations: prediction quality, confidence scores, and speed results.

**Confidence Calibration**: Looking at the boxplot from Task 2, which scene types was ResNet18 most and least confident about? Does that match your intuition about which scenes are visually distinctive?

**Production Recommendation**: Your team needs to classify user-uploaded outdoor photos into six scene categories (buildings, forest, glacier, mountain, sea, street) as part of a data pipeline. Write a 3–5 sentence recommendation covering:

- Which model you would suggest starting with, and why
- What preprocessing steps the pipeline would need to include
- One limitation or risk you would flag before the team ships it

The goal here is not to give the "right" answer — it is to practice translating what you observed in code into a decision with reasoning behind it.
