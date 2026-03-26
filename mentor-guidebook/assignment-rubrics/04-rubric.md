# Assignment 4: Deep Learning

**Part 1: Warmup Exercises — 40 points**

| Section | Points | What reviewers look for |
|---|---|---|
| Tensors Q1–Q5 | 15 | All five questions complete with correct printed output; comments explain reasoning, not just what the code does |
| Pretrained Models Q1–Q4 | 13 | Model loaded with modern weights API; `.eval()` and `.to(device)` used correctly; comments show understanding of why each step matters |
| Running Inference Q1–Q4 | 12 | `get_top5_predictions` function works; softmax applied correctly; visualization saved with title and axis labels; comments engage with the predictions rather than just restating the code |

**Part 2: Mini-Project — 50 points**

| Task | Points | What reviewers look for |
|---|---|---|
| Task 1: Setup and Data Loading | 5 | Data loads without errors; sample grid saved; comment addresses the ImageNet-vs-Intel class mismatch question |
| Task 2: Baseline Inference | 10 | Results stored in the correct structure; confidence boxplot saved with labels; comment addresses the confidence-vs-accuracy distinction |
| Task 3: Multi-Model Comparison | 13 | All three models loaded with their own transforms; comparison grid saved and readable; comment engages with agreement/disagreement observations |
| Task 4: Benchmarking | 10 | `torch.cuda.synchronize()` used correctly; speed chart saved; both comment prompts addressed with reference to actual results |
| Task 5: Feature Extraction | 7 (+3 bonus) | Core: feature extractor constructed using `nn.Identity()`; PCA scatter plot saved with legend; comment addresses clustering and the feature extraction vs. fine-tuning tradeoff. **Stretch goal bonus**: fine-tuning loop runs without errors; comparison table printed; comment addresses all three stretch goal questions |
| Task 6: Summary | 5 | All three prompts addressed; recommendation references specific observations from earlier tasks rather than speaking in generalities |

**Code Quality — 10 points**

Comments explain *why*, not just *what*. Output is readable — no raw unformatted tensor dumps. Each saved figure has a descriptive title and labeled axes. Notebooks run top to bottom without errors.
