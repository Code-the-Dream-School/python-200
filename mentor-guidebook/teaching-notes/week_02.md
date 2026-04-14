# Week 2: Introduction to Machine Learning

## Overview

Students got their first look at machine learning as a field — what it is, where it fits in the AI landscape, and why it matters. They then got hands-on with scikit-learn's API and applied it to their first real model: linear regression for predicting continuous values.

## Key Concepts

**The AI/ML/DL hierarchy** — AI is the broadest term (machines doing intelligent things). ML is a subset where machines learn from data rather than following explicit rules. Deep learning is a subset of ML using neural networks. Students should be able to explain where each fits.

**Supervised vs. unsupervised learning** — Supervised learning uses labeled examples (input + correct answer) to train a model. Unsupervised learning finds structure in data without labels (clustering, dimensionality reduction). This week focused on supervised learning.

**The scikit-learn API pattern** — Nearly everything in scikit-learn follows three steps: (1) create the model object, (2) `.fit()` it on training data, (3) `.predict()` on new data. Once students internalize this pattern, switching between algorithms is straightforward.

**Train/test split** — You evaluate a model on data it has never seen to get an honest estimate of performance. If you evaluate on training data, you'll be measuring how well the model memorized, not how well it generalizes.

**Regression metrics** — R² measures how much variance the model explains (1.0 is perfect, 0 means the model is no better than predicting the mean). RMSE and MAE measure average prediction error in the original units.

## Common Questions

- **"Why can't I just use all my data for training?"** — You could, but then you'd have no way to know if your model actually works on new data. It's like studying the answers to a specific test — you might ace that test but fail a different one.
- **"What's the difference between regression and classification?"** — Regression predicts a continuous number (price, temperature). Classification predicts a category (spam/not spam, dog/cat). This week is regression; next week is classification.
- **"What does a negative R² mean?"** — It means the model is literally worse than just predicting the average every time. This usually signals a bug or a completely wrong approach.

## Watch Out For

- **Data leakage** — Students sometimes forget to split *before* preprocessing (e.g., fitting a scaler on all the data, then splitting). This lets information from the test set influence training, which inflates performance estimates. This is covered in Week 3 but worth previewing if it comes up.
- **Overfitting intuition** — Students may not immediately grasp why a perfect fit on training data is bad. The lesson uses a visualization — walk through it if anyone is confused.
- **Conflating model fitting with model understanding** — scikit-learn makes it so easy to call `.fit()` that students may not think about *what* the model is actually doing. Encourage them to interpret the coefficients, not just the R² score.

## Suggested Activities

1. **API pattern quiz:** Without looking at the lesson, ask students to write (from memory) the three lines needed to train any scikit-learn model and make a prediction. This reinforces the core pattern.

2. **Metrics interpretation:** Give students a scenario: "Your model predicts house prices. R² = 0.45, RMSE = $85,000. Is this a good model?" There's no single right answer — discuss what "good" means in context and what baseline you'd compare to.

3. **Feature brainstorm:** Take the student performance dataset from the assignment. Ask: "Besides the features in the dataset, what other variables do you think would improve this model's predictions? What data would you need to collect?"
