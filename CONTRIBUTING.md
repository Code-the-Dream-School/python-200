# Contributing to Python 200
Thanks for your interest in helping to develop Python 200! We're excited to have you on board! :tada: If you ever need help with any of the steps here, please reach out and ask for help! Contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to contribute
GitHub contributions will mainly come through PRs. Before making a PR, you need to first set up a local version of the repo. There, you can work on feature branches, such as individual lesson plans, that will get merged into `main`. 

### Fork and clone repo
Firs, set things up so you have a local copy of the repo based on a fork of the CTD repo:
1. Fork the Python 200 repo at GitHub
2. Clone your fork locally 

    git clone https://github.com/YOUR-USERNAME/python-200.git    
    cd python-200

3. Set the upstream remote     
    git remote add upstream https://github.com/Code-the-Dream/python-200.git

### Work locally and make PRs
Once you are set up with the repo, you can start your work by making a local feature branch (let's call it `week1_module2`). 

> Before creating a branch, be sure your local `main` is up to date with the upstream `main` (for this you can do `git pull upstream main`).

1. Create and switch to feature branch

    git checkout -b week1_module2

2. Make your edits and commit
Once you are in your branch, make the edits you want, and make your commits. Please be sure to use clear and descriptive commit messages.

    git commit -m "Add explanation of p-value for week1 lesson"

3. Push branch to your fork 

    git push origin week1_module2

4. Open a pull request
At your fork on GitHub, click on "Compare & Pull Request" and open a PR to merge your branch into the `main` branch of the upstream repo. Any additional commits you push to that same branch will automatically be added to the open PR. 

The PR will go through review before being merged into `main`. Thank you for your efforts! 

## Pull request guidelines
- Please keep PRs focused and not too large (avoid bundling unrelated changes). 
- Do not commit sensitive data (such as API keys or passwords).
- Add a short summary of what you are changing in the PR description. 
- Use `snake_case` for file and folder names
  - Exception: capitalizing abbreviations (`AI`) is ok for folders if it helps readability
- If relevant, please mention related issues in the PR by mentioning the `#issue-number`. 
- Each lesson directory contains a `resources` folder to place essential images, data (CSV/JSON files), etc. for the lesson. A few guidelines for resources: 
  - Keep file sizes small (under 1MB); large datasets should live outside the repo to be downloaded in individual lessons.
  - Use jpg for images to keep files small.
  - Use descriptive filenames, e.g. `histogram_example.jpg` or `sales_data_sample.csv`.
  - Reference resources with relative paths. For example:

    ```markdown
    ![Histogram Example](resources/histogram_example.jpg)


Again, thank you for helping with Python 200! :heart:

As mentioned above, please reach out if you need help getting started, or get stuck with any of the above steps. We are happy to provide support.

