# Lesson resources
Resources such as essential images, data (CSV/JSON files), etc., for Python 200. Each module for Python 200 has its own resources sub-directory:

    analysis/
    ml/
    ai/
    cloud/

## Guidelines for Contributors
- Please put resources in their appropriate module's directory.
- Keep file sizes small (under 1MB); large datasets should live outside the repo to be downloaded in individual lessons.
- Use jpg for images to keep files small.  
- Use descriptive filenames, e.g. `histogram_example.jpg` or `mnist_sample.csv`.
- Reference resources in lessons with relative paths, e.g.:

  ```markdown
  ![Histogram Example](resources/analysis/histogram_example.jpg)
