# Contributing to MAPIE Documentation

This guide explains how to add, edit, and preview MAPIE's documentation built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

## Local Preview

Start the development server:

```bash
mkdocs serve -a 127.0.0.1:8787
```

Open [http://127.0.0.1:8787](http://127.0.0.1:8787) in your browser. Pages reload automatically when you save changes.

## Project Structure

```
mkdocs.yml                  # Main configuration
doc/
├── index.md                # Homepage
├── getting-started/        # Getting started guides
├── theory/                 # Theoretical descriptions
├── calibration/            # Calibration section
├── api/                    # API reference (auto-generated from docstrings)
├── images/                 # Images used in documentation
├── stylesheets/extra.css   # Custom CSS overrides
├── javascripts/mathjax.js  # MathJax configuration
└── hooks/                  # MkDocs build hooks
examples/
├── regression/             # Gallery example scripts
├── classification/
├── calibration/
├── risk_control/
└── mondrian/
```

## Adding a New Documentation Page

### 1. Create the Markdown File

Create a `.md` file in the appropriate directory under `doc/`. For example, to add a new theory page:

```bash
doc/theory/my-new-topic.md
```

Write your content using standard Markdown:

```markdown
# My New Topic

Introduction text here.

## Section Title

Content with **bold**, `code`, and math: $\alpha = 0.1$.

$$
\hat{C}_n(X_{n+1}) = \hat{\mu}(X_{n+1}) \pm \hat{q}_{1-\alpha}\{|Y_i - \hat{\mu}(X_i)|\}
$$

!!! note
    Use admonitions for important information.
```

### 2. Register the Page in Navigation

Edit `mkdocs.yml` and add your page to the `nav` section:

```yaml
nav:
  - Conformal Prediction:
    - Regression:
      - Theoretical Description: theory/regression.md
      - My New Topic: theory/my-new-topic.md   # ← add here
```

### 3. Add Images

Place images in `doc/images/` and reference them:

```markdown
![Description](images/my-image.png)
```

## Adding a New Gallery Example

Gallery examples are Python scripts in `examples/` that are automatically rendered as interactive documentation pages by the [mkdocs-gallery](https://smarie.github.io/mkdocs-gallery/) plugin.

### 1. Create the Example Script

Create a `.py` file in the appropriate subdirectory. The filename **must** start with `plot_`:

```
examples/regression/1-quickstart/plot_my_example.py
```

### 2. Script Structure

Gallery scripts use a specific format with docstrings and comment blocks:

```python
"""
# Example Title

Brief description of what this example demonstrates.
"""

##############################################################################
# Section heading as a comment block.
# This text will be rendered as Markdown.

import numpy as np
from mapie.regression import SplitConformalRegressor

##############################################################################
# Another explanatory section.
# You can use **Markdown** formatting here.
#
# - Bullet points work
# - Math works: $\alpha = 0.1$

# Code that generates a plot will be shown with its output
import matplotlib.pyplot as plt

plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("My Plot")
plt.show()
```

Key rules:

- **Title block**: The first triple-quoted docstring becomes the page title (use `# Markdown heading`).
- **Comment blocks**: Lines starting with `#` followed by `#` separators are rendered as Markdown text sections.
- **Code blocks**: Regular Python code is displayed as executable code with output.
- **Plots**: Any matplotlib figure created will be captured and displayed.

### 3. Add a Subsection Directory (if needed)

If you're creating a new subsection (e.g., `3-scientific-articles/`), add a `README.md` in the directory:

```
examples/regression/3-scientific-articles/README.md
```

```markdown
# Scientific Articles

Examples reproducing results from scientific papers.
```

### 4. Add a New Top-Level Gallery Section (if needed)

To add an entirely new gallery section (e.g., `examples/time_series/`):

1. Create the directory with subdirectories:
  ```
   examples/time_series/
   examples/time_series/README.md
   examples/time_series/1-quickstart/
   examples/time_series/1-quickstart/README.md
   examples/time_series/1-quickstart/plot_example.py
  ```
2. Register it in `mkdocs.yml` under the `gallery` plugin:
  ```yaml
   plugins:
     - gallery:
         examples_dirs:
           - examples/regression
           - examples/time_series       # ← add here
         gallery_dirs:
           - doc/generated/regression
           - doc/generated/time_series  # ← add here
  ```
3. Add a navigation entry in the relevant topic section and add the new
   gallery link to `doc/all-examples/index.md`:
  ```yaml
   nav:
     - Time Series:
       - Examples: generated/time_series  # ← add here
  ```

## Editing API Documentation

API pages are auto-generated from Python docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

### How It Works

Each API page in `doc/api/` contains directives like:

```markdown
::: mapie.regression.SplitConformalRegressor
```

This renders the class documentation directly from the source code docstrings.

### Adding a New Class to API Docs

1. Open the relevant API file (e.g., `doc/api/regression.md`).
2. Add a mkdocstrings directive:
  ```markdown
   ::: mapie.my_module.MyNewClass
  ```
3. For pages with subsections (like conformity-scores, metrics), use `heading_level: 3`:
  ```markdown
   ::: mapie.my_module.MyNewClass
       options:
         heading_level: 3
  ```

### Docstring Format

MAPIE uses **numpy-style** docstrings. Use single backticks for inline code:

```python
def my_method(self, X, alpha=0.1):
    """Compute prediction intervals.

    Parameters
    ----------
    X : ArrayLike of shape (n_samples, n_features)
        Input data.

    alpha : float
        Significance level. Defaults to `0.1`.

    Returns
    -------
    prediction_intervals : NDArray of shape (n_samples, 2)
        Lower and upper bounds of the prediction intervals.
    """
```

## Using Math Equations

MathJax is configured for rendering LaTeX math.

- **Inline**: `$\alpha$` renders as $\alpha$
- **Block**:
  ```markdown
  $$
  \hat{q}_{1-\alpha} = \text{Quantile}\left(1 - \alpha; \frac{1}{n} \sum_{i=1}^{n} \delta_{s_i}\right)
  $$
  ```

## Using Admonitions

MkDocs Material supports various admonition types:

```markdown
!!! note "Optional Title"
    Note content here.

!!! warning
    Warning content here.

!!! example
    Example content here.

!!! tip
    Tip content here.
```

## Building for Production

Build the static site:

```bash
mkdocs build
```

The output goes to `site/` (gitignored). The GitHub Actions workflow in `.github/workflows/deploy-docs.yml` handles deployment automatically on push to main.