# Zenbase Python SDK

## Installation

<a href="https://pypi.python.org/pypi/zenbase"><img src="https://img.shields.io/pypi/v/zenbase.svg?style=flat-square&label=pypi+zenbase" alt="zenbase Python package on PyPi"></a>

Zenbase requires Python ≥3.10. You can install it using your favorite package manager:

```bash
pip install zenbase
poetry add zenbase
rye add zenbase
```

## Usage

Zenbase is designed to require minimal changes to your existing codebase and integrate seamlessly with your existing eval/observability platforms. It works with any AI SDK (OpenAI, Anthropic, Cohere, Langchain, etc.).


### Labeled Few-Shot Learning Cookbooks:

LabeledFewShot will be useful for tasks that are just one layer of prompts.

| Cookbook                                                      | Run in Colab                                                                                                                                                                                             |
|---------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [langsmith.ipynb](cookbooks/labeled_few_shot/langsmith.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/zenbase-ai/lib/blob/main/py/cookbooks/labeled_few_shot/langsmith.ipynb) |
| [arize.ipynb](cookbooks/labeled_few_shot/arize.ipynb)         | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/zenbase-ai/lib/blob/main/py/cookbooks/labeled_few_shot/arize.ipynb)     |
| [langfuse.ipynb](cookbooks/labeled_few_shot/langfuse.ipynb)   | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/zenbase-ai/lib/blob/main/py/cookbooks/labeled_few_shot/langfuse.ipynb)  |
| [parea.ipynb](cookbooks/labeled_few_shot/parea.ipynb)         | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/zenbase-ai/lib/blob/main/py/cookbooks/labeled_few_shot/parea.ipynb)     |
| [lunary.ipynb](cookbooks/labeled_few_shot/lunary.ipynb)       | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/zenbase-ai/lib/blob/main/py/cookbooks/labeled_few_shot/lunary.ipynb)    |

### Bootstrap Few-Shot Learning Cookbooks:

BootstrapFewShot will be useful for tasks that are multiple layers of prompts.

| Cookbook                                                        | Run in Colab                                                                                                                                                                                               |
|-----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [langsmith.ipynb](cookbooks/bootstrap_few_shot/langsmith.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/zenbase-ai/lib/blob/main/py/cookbooks/bootstrap_few_shot/langsmith.ipynb) |
| [arize.ipynb](cookbooks/bootstrap_few_shot/arize.ipynb)         | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/zenbase-ai/lib/blob/main/py/cookbooks/bootstrap_few_shot/arize.ipynb)     |
| [langfuse.ipynb](cookbooks/bootstrap_few_shot/langfuse.ipynb)   | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/zenbase-ai/lib/blob/main/py/cookbooks/bootstrap_few_shot/langfuse.ipynb)  |
| [parea.ipynb](cookbooks/bootstrap_few_shot/parea.ipynb)         | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/zenbase-ai/lib/blob/main/py/cookbooks/bootstrap_few_shot/parea.ipynb)     |
| [lunary.ipynb](cookbooks/bootstrap_few_shot/lunary.ipynb)       | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/zenbase-ai/lib/blob/main/py/cookbooks/bootstrap_few_shot/lunary.ipynb)    |

## Development setup

This repo uses Python 3.10 and [rye](https://rye.astral.sh/) to manage dependencies. Once you've gotten rye installed, you can install dependencies by running:

```bash
rye sync
```

And activate the virtualenv with:

```bash
. .venv/bin/activate
```

You can run tests with:

```bash
rye test # pytest -sv to see prints and verbose output
rye test -- -m helpers # integration tests with helpers
```
