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

| Cookbook                                       |
| ---------------------------------------------- |
| [langsmith.ipynb](./cookbooks/langsmith.ipynb) |
| [langfuse.ipynb](./cookbooks/langfuse.ipynb)   |
| [parea.ipynb](./cookbooks/parea.ipynb)         |
| [lunary.ipynb](./cookbooks/lunary.ipynb)       |

## Repo setup

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
rye test -- --helpers # integration tests with helpers
```
