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

| Cookbook                                       | Run in Colab                                                                                                                                                          |
| ---------------------------------------------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [langsmith.ipynb](./cookbooks/langsmith.ipynb) | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/14Lh8Azk_Ocnm2GvGvFHFz_hJ1tFNvOJW?usp=sharing) |
| [langfuse.ipynb](./cookbooks/langfuse.ipynb)   | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1EMR_PQfsfawTvTjZSxcS_FQUr8s_gVwR?usp=sharing) |
| [parea.ipynb](./cookbooks/parea.ipynb)         | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1oBFD8MhHjHxCltuosFeXHkqOsdeXwJ_N?usp=sharing)   |
| [lunary.ipynb](./cookbooks/lunary.ipynb)       | [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1HXpW6E7AgoBbVWiiUxxtzztU6Gxy6iEA?usp=sharing) |

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
rye test -- -m helpers # integration tests with helpers
```
