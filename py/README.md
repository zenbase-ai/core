# zenbase-ai/lib/py

## Getting started

This repo uses Python 3.12 and [rye](https://rye.astral.sh/) to manage dependencies. Once you've gotten rye installed, you can install dependencies by running:

```bash
rye sync
```

And activate the virtualenv with:

```bash
. .venv/bin/activate
```

## Running tests

```bash
rye test # pytest -sv to see prints and verbose output
```
