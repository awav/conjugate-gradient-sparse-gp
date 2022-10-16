# Numerically Stable Sparse Gaussian Processes via Minimum Separation using Cover Trees

## Install develop version


```python
> pip install -r requirements.txt
> pip install -e .
```

To use UCI datasets you would also need to follow the installation instructions of the [`bayesian_benchmarks`](https://github.com/hughsalimbeni/bayesian_benchmarks) package.

## Usage

The experiments script used for generating tables and plots are prefixed with `paper_*`:

### `paper_cli_uci.py`

Command line interface for prediction using SGPR and CDGP models with loaded hyperparameters.

```
Usage: paper_cli_uci.py [OPTIONS] COMMAND [ARGS]...

  This is a core command for all CLI functions.

Options:
  -mc, --model-class [sgpr|cdgp]  [required]
  -p, --precision DTYPE           [required]
  -j, --jitter FLOAT              [required]
  -c, --config-dir PATH
  --jit / --no-jit
  --help                          Show this message and exit.

Commands:
  covertree
  greedy
  kmeans
  kmeans2
  oips
  uniform
```