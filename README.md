# Fast Convergence of Conjugate Gradients in Sparse Gaussian Processes

## Install develop version


```python
> pip install -r requirements.txt
> pip install -e .
```

To use UCI datasets you would also need to follow the installation instructions of the [`bayesian_benchmarks`](https://github.com/hughsalimbeni/bayesian_benchmarks) package.

## Usage

The experiments script used for generating tables and plots are prefixed with `paper_*`:

### `paper_cli.py`

Command line interface for training SGPR, LpSVGP and CDGP models.

```bash
❯ python paper_cli.py --help
Usage: paper_cli.py [OPTIONS] COMMAND [ARGS]...

  This is a core command for all CLI functions.

Options:
  -d, --dataset DATASET  [required]
  -p, --precision DTYPE
  -j, --jitter FLOAT
  -k, --kernel DATASET
  -l, --logdir PATH
  -s, --seed INTEGER
  --jit / --no-jit
  --help                 Show this message and exit.

Commands:
  train-adam-covertree
```

And usage of `train-adam-covertree` command:

```bash
❯ python paper_cli.py -d DATASET train-adam-covertree --help
Usage: paper_cli.py train-adam-covertree [OPTIONS]

Options:
  -mc, --model-class [sgpr|cdgp]  [required]
  -n, --num-iterations INTEGER    [required]
  -s, --spatial-resolution FLOAT  [required]
  -b, --batch-size INTEGER        [required]
  -tb, --test-batch-size INTEGER
  -l, --learning-rate FLOAT
  -e, --error-threshold FLOAT
  --tip / --no-tip
  --help                          Show this message and exit.
```


### `paper_condition_wasserstein.py`

