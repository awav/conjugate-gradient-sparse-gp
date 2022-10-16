from typing import Sequence, Union
import functools
import operator
import pandas as pd
from itertools import product
import click
import numpy as np
import json
from cli_utils import expand_paths_with_wildcards
from pathlib import Path
import tempfile
from tinydb import TinyDB, where
import matplotlib.pyplot as plt

# Experiment changes `model', `precision' [fp32 | fp64], and `clustering_type' [uniform | kmeans | oips | covertree], `jitter'.


@click.command()
@click.option("--output-dir", "-o", type=click.Path(file_okay=False), required=True)
@click.argument("files", nargs=-1, type=click.Path(dir_okay=False))
def main(output_dir, files: Sequence[Union[Path, str]]):
    """
    Loads the data from given files, post processes and find mean and standard deviations
    for metrics `test/rmse`, `test/nlpd`, `train/elbo`, `condition_number`.
    """

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    files = files if isinstance(files, (list, tuple)) else [files]
    expanded_files = expand_paths_with_wildcards(files)

    def load_json_results():
        for filepath in expanded_files:
            with open(filepath) as opened_file:
                result = json.load(opened_file)
                yield result

    results = list(load_json_results())
    keys_sequence = ["model", "precision", "clustering_type", "jitter"]
    key_values = {key: set() for key in keys_sequence}

    for result in results:
        for key, values in key_values.items():
            if key in result:
                values.add(result[key])

    def foldl_queries(keys_values):
        def query_merger(acc_queries, key_value):
            key, value = key_value
            query = where(key) == value
            if acc_queries is None:
                return query
            return acc_queries & query

        return functools.reduce(query_merger, keys_values, None)

    def query_name_str(key_vals):
        def fmt(key, val):
            if key == "jitter":
                return f"{val:.0e}"
            return str(val)

        name = "_".join([fmt(key, val) for key, val in key_vals])
        return name

    ip_key = "num_inducing_points"

    # Post processing

    with tempfile.NamedTemporaryFile(prefix="tinydb.", suffix=".json") as tmpfile:
        db_filepath = tmpfile.name
        db = TinyDB(db_filepath)
        db.insert_multiple(results)

        value_sequence = [list(key_values[key]) for key in keys_sequence]
        product_query_values = product(*value_sequence)

        for query_values in product_query_values:
            key_value_pairs = list(zip(keys_sequence, query_values))
            name = query_name_str(key_value_pairs)
            full_query = foldl_queries(key_value_pairs)
            selected_rows = db.search(full_query)
            if selected_rows == []:
                continue
            df = pd.DataFrame(selected_rows)
            df = df.sort_values(by=ip_key)

            output_filepath = output_dir / (name + ".csv")
            df.to_csv(output_filepath)

    click.echo(f"Finished")


if __name__ == "__main__":
    main()
