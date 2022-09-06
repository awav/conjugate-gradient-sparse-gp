from asyncio import selector_events
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
from tinydb import TinyDB, where, Query

# Experiment changes `model', `precision' [fp32 | fp64], and `clustering_type' [uniform | kmeans | oips | covertree], `jitter'.


@click.command()
@click.argument("files", nargs=-1, type=click.Path(dir_okay=False))
def main(files: Sequence[Union[Path, str]]):
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

    def query_name_str(values):
        name = "_".join([str(v) for v in values])
        return name

    metrics = ["train/elbo", "test/rmse", "test/nlpd", "condition_number"]
    aggregation = ["mean", "std"]

    with tempfile.NamedTemporaryFile(prefix="tinydb.", suffix=".json") as tmpfile:
        db_filepath = tmpfile.name
        db = TinyDB(db_filepath)
        db.insert_multiple(results)

        value_sequence = [list(key_values[key]) for key in keys_sequence]
        product_query_values = product(*value_sequence)

        output_data = {}

        for query_values in product_query_values:
            name = query_name_str(query_values)
            key_value_pairs = list(zip(keys_sequence, query_values))
            full_query = foldl_queries(key_value_pairs)
            selected_rows = db.search(full_query)
            df = pd.DataFrame(selected_rows)
            df = df.sort_values(by="num_inducing_points")
            instructions = {metric: aggregation for metric in metrics}
            data_collection = df.groupby(by="num_inducing_points").agg(instructions)
            output_data[name] = data_collection
        
        

    click.echo(f"Finished")


if __name__ == "__main__":
    main()
