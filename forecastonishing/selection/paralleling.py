"""
This file contains functions that allows running adaptive
selection in parallel.

@author: Nikolay Lysenko
"""


from functools import partial
from typing import List, Dict, Any, Optional

import pandas as pd
from sklearn.base import clone

import pathos.multiprocessing as mp  # It can serialize class methods.


def add_partition_key(
        df: pd.DataFrame,
        series_keys: List[str],
        n_partitions: int
        ) -> pd.DataFrame:
    """
    Add to `df` a new column that helps to balance load between
    different processes uniformly.

    :param df:
        data to be transformed in long format
    :param series_keys:
        columns that are identifiers of unique time series
    :param n_partitions:
        number of processes that will be used for parallel
        execution
    :return:
        DataFrame with a new column named 'partition_key'
    """
    keys_df = df[series_keys].drop_duplicates()
    keys_df = keys_df \
        .reset_index() \
        .rename(columns={'index': 'partition_key'})
    keys_df['partition_key'] = keys_df['partition_key'].apply(
        lambda x: x % n_partitions
    )
    df = df.merge(keys_df, on=series_keys)
    return df


def fit_selector_to_one_partition(
        df: pd.DataFrame,
        selector_instance: Any,
        selector_kwargs: Dict[str, Any],
        fit_kwargs: Dict[str, Any]
        ) -> 'type(selector_instance)':
    """
    """
    selector = clone(selector_instance).set_params(**selector_kwargs)
    selector.fit(df, **fit_kwargs)
    return selector


def fit_selector_in_parallel(
        selector_instance: Any,
        selector_kwargs: Dict[str, Any],
        df: pd.DataFrame,
        name_of_target: str,
        series_keys: List[str],
        scoring_keys: Optional[List[str]] = None,
        n_processes: int = 1
        ) -> 'type(selector_instance)':
    """
    """
    fit_kwargs = {
        'name_of_target': name_of_target,
        'series_keys': series_keys,
        'scoring_keys': scoring_keys or series_keys
    }
    try:
        df = add_partition_key(df, series_keys, n_processes)
        fit_to_one_partition = partial(
            fit_selector_to_one_partition,
            selector_instance=selector_instance,
            selector_kwargs=selector_kwargs,
            fit_kwargs=fit_kwargs
        )
        selectors = mp.Pool(n_processes).map(
            fit_to_one_partition,
            [group for _, group in df.groupby('partition_key', as_index=False)]
        )
        results_tables = [
            selector.best_scores_ for selector in selectors
        ]

        best_scores = pd.concat(results_tables)
        selector = selectors[0]  # An arbitrary fitted selector.
        selector.best_scores_ = best_scores
        return selector
    finally:
        df.drop('partition_key', axis=1, inplace=True)
