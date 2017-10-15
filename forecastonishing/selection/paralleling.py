"""
This file contains functions that allows running adaptive
selection in parallel.

@author: Nikolay Lysenko
"""


from typing import List, Any, Optional

import pandas as pd
from sklearn.base import clone

# It can serialize class methods and lambda functions.
import pathos.multiprocessing as mp


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


def fit_selector_in_parallel(
        selector_instance: Any,
        df: pd.DataFrame,
        name_of_target: str,
        series_keys: List[str],
        scoring_keys: Optional[List[str]] = None,
        n_processes: int = 1
        ) -> 'type(selector_instance)':
    """
    Create a new selector of specified parameters and fit it with
    paralleling based on enumeration of unique time series.

    :param selector_instance:
        instance that specifies class of resulting selector
        and its initial parameters
    :param df:
        DataFrame in long format that contains time series
    :param name_of_target:
        name of target column
    :param series_keys:
        columns that are identifiers of unique time series
    :param scoring_keys:
        identifiers of groups such that best forecasters are
        selected per a group, not per an individual time series,
        see more in documentation on `fit` method of selector
    :param n_processes:
        number of parallel processes, default is 1
    :return:
        new fitted instance of selector
    """
    fit_kwargs = {
        'name_of_target': name_of_target,
        'series_keys': series_keys,
        'scoring_keys': scoring_keys or series_keys
    }
    try:
        df = add_partition_key(df, series_keys, n_processes)
        selectors = mp.Pool(n_processes).map(
            lambda x: clone(selector_instance).fit(x, **fit_kwargs),
            [group for _, group in df.groupby('partition_key', as_index=False)]
        )  # pragma: no cover (`coverage` has issues with multiprocessing)
        results_tables = [
            selector.best_scores_ for selector in selectors
        ]

        best_scores = pd.concat(results_tables)
        selector = selectors[0]  # An arbitrary fitted selector.
        selector.best_scores_ = best_scores
        return selector
    finally:
        df.drop('partition_key', axis=1, inplace=True)


def predict_with_selector_in_parallel(
        selector: Any,
        df: pd.DataFrame,
        n_processes: int = 1
        ) -> pd.DataFrame:
    """
    Predict future values of series with paralleling by series keys.

    :param selector:
        instance that has been fitted before
    :param df:
        DataFrame in long format that contains time series
    :param n_processes:
        number of parallel processes, default is 1
    :return:
        DataFrame in long format with predictions
    """
    try:
        df = add_partition_key(df, selector.series_keys_, n_processes)
        predictions = mp.Pool(n_processes).map(
            lambda x: selector.predict(x),
            [group for _, group in df.groupby('partition_key', as_index=False)]
        )  # pragma: no cover (`coverage` has issues with multiprocessing)
        result = pd.concat(predictions)
        return result
    finally:
        df.drop('partition_key', axis=1, inplace=True)
