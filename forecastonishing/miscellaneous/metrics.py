"""
It is a collection of metrics that can be used for evaluation of
forecasts quality.

@author: Nikolay Lysenko
"""


from functools import partial
from typing import List

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score


def overall_r_squared(
        df: pd.DataFrame,
        target_column: str = 'actual_value',
        predictions_column: str = 'prediction'
        ) -> float:
    """
    Compute coefficient of determination ignoring identities of
    time series.

    :param df:
        DataFrame in long format with time series and predictions
    :param target_column:
        name of column with actual values
    :param predictions_column:
        name of column with predictions
    :return:
        overall R^2 coefficient of determination
    """
    return r2_score(df[target_column], df[predictions_column])


def averaged_r_squared(
        df: pd.DataFrame,
        series_keys: List[str],
        target_column: str = 'actual_value',
        predictions_column: str = 'prediction'
        ) -> float:
    """
    Compute coefficient of determination for each of time series
    and then average results.

    :param df:
        DataFrame in long format with time series and predictions
    :param series_keys:
        identifiers of individual time series
    :param target_column:
        name of column with actual values
    :param predictions_column:
        name of column with predictions
    :return:
        averaged R^2 coefficient of determination
    """
    r_squared_fn = partial(
        overall_r_squared,
        target_column=target_column,
        predictions_column=predictions_column
    )
    return df.groupby(series_keys).apply(r_squared_fn).mean()


def overall_censored_mape(
        df: pd.DataFrame,
        target_column: str = 'actual_value',
        predictions_column: str = 'prediction',
        censorship_level: float = 1.0
        ) -> float:
    """
    Compute censored from above MAPE (mean absolute
    percentage error) ignoring time series identities.

    :param df:
        DataFrame in long format with time series and predictions
    :param target_column:
        name of column with actual values
    :param predictions_column:
        name of column with predictions
    :param censorship_level:
        maximal relative error, all errors that are higher
        are replaced with it, default is 1.0 (100%)
    :return:
        overall censored from above MAPE
    """
    result = (
        100 * (
            (df[target_column] - df[predictions_column]).abs() /
            df[target_column]
        )
        .fillna(0)  # If there are no missings, `np.nan` occurs due to 0 / 0.
        .replace(np.inf, censorship_level)
        .apply(lambda x: min(x, censorship_level))
        .mean()
    )
    return result


def averaged_censored_mape(
        df: pd.DataFrame,
        series_keys: List[str],
        target_column: str = 'actual_value',
        predictions_column: str = 'prediction',
        censorship_level: float = 1.0
        ) -> float:
    """
    Compute censored from above MAPE (mean absolute
    percentage error) for each of time series and then
    average results.
    If length of series varies, this can differ from overall
    censored MAPE.

    :param df:
        DataFrame in long format with time series and predictions
    :param series_keys:
        identifiers of individual time series
    :param target_column:
        name of column with actual values
    :param predictions_column:
        name of column with predictions
    :param censorship_level:
        maximal relative error, all errors that are higher
        are replaced with it, default is 1.0 (100%)
    :return:
        averaged censored from above MAPE
    """
    overall_mape_fn = partial(
        overall_censored_mape,
        target_column=target_column,
        predictions_column=predictions_column,
        censorship_level=censorship_level
    )
    result = (
        df
        .groupby(series_keys)
        .apply(overall_mape_fn)
        .mean()
    )
    return result
