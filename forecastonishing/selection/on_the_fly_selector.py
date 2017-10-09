"""
Inside this module, there is a class that runs adaptive selection
of short-term forecasting models. Some functions that helps to
run adaptive selection in parallel can be found here as well.

Time series forecasting has a property that all observations are
ordered. Depending on position, behavior of series can vary and so
one method can yield better results for some moments while
another method can outperform it for some other moments. This is
the reason why adaptive selection is useful for many series.

@author: Nikolay Lysenko
"""


from typing import List, Dict, Callable, Any, Optional
from functools import partial

from joblib import Parallel, delayed  # TODO: Use it in paralleling functions.
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_squared_error

from forecastonishing.miscellaneous.simple_forecasters import (
    MovingAverageForecaster,
    MovingMedianForecaster,
    ExponentialMovingAverageForecaster
)


class OnTheFlySelector(BaseEstimator, RegressorMixin):
    """
    This class provides functionality for adaptive short-term
    forecasting.

    The class is designed for a case of many time series and many
    simple forecasters - if so, it is too expensive to store all
    forecasts in any place other than operating memory and it
    is better to compute them on-the-fly and then store only selected
    values.

    What about terminology, simple forecaster means a forecaster that
    has no fitting. By default, the class uses moving average,
    moving median, and exponential moving average, but you can pass
    your own simple forecaster to initialization of a new instance.

    Selection is preferred over stacking, because base forecasters are
    quite similar to each other and so they have many common mistakes.

    Advantages of adaptive on-the-fly selection are as follows:
    * It always produces sane results, abnormal over-forecasts or
      under-forecasts are impossible;
    * Each time series is tailored individually, this is not a model
      that predicts for several time series without taking into
      consideration their identities;
    * Uses no external features and so can be used for modelling of
      residuals of a more complicated model;
    * Can be easily paralleled or distributed, can deal with thousands
      of time series in one call.

    Limitations of adaptive on-the-fly selection are as follows:
    * Not suitable for time series that are strongly influenced by
      external factors;
    * Not suitable for non-stationary time series (e.g., time series
      with trend or seasonality) until they are not made stationary;
    * Long-term forecasts made with it converges to constants.

    :param candidates:
        forecasters to select from, mapping from instances of
        regressors to kwargs of their initializations, default value
        results in moving averages of ten distinct windows,
        moving medians of eight distinct windows and exponential
        moving averages of ten distinct half-lifes
    :param evaluation_fn:
        function that is used for selection of the best forecasters,
        the bigger its value, the better is a forecaster, default is
        negative mean squared error
    :param horizon:
        number of steps ahead to be forecasted at each iteration,
        default is 1
    :param n_evaluational_steps:
        number of iterations at each of which forecasters make
        predictions, the next step is obtained from the preceding
        step by going one step forward, default value is 1, i.e.,
        forecasters are evaluated at the last `horizon` observations
        from each series.
    :param verbose:
        if it is greater than 0, a progress bar with tried candidates
        is shown, default is 0
    """

    def __init__(
            self,
            candidates: Optional[Dict[Any, List[Any]]] = None,
            evaluation_fn: Optional[Callable] = None,
            horizon: Optional[int] = 1,
            n_evaluational_steps: Optional[int] = 1,
            verbose: Optional[int] = 0
            ):
        self.candidates = candidates
        self.evaluation_fn = evaluation_fn
        self.horizon = horizon
        self.n_evaluational_steps = n_evaluational_steps
        self.verbose = verbose

    def __get_candidates(self) -> List[Any]:
        # Get list of instances that are ready to fitting.
        if self.candidates is None:
            raw_description = {
                MovingAverageForecaster():
                    [{'rolling_kwargs': {'window': w, 'min_periods': 1}}
                     for w in range(1, 11)],
                MovingMedianForecaster():
                    [{'rolling_kwargs': {'window': w, 'min_periods': 1}}
                     for w in range(3, 11)],
                ExponentialMovingAverageForecaster():
                    [{'ewm_kwargs': {
                          'halflife': h,
                          'min_periods': 1,
                     },
                     'n_steps_to_use': 10}
                     for h in np.arange(1, 6, 0.5)]
            }
        else:
            raw_description = self.candidates
        nested = [[clone(forecaster).set_params(**kwargs)
                   for forecaster, list_of_kwargs in raw_description.items()
                   for kwargs in list_of_kwargs]]
        forecasters = [item for sublist in nested for item in sublist]
        return forecasters

    def __create_table_for_results(
            self,
            df: pd.DataFrame
            ) -> type(None):
        # Create `pd.DataFrame` where scores of candidates will be stored.
        best_scores = df[self.scoring_keys_].drop_duplicates()
        best_scores['score'] = -np.inf
        best_scores['forecaster'] = None
        best_scores.set_index(self.scoring_keys_, inplace=True)
        self.best_scores_ = best_scores

    def __update_table_with_results(
            self,
            tried_candidate: Any,
            candidate_scores: pd.Series
            ) -> type(None):
        # Update results table after a new candidate is tried.

        scores = candidate_scores.to_frame(name='curr_score')
        scores['curr_forecaster'] = tried_candidate

        self.best_scores_ = self.best_scores_.merge(
            scores, how='left', left_index=True, right_index=True
        )
        self.best_scores_['score'] = self.best_scores_.apply(
            lambda x: max(x['score'], x['curr_score']), axis=1
        )
        self.best_scores_['forecaster'] = self.best_scores_.apply(
            lambda x: x['curr_forecaster']
                      if x['curr_score'] >= x['score']
                      else x['forecaster'],
            axis=1)
        self.best_scores_.drop('curr_score', axis=1, inplace=True)
        self.best_scores_.drop('curr_forecaster', axis=1, inplace=True)

    def _evaluate_forecaster_on_one_series(
            self,
            ser: pd.Series,
            forecaster: Any
            ) -> float:
        # Evaluate performance of `forecaster` on `ser`.
        frontier = len(ser) - self.horizon - self.n_evaluational_steps + 1
        actual_values = np.array([])
        predictions = np.array([])
        for i in range(self.n_evaluational_steps):
            curr_actual = ser[(frontier + i):(frontier + i + self.horizon)]
            actual_values = np.append(actual_values, curr_actual.values)
            curr_predictions = forecaster.predict(
                ser[:frontier + i],
                self.horizon
            )
            predictions = np.append(predictions, curr_predictions)
        score = self.evaluation_fn_(actual_values, predictions)
        return score

    def _evaluate_forecaster_on_whole_dataframe(
            self,
            df: pd.DataFrame,
            forecaster: Any
            ) -> pd.Series:
        # Evaluate performance of `forecaster` on `df`.
        evaluate_on_one_series = partial(
            self._evaluate_forecaster_on_one_series,
            forecaster=forecaster
        )
        result = (
            df[self.series_keys_ + [self.name_of_target_]]
            .groupby(self.series_keys_)[self.name_of_target_]
            .apply(
               evaluate_on_one_series
            )
        )
        return result

    def _fit(self, df: pd.DataFrame) -> 'OnTheFlySelector':
        # Implement internal logic of fitting.

        extra_keys = list(set(self.scoring_keys_) - set(self.series_keys_))
        if extra_keys:
            raise ValueError(
                '`scoring_keys` contains a key not from `series_keys`'
            )
        detailed_keys = list(set(self.series_keys_) - set(self.scoring_keys_))
        # Rearrange `series_keys` in order to have `scoring_keys` first.
        self.series_keys_ = self.scoring_keys_ + detailed_keys

        self.__create_table_for_results(df)

        candidates = self.__get_candidates()
        candidates = tqdm(candidates) if self.verbose > 0 else candidates
        for candidate in candidates:
            candidate.fit(None)  # TODO: Fit it to a random series from `df`.
            scores = self._evaluate_forecaster_on_whole_dataframe(
                df, candidate
            )
            # Average scores if forecasters are selected for sets of series.
            if detailed_keys:
                scores = scores.groupby(
                    level=list(range(len(self.scoring_keys_)))
                ).mean()
            self.__update_table_with_results(candidate, scores)
        return self

    def fit(
            self,
            df: pd.DataFrame,
            name_of_target: str,
            series_keys: List[str],
            scoring_keys: Optional[List[str]] = None
            ) -> 'OnTheFlySelector':
        """
        Figure out which forecaster should be used for each series.

        :param df:
            DataFrame in long format with (many) time series
        :param name_of_target:
            name of target column
        :param series_keys:
            columns that are identifiers of unique time series
        :param scoring_keys:
            identifiers of groups such that best forecasters are
            selected per a group, not per an individual time series,
            this can be used if you want to reduce overfitting and you
            know which series exhibit similar behavior,
            all elements of this list must be in `series_keys` too,
            default value corresponds to no grouping
        :return:
            fitted instance
        """
        # Trailing underscore means that this attribute is due to fitting.
        # Here, `sklearn` style contradicts to hints of some linters.
        self.evaluation_fn_ = (
            self.evaluation_fn
            if self.evaluation_fn is not None
            else lambda x, y: -mean_squared_error(x, y)
        )
        self.name_of_target_ = name_of_target
        self.series_keys_ = series_keys
        self.scoring_keys_ = scoring_keys or series_keys
        self._fit(df)
        return self


def add_partition_key(
        df: pd.DataFrame,
        series_keys: List[str],
        n_jobs: int
        ) -> pd.DataFrame:
    """
    Add to `df` a new column that helps to balance load between
    different processes uniformly.

    :param df:
        data to be transformed in long format
    :param series_keys:
        columns that are identifiers of unique time series
    :param n_jobs:
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
        lambda x: x % n_jobs
    )
    df = df.merge(keys_df, on=series_keys)
    return df
