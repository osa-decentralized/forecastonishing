"""
Inside this module, there is a class that runs adaptive selection
of short-term forecasting models.

Time series forecasting has a property that all observations are
ordered. Depending on position, behavior of series can vary and so
one method can yield better results for some moments while
another method can outperform it for some other moments. This is
the reason why adaptive selection is useful for many series.

@author: Nikolay Lysenko
"""


from typing import List, Dict, Callable, Any, Optional

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

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
    has no fitting. By default, class uses moving average,
    moving median, and exponential moving average, but you can pass
    your own simple forecaster for initialization of a new instance.

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
    * Can be easily paralleled and distributed, can deal with
      thousands of time series in one call.

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
        the bigger its value, the better is a forecaster
    :param horizon:
        number of steps ahead to be forecasted at each iteration,
        default is 1
    :param n_evaluational_steps:
        number of iterations at each of which forecasters make
        predictions, the next step is obtained from the preceding
        step by going one step forward, default value is 1, i.e.,
        forecasters are evaluated at the last `horizon` observations
        from each series.
    :param n_jobs:
        number of jobs for internal paralleling, default is 1
    """

    def __init__(
            self,
            candidates: Optional[Dict[Any, List[Any]]] = None,
            evaluation_fn: Optional[Callable] = None,
            horizon: Optional[int] = 1,
            n_evaluational_steps: Optional[int] = 1,
            n_jobs: Optional[int] = 1
            ):
        self.candidates = candidates
        self.evaluation_fn = evaluation_fn
        self.horizon = horizon
        self.n_evaluational_steps = n_evaluational_steps
        self.n_jobs = n_jobs
        # Trailing underscore means that attribute is due to fitting.
        self.name_of_target_ = None
        self.series_keys_ = None
        self.scoring_keys_ = None
        self.best_scores_ = None

    def __get_candidates(self) -> List[Any]:
        # Get list of instances that are ready for predicting with them.
        if self.candidates is None:
            raw_description = {
                MovingAverageForecaster():
                    [{'window': w, 'min_periods': 1} for w in range(1, 11)],
                MovingMedianForecaster():
                    [{'window': w, 'min_periods': 1} for w in range(3, 11)],
                ExponentialMovingAverageForecaster():
                    [{'halflife': h, 'min_periods': 1}
                     for h in np.arange(1, 6, 0.5)]
            }
        else:
            raw_description = self.candidates
        nested = [[forecaster.set_params(**kwargs)
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
        # TODO: Do we need enumeration that prevents weird issues?

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

    def _evaluate_forecaster_on_one_series(
            self,
            ser: pd.Series,
            forecaster: Any
            ) -> float:
        # Evaluate performance of `forecaster` on `ser`.
        frontier = len(ser) - self.horizon - self.n_evaluational_steps + 1
        actual_values = pd.Series()
        predictions = pd.Series()
        for i in range(self.n_evaluational_steps):
            curr_actual = ser[(frontier + i):(frontier + i + self.horizon)]
            actual_values = actual_values.append(curr_actual)
            curr_predictions = forecaster.predict(
                ser[:frontier + i],
                self.horizon
            )
            predictions = predictions.append(curr_predictions)
        score = self.evaluation_fn(actual_values, predictions)
        return score

    def _fit(
            self,
            df: pd.DataFrame
            ) -> 'OnTheFlySelector':
        # Implement internal logic of fitting.

        extra_keys = list(set(self.scoring_keys_) - set(self.series_keys_))
        if extra_keys:
            raise ValueError(
                '`scoring_keys` contains a key not from `series_keys`'
            )
        detailed_keys = list(set(self.series_keys_) - set(self.scoring_keys_))

        self.__create_table_for_results(df)
        candidates = self.__get_candidates()
        for candidate in candidates:
            pass  # TODO: Use `joblib.Parallel` with `df.groupby`.

        return self

