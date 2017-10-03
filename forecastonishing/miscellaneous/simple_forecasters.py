"""
This module provides `sklearn` API for simple forecasters such as
moving average, moving median, and exponential moving average.

@author: Nikolay Lysenko
"""


from typing import Dict, Callable, Optional

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class BaseSimpleForecaster(BaseEstimator, RegressorMixin):
    """
    A parent class for simple forecasters.

    :param forecasting_fn:
        function that makes forecast for the next step
    """

    def __init__(self, forecasting_fn: Callable):
        self.forecasting_fn = forecasting_fn

    def fit(
            self,
            X: Optional[pd.DataFrame] = None,
            y: Optional[pd.Series] = None
            ) -> 'BaseSimpleForecaster':
        """
        An empty method added only for the sake of compatibility
        with `sklearn` API. Simple forecasters have no fitting.

        :param X:
            supplementary features, you can leave this argument
            untouched
        :param y:
            target time series, you can leave this argument
            untouched
        :return:
            class instance without any changes
        """
        return self

    def predict(
            self,
            X: Optional[pd.DataFrame],
            y: pd.Series,
            horizon: int = 1
            ) -> pd.Series:
        """
        Predict time series several steps ahead.

        :param X:
            supplementary features, you can pass `None`
        :param y:
            target time series
        :param horizon:
            number of steps ahead
        :return:
            predictions for future steps
        """
        ser = y.copy()
        for i in range(horizon):
            ser.append(pd.Series(self.forecasting_fn(ser).iloc[-1]))
        return ser.iloc[-horizon:]


class MovingAverageForecaster(BaseSimpleForecaster):
    """
    Maker of forecasts with moving average.

    :param rolling_kwargs:
        parameters of rolling (moving) window
    """

    def __init__(self, rolling_kwargs: Optional[Dict] = None):
        self.rolling_kwargs = (
            rolling_kwargs if rolling_kwargs is not None else dict()
        )
        super().__init__(lambda x: x.rolling(**self.rolling_kwargs).mean())


class MovingMedianForecaster(BaseSimpleForecaster):
    """
    Maker of forecasts with moving median.

    :param rolling_kwargs:
        parameters of rolling (moving) window
    """

    def __init__(self, rolling_kwargs: Optional[Dict] = None):
        self.rolling_kwargs = (
            rolling_kwargs if rolling_kwargs is not None else dict()
        )
        super().__init__(lambda x: x.rolling(**self.rolling_kwargs).median())


class ExponentialMovingAverageForecaster(BaseSimpleForecaster):
    """
    Maker of forecasts with exponential moving average.

    :param ewm_kwargs:
        parameters of exponential window
    """

    def __init__(self, ewm_kwargs: Optional[Dict] = None):
        self.ewm_kwargs = (
            ewm_kwargs if ewm_kwargs is not None else dict()
        )
        super().__init__(lambda x: x.ewm(**self.ewm_kwargs).mean())
