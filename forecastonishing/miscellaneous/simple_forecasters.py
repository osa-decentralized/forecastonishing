"""
This module endows simple forecasters such as moving average,
moving median, and exponential moving average with API that is quite
similar to that of `sklearn`.

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

    def __init__(
            self,
            forecasting_fn: Callable
            ):
        self.forecasting_fn = forecasting_fn

    def fit(
            self,
            ser: Optional[pd.Series] = None
            ) -> 'BaseSimpleForecaster':
        """
        An empty method added only for the sake of similarity to
        `sklearn` API. Simple forecasters have no fitting.

        :param ser:
            target time series, you can leave this argument
            untouched
        :return:
            class instance without any changes
        """
        return self

    def predict(
            self,
            ser: pd.Series,
            horizon: int = 1
            ) -> pd.Series:
        """
        Predict time series several steps ahead.

        :param ser:
            target time series
        :param horizon:
            number of steps ahead
        :return:
            predictions for future steps
        """
        new_ser = ser.copy()
        for i in range(horizon):
            new_ser.append(pd.Series(self.forecasting_fn(new_ser).iloc[-1]))
        return new_ser.iloc[-horizon:]


class MovingAverageForecaster(BaseSimpleForecaster):
    """
    Maker of forecasts with moving average.

    :param rolling_kwargs:
        parameters of rolling (moving) window
    """

    def __init__(self, rolling_kwargs: Optional[Dict] = None):
        self.rolling_kwargs = rolling_kwargs or dict()
        if self.rolling_kwargs.get('window', None) is None:
            self.rolling_kwargs['window'] = 3
        super().__init__(
            lambda x: x
            .tail(self.rolling_kwargs['window'])
            .rolling(**self.rolling_kwargs)
            .mean()
        )


class MovingMedianForecaster(BaseSimpleForecaster):
    """
    Maker of forecasts with moving median.

    :param rolling_kwargs:
        parameters of rolling (moving) window
    """

    def __init__(self, rolling_kwargs: Optional[Dict] = None):
        self.rolling_kwargs = rolling_kwargs or dict()
        if self.rolling_kwargs.get('window', None) is None:
            self.rolling_kwargs['window'] = 3
        super().__init__(
            lambda x: x
            .tail(self.rolling_kwargs['window'])
            .rolling(**self.rolling_kwargs)
            .median()
        )


class ExponentialMovingAverageForecaster(BaseSimpleForecaster):
    """
    Maker of forecasts with exponential moving average.

    :param ewm_kwargs:
        parameters of exponential window and one additional
        parameter with key 'n_steps_to_use' that specifies
        the length of current tail of a series to be used
        for predicting, its default value is length of a
        series
    """

    def __init__(self, ewm_kwargs: Optional[Dict] = None):
        self.ewm_kwargs = ewm_kwargs or dict()
        if self.ewm_kwargs.get('n_steps_to_use', None) is None:
            self.ewm_kwargs['n_steps_to_use'] = len
        else:
            self.ewm_kwargs['n_steps_to_use'] = (
                lambda _: self.ewm_kwargs['n_steps_to_use']
            )
        super().__init__(
            lambda x: x
            .tail(self.ewm_kwargs['n_steps_to_use'](x))
            .ewm(**{k: v
                    for k, v in self.ewm_kwargs.items()
                    if k != 'n_steps_to_use'})
            .mean()
        )
