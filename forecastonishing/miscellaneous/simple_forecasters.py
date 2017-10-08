"""
This module endows simple forecasters such as moving average,
moving median, and exponential moving average with API that is quite
similar to that of `sklearn`.

@author: Nikolay Lysenko
"""


from typing import Dict, Callable, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


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
        Simple forecasters have no fitting from machine learning
        (implementation-independent) point of view.
        However, conversion of `None` values to default values is run
        here for the sake of compatibility with `sklearn` API.

        :param ser:
            target time series, you can leave this argument
            untouched for some of forecasters
        :return:
            class instance with new attributes that are created here,
            because `sklearn` style requires to create them in `fit`
        """
        self.is_fitted_ = True
        if hasattr(self, 'rolling_kwargs'):
            self.rolling_kwargs_ = self.rolling_kwargs or dict()
            if self.rolling_kwargs_.get('window', None) is None:
                self.rolling_kwargs_['window'] = 3
        if hasattr(self, 'ewm_kwargs'):
            self.ewm_kwargs_ = self.ewm_kwargs or dict()
            if self.ewm_kwargs_.get('n_steps_to_use', None) is None:
                if ser is None:
                    raise(ValueError('Fitting EMAF to no series'))
                self.ewm_kwargs_['n_steps_to_use'] = len(ser)
        return self

    def predict(
            self,
            ser: pd.Series,
            horizon: int = 1
            ) -> np.ndarray:
        """
        Predict time series several steps ahead.

        :param ser:
            target time series
        :param horizon:
            number of steps ahead
        :return:
            predictions for future steps
        """
        check_is_fitted(self, 'is_fitted_')
        new_ser = ser.copy()
        for i in range(horizon):
            new_ser = np.append(new_ser.values,
                                self.forecasting_fn(new_ser).iloc[-1])
            new_ser = pd.Series(new_ser)
        return new_ser.iloc[-horizon:].values


class MovingAverageForecaster(BaseSimpleForecaster):
    """
    Maker of forecasts with moving average.

    :param rolling_kwargs:
        parameters of rolling (moving) window
    """

    def __init__(self, rolling_kwargs: Optional[Dict] = None):
        self.rolling_kwargs = rolling_kwargs
        super().__init__(
            lambda x: x
            .tail(self.rolling_kwargs_['window'])
            .rolling(**self.rolling_kwargs_)
            .mean()
        )


class MovingMedianForecaster(BaseSimpleForecaster):
    """
    Maker of forecasts with moving median.

    :param rolling_kwargs:
        parameters of rolling (moving) window
    """

    def __init__(self, rolling_kwargs: Optional[Dict] = None):
        self.rolling_kwargs = rolling_kwargs
        super().__init__(
            lambda x: x
            .tail(self.rolling_kwargs_['window'])
            .rolling(**self.rolling_kwargs_)
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
        series to which an instance is fitted
    """

    def __init__(self, ewm_kwargs: Optional[Dict] = None):
        self.ewm_kwargs = ewm_kwargs
        super().__init__(
            lambda x: x
            .tail(self.ewm_kwargs_['n_steps_to_use'])
            .ewm(**{k: v
                    for k, v in self.ewm_kwargs_.items()
                    if k != 'n_steps_to_use'})
            .mean()
        )
