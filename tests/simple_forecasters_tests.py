"""
This module contains tests of code from the file named
`../forecastonishing/miscellaneous/simple_forecasters.py`.

@author: Nikolay Lysenko
"""


import unittest

import numpy as np
import pandas as pd

from forecastonishing.miscellaneous.simple_forecasters import (
    MovingAverageForecaster,
    MovingMedianForecaster,
    ExponentialMovingAverageForecaster
)


class TestMovingAverageForecaster(unittest.TestCase):
    """
    Tests of `MovingAverageForecaster` class.
    """

    def test_fit(self) -> type(None):
        """
        Test `fit` method.

        :return:
            None
        """
        maf = MovingAverageForecaster()
        maf.fit(None)
        self.assertTrue(hasattr(maf, 'is_fitted_'))
        maf = MovingAverageForecaster()
        maf.fit(pd.Series([0, 1]))
        self.assertTrue(hasattr(maf, 'is_fitted_'))

    def test_predict(self) -> type(None):
        """
        Test `predict` method.

        :return:
            None
        """
        ser = pd.Series([1, 2, 6], dtype=np.float32)
        maf = MovingAverageForecaster({'window': 3})
        maf.fit(ser)
        result = maf.predict(ser)
        true_answer = np.array([3], dtype=np.float32)
        self.assertTrue(np.array_equal(result, true_answer))

    def test_predict_with_horizon(self) -> type(None):
        """
        Test `predict` method with forecasting multiple steps ahead.

        :return:
            None
        """
        ser = pd.Series([1, 2, 6], dtype=np.float32)
        maf = MovingAverageForecaster({'window': 3})
        maf.fit(ser)
        result = maf.predict(ser, horizon=3)
        true_answer = np.array([3, 11 / 3, 38 / 9], dtype=np.float32)
        self.assertTrue(np.allclose(result, true_answer))


class TestMovingMedianForecaster(unittest.TestCase):
    """
    Tests of `MovingMedianForecaster` class.
    """

    def test_fit(self) -> type(None):
        """
        Test `fit` method.

        :return:
            None
        """
        mmf = MovingMedianForecaster()
        mmf.fit(None)
        self.assertTrue(hasattr(mmf, 'is_fitted_'))
        mmf = MovingMedianForecaster()
        mmf.fit(pd.Series([0, 1]))
        self.assertTrue(hasattr(mmf, 'is_fitted_'))

    def test_predict(self) -> type(None):
        """
        Test `predict` method.

        :return:
            None
        """
        ser = pd.Series([1, 2, 6], dtype=np.float32)
        mmf = MovingMedianForecaster({'window': 3})
        mmf.fit(ser)
        result = mmf.predict(ser)
        true_answer = np.array([2], dtype=np.float32)
        self.assertTrue(np.array_equal(result, true_answer))

    def test_predict_with_horizon(self) -> type(None):
        """
        Test `predict` method with forecasting multiple steps ahead.

        :return:
            None
        """
        ser = pd.Series([1, 2, 6, 4], dtype=np.float32)
        mmf = MovingMedianForecaster({'window': 4})
        mmf.fit(ser)
        result = mmf.predict(ser, horizon=3)
        true_answer = np.array([3, 3.5, 3.75], dtype=np.float32)
        self.assertTrue(np.array_equal(result, true_answer))


class TestExponentialMovingAverageForecaster(unittest.TestCase):
    """
    Tests of `ExponentialMovingAverageForecaster` class.
    """

    def test_fit(self) -> type(None):
        """
        Test `fit` method.

        :return:
            None
        """
        eaf = ExponentialMovingAverageForecaster()
        eaf.fit(pd.Series([0, 1]))
        self.assertTrue(hasattr(eaf, 'is_fitted_'))

    def test_predict(self) -> type(None):
        """
        Test `predict` method.

        :return:
            None
        """
        ser = pd.Series([1, 2, 6], dtype=np.float32)
        eaf = ExponentialMovingAverageForecaster({'alpha': 0.5})
        eaf.fit(ser)
        result = eaf.predict(ser)
        true_answer = np.array([4.142857142857142], dtype=np.float32)
        self.assertTrue(np.allclose(result, true_answer))

    def test_predict_with_horizon(self) -> type(None):
        """
        Test `predict` method with forecasting multiple steps ahead.

        :return:
            None
        """
        ser = pd.Series([1, 2, 6], dtype=np.float32)
        eaf = ExponentialMovingAverageForecaster({'alpha': 0.5})
        eaf.fit(ser)
        result = eaf.predict(ser, horizon=3)
        true_answer = np.array([4.142857142857142,
                                4.36734693877551,
                                4.536443148688046], dtype=np.float32)
        self.assertTrue(np.allclose(result, true_answer))


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [
        TestMovingAverageForecaster(),
        TestMovingMedianForecaster(),
        TestExponentialMovingAverageForecaster()
    ]
    for tester in testers:
        suite = test_loader.loadTestsFromModule(tester)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()
