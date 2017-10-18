"""
This module contains tests of code from the file named
`../forecastonishing/miscellaneous/metrics.py`.

@author: Nikolay Lysenko
"""


import unittest

import numpy as np
import pandas as pd

from forecastonishing.miscellaneous.metrics import (
    overall_r_squared,
    averaged_r_squared,
    overall_censored_mape,
    averaged_censored_mape
)


def get_example() -> pd.DataFrame():
    """
    Get example of DataFrame with actual values and predictions.

    :return:
        example DataFrame
    """
    df = pd.DataFrame(
        [[1, 2, 3],
         [1, 4, 5],
         [2, 10, 8],
         [2, 8, 10]],
        columns=['key', 'actual_value', 'prediction']
    )
    return df


class TestMetrics(unittest.TestCase):
    """
    Tests of evaluational metrics.
    """

    def test_overall_r_squared(self) -> type(None):
        """
        Test `overall_r_squared` function.

        :return:
            None
        """
        df = get_example()
        score = overall_r_squared(df)
        self.assertEquals(score, 0.75)

    def test_averaged_r_squared(self) -> type(None):
        """
        Test `averaged_r_squared` function.

        :return:
            None
        """
        df = get_example()
        score = averaged_r_squared(df, ['key'])
        self.assertEquals(score, -1.5)

    def test_overall_censored_mape(self) -> type(None):
        """
        Test `overall_censored_mape` function.

        :return:
            None
        """
        df = get_example()
        score = overall_censored_mape(df)
        self.assertEquals(score, 30)

    def test_overall_censored_mape_with_zeros(self) -> type(None):
        """
        Test correct work of `overall_censored_mape` function
        with zero forecasts made for zero actual values.

        :return:
            None
        """
        first_df = get_example()
        second_df = pd.DataFrame(
            [[3, 0, 0],
             [3, 0, 0],
             [4, 0, 0],
             [4, 0, 0]],
            columns=['key', 'actual_value', 'prediction']
        )
        df = pd.concat([first_df, second_df])
        score = overall_censored_mape(df)
        self.assertEquals(score, 15)

    def test_overall_censored_mape_with_missings(self) -> type(None):
        """
        Test correct work of `overall_censored_mape` function
        with some values missed.

        :return:
            None
        """
        df = get_example()
        df.loc[0, 'prediction'] = None
        df.loc[1, 'actual_value'] = np.nan
        score = overall_censored_mape(df)
        self.assertEquals(score, 22.5)

    def test_averaged_censored_mape(self) -> type(None):
        """
        Test `averaged_r_censored_mape` function.

        :return:
            None
        """
        df = get_example()
        score = averaged_censored_mape(df, ['key'])
        self.assertEquals(score, 30)

    def test_averaged_censored_mape_with_empty_series(self) -> type(None):
        """
        Test correct work of `overall_censored_mape` function
        with DataFrame that contains an empty time series.

        :return:
            None
        """
        df = get_example()
        df.loc[0, 'prediction'] = None
        df.loc[1, 'actual_value'] = np.nan
        score = averaged_censored_mape(df, ['key'])
        self.assertEquals(score, 22.5)


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [
        TestMetrics()
    ]
    for tester in testers:
        suite = test_loader.loadTestsFromModule(tester)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()
