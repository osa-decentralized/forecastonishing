"""
This module contains tests of code from the file named
`../forecastonishing/selection/on_the_fly_selector.py`.

@author: Nikolay Lysenko
"""


import unittest

import numpy as np
import pandas as pd

from forecastonishing.selection.on_the_fly_selector import OnTheFlySelector
from forecastonishing.miscellaneous.simple_forecasters import (
    MovingAverageForecaster,
    MovingMedianForecaster
)


class TestOnTheFlySelector(unittest.TestCase):
    """
    Tests of `OnTheFlySelector` class.
    """

    def test_fit(self) -> type(None):
        """
        Test `fit` method.

        :return:
            None
        """
        candidates = {
            MovingAverageForecaster():
                [{'rolling_kwargs': {'window': w, 'min_periods': 1}}
                 for w in range(1, 3)],
            MovingMedianForecaster():
                [{'rolling_kwargs': {'window': w, 'min_periods': 1}}
                 for w in range(3, 4)]
        }
        selector = OnTheFlySelector(candidates)

        df = pd.DataFrame(
            [[1, 2],
             [1, 3],
             [1, 6],
             [1, 5],
             [2, 3],
             [2, 4],
             [2, 4.5],
             [2, 1]],
            columns=['key', 'target']
        )

        selector.fit(df, 'target', ['key'])

        self.assertTrue(isinstance(
            selector.best_scores_['forecaster'][1], MovingAverageForecaster
        ))
        self.assertEquals(
            selector.best_scores_['forecaster'][1].get_params(),
            {'rolling_kwargs': {'window': 2, 'min_periods': 1}}
        )
        self.assertEquals(selector.best_scores_['score'][1], -0.25)
        self.assertTrue(isinstance(
            selector.best_scores_['forecaster'][2], MovingMedianForecaster
        ))
        self.assertEquals(
            selector.best_scores_['forecaster'][2].get_params(),
            {'rolling_kwargs': {'window': 3, 'min_periods': 1}}
        )
        self.assertEquals(selector.best_scores_['score'][2], -9)

    def test_fit_with_scoring_keys(self) -> type(None):
        """
        Test `fit` method with group-wise selection instead of
        series-wise selection.

        :return:
            None
        """
        candidates = {
            MovingAverageForecaster():
                [{'rolling_kwargs': {'window': w, 'min_periods': 1}}
                 for w in range(1, 3)],
            MovingMedianForecaster():
                [{'rolling_kwargs': {'window': w, 'min_periods': 1}}
                 for w in range(3, 4)]
        }
        selector = OnTheFlySelector(candidates)

        df = pd.DataFrame(
            [[1, 1, 2],
             [1, 1, 3],
             [1, 1, 6],
             [1, 1, 5],
             [1, 2, 3],
             [1, 2, 4],
             [1, 2, 4.5],
             [1, 2, 1],
             [2, 1, 3],
             [2, 1, 2],
             [2, 1, -1],
             [2, 1, 7],
             [2, 2, 0],
             [2, 2, 3],
             [2, 2, 2],
             [2, 2, 5]],
            columns=['group', 'object', 'target']
        )

        selector.fit(df, 'target', ['group', 'object'], scoring_keys=['group'])

        self.assertTrue(isinstance(
            selector.best_scores_['forecaster'][1], MovingAverageForecaster
        ))
        self.assertEquals(
            selector.best_scores_['forecaster'][1].get_params(),
            {'rolling_kwargs': {'window': 2, 'min_periods': 1}}
        )
        self.assertEquals(selector.best_scores_['score'][1], -5.40625)
        self.assertTrue(isinstance(
            selector.best_scores_['forecaster'][2], MovingMedianForecaster
        ))
        self.assertEquals(
            selector.best_scores_['forecaster'][2].get_params(),
            {'rolling_kwargs': {'window': 3, 'min_periods': 1}}
        )
        self.assertEquals(selector.best_scores_['score'][2], -17)


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [
        TestOnTheFlySelector()
    ]
    for tester in testers:
        suite = test_loader.loadTestsFromModule(tester)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()
