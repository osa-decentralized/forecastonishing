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
        print(selector.best_scores_)
        true_answer = None
        # TODO: Finish this test.


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
