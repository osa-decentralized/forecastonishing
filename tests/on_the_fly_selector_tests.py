"""
This module contains tests of code from the file named
`../forecastonishing/selection/on_the_fly_selector.py`.

@author: Nikolay Lysenko
"""


import unittest

import pandas as pd

from forecastonishing.selection.on_the_fly_selector import (
    OnTheFlySelector,
    add_partition_key
)
from forecastonishing.miscellaneous.simple_forecasters import (
    MovingAverageForecaster,
    MovingMedianForecaster,
    ExponentialMovingAverageForecaster
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

    def test_fit_with_default_candidates(self) -> type(None):
        """
        Test `fit` method with default set of candidates.

        :return:
            None
        """
        selector = OnTheFlySelector()

        df = pd.DataFrame(
            [[1, 2],
             [1, 3],
             [1, 6],
             [1, 5],
             [1, 8],
             [1, 3],
             [1, 4],
             [1, 2],
             [1, 5],
             [1, 7],
             [1, 5.5],
             [2, 3],
             [2, 4],
             [2, 4.5],
             [2, 1],
             [2, 0],
             [2, 0],
             [2, 9],
             [2, 4],
             [2, 8],
             [2, 5],
             [2, 3]],
            columns=['key', 'target']
        )

        selector.fit(df, 'target', ['key'])

        self.assertTrue(isinstance(
            selector.best_scores_['forecaster'][1],
            ExponentialMovingAverageForecaster
        ))
        self.assertEquals(
            selector.best_scores_['forecaster'][1].get_params(),
            {'ewm_kwargs': {'halflife': 1.0, 'min_periods': 1},
             'n_steps_to_use': 10}
        )
        self.assertAlmostEqual(selector.best_scores_['score'][1], -0.001978206)
        self.assertTrue(isinstance(
            selector.best_scores_['forecaster'][2], MovingAverageForecaster
        ))
        self.assertEquals(
            selector.best_scores_['forecaster'][2].get_params(),
            {'rolling_kwargs': {'window': 10, 'min_periods': 1}}
        )
        self.assertAlmostEqual(selector.best_scores_['score'][2], -0.7225)

    def test_fit_with_evaluational_interval(self) -> type(None):
        """
        Test `fit` method with selection based on results for
        several steps.

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
        selector = OnTheFlySelector(candidates, n_evaluational_steps=3)

        df = pd.DataFrame(
            [[1, 2],
             [1, 3],
             [1, 6],
             [1, 5],
             [1, 4],
             [1, 3],
             [2, 3],
             [2, 4],
             [2, 4.5],
             [2, 1],
             [2, 7],
             [2, 2]],
            columns=['key', 'target']
        )

        selector.fit(df, 'target', ['key'])

        self.assertTrue(isinstance(
            selector.best_scores_['forecaster'][1], MovingAverageForecaster
        ))
        self.assertEquals(
            selector.best_scores_['forecaster'][1].get_params(),
            {'rolling_kwargs': {'window': 1, 'min_periods': 1}}
        )
        self.assertEqual(selector.best_scores_['score'][1], -1)
        self.assertTrue(isinstance(
            selector.best_scores_['forecaster'][2], MovingMedianForecaster
        ))
        self.assertEquals(
            selector.best_scores_['forecaster'][2].get_params(),
            {'rolling_kwargs': {'window': 3, 'min_periods': 1}}
        )
        self.assertAlmostEqual(selector.best_scores_['score'][2], -8.083333333)

    def test_fit_with_horizon(self) -> type(None):
        """
        Test `fit` method with horizon.

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
        selector = OnTheFlySelector(candidates, horizon=3)

        df = pd.DataFrame(
            [[1, 2],
             [1, 3],
             [1, 6],
             [1, 5],
             [1, 4],
             [1, 1],
             [2, 3],
             [2, 4],
             [2, 4.5],
             [2, 1],
             [2, 9],
             [2, 0]],
            columns=['key', 'target']
        )

        selector.fit(df, 'target', ['key'])

        self.assertTrue(isinstance(
            selector.best_scores_['forecaster'][1], MovingMedianForecaster
        ))
        self.assertEquals(
            selector.best_scores_['forecaster'][1].get_params(),
            {'rolling_kwargs': {'window': 3, 'min_periods': 1}}
        )
        self.assertEquals(selector.best_scores_['score'][1], -3)
        self.assertTrue(isinstance(
            selector.best_scores_['forecaster'][2], MovingMedianForecaster
        ))
        self.assertEquals(
            selector.best_scores_['forecaster'][2].get_params(),
            {'rolling_kwargs': {'window': 3, 'min_periods': 1}}
        )
        self.assertEquals(selector.best_scores_['score'][2], -50 / 3)

    # TODO: Test `fit` with both `horizon` and `n_evaluational_steps`.

    def test_predict(self) -> type(None):
        """
        Test `predict` method.

        :return:
            None
        """
        selector = OnTheFlySelector(horizon=2)
        selector.best_scores_ = pd.DataFrame(
            [[-1, MovingAverageForecaster({'window': 2}).fit(None)],
             [-2, MovingMedianForecaster({'window': 4}).fit(None)]],
            columns=['score', 'forecaster'],
            index=[1, 2]
        )
        selector.best_scores_.index.name = 'key'
        selector.name_of_target_ = 'target'
        selector.scoring_keys_ = ['key']
        selector.series_keys_ = ['key']
        selector.evaluation_fn_ = None

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
        result = selector.predict(df)

        true_answer = pd.DataFrame(
            [[1, 5.5],
             [1, 5.25],
             [2, 3.5],
             [2, 3.75]],
            columns=['key', 'prediction'],
            index=[0, 1, 0, 1]
        )
        self.assertTrue(result.equals(true_answer))


class TestParallelingFunctions(unittest.TestCase):
    """
    Tests of functions from the file named `on_the_fly_selector.py`.
    """

    def test_add_partition_key(self) -> type(None):
        """
        Test `add_partition_key` function.

        :return:
            None
        """
        df = pd.DataFrame(
            [[1, 2],
             [1, 3],
             [2, 6],
             [2, 5],
             [3, 3],
             [3, 4],
             [4, 4.5],
             [4, 1]],
            columns=['key', 'target']
        )
        result = add_partition_key(df, ['key'], n_jobs=3)
        self.assertTrue(result.groupby('partition_key').apply(len).max() == 4)
        self.assertTrue(result.groupby('partition_key').apply(len).min() == 2)


def main():
    test_loader = unittest.TestLoader()
    suites_list = []
    testers = [
        TestOnTheFlySelector(),
        TestParallelingFunctions()
    ]
    for tester in testers:
        suite = test_loader.loadTestsFromModule(tester)
        suites_list.append(suite)
    overall_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner().run(overall_suite)


if __name__ == '__main__':
    main()
