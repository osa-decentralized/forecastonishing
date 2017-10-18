"""
This module is for tools that hide too specialized code under
easy-to-read names.

@author: Nikolay Lysenko
"""


from forecastonishing.selection import on_the_fly_selector


class Helper:
    """
    A collection of static functions.
    """

    @staticmethod
    def why_adaptive_selection() -> type(None):
        """
        Extract from docstring and show a snippet that answers
        the question "Why adaptive selection should be used?".

        :return:
            None
        """
        answer = '\n    '.join(
            on_the_fly_selector.__doc__.split('\n\n')[1].split('\n')
        )
        print('    ' + answer)

    @staticmethod
    def what_is_on_the_fly_selector() -> type(None):
        """
        Extract from docstring and show a snippet that answers
        the question "What is `OnTheFlySelector` class?".

        :return:
            None
        """
        print(on_the_fly_selector.OnTheFlySelector.__doc__.split(':param')[0])
