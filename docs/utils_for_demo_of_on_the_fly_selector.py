"""
This module is for tools that hide too specialized code under
easy-to-read names.

@author: Nikolay Lysenko
"""


import os
import re

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

    @staticmethod
    def which_parameters_does_on_the_fly_selector_have() -> type(None):
        """
        Extract from docstring and type annotations a snippet
        with parameters' descriptions and then show it. 
        
        :return:
            None
        """
        # A string constant that must be absent in docstring.
        reserved_symbol = '~'

        # Extraction of verbal descriptions.
        docstring = on_the_fly_selector.OnTheFlySelector.__doc__
        processed_docstring = docstring.replace('\n        ', reserved_symbol)
        pattern = ':param.*'
        descriptions = re.findall(pattern, processed_docstring)
        descriptions = [
            x.replace(reserved_symbol, '\n        ') for x in descriptions
        ]

        # Extraction of types from type hints.
        type_annotations = {}
        source_path = '/../forecastonishing/selection/on_the_fly_selector.py'
        with open(os.getcwd() + source_path) as sources:
            ready = False
            relevant_lines = []
            for line in sources:
                if 'self' in line:
                    ready = True
                if ready:
                    relevant_lines.append(line)
                if ready and ')' in line:
                    break
            signature_code = ''.join([x.strip() for x in relevant_lines])

            # Escape commas that do not separate class parameters.
            escaped_symbols = []
            level_of_square_brackets = 0
            for symbol in signature_code:
                if symbol == '[':
                    level_of_square_brackets = level_of_square_brackets + 1
                elif symbol == ']':
                    level_of_square_brackets = level_of_square_brackets - 1
                elif symbol == ',' and level_of_square_brackets > 0:
                    symbol = reserved_symbol
                escaped_symbols.append(symbol)
            escaped_code = ''.join(escaped_symbols)

            class_parameters = escaped_code.split(',')
            for parameter_description in class_parameters:
                name_and_type = parameter_description.split(': ')
                if len(name_and_type) == 2:
                    parameter_name = name_and_type[0]
                    parameter_type = name_and_type[1].rstrip('):')
                    parameter_type = (
                        parameter_type.replace(reserved_symbol, ',')
                    )
                    type_annotations[parameter_name] = parameter_type

        # Merging, preparation of the final result.
        result = [
            x.split('\n')[0] +
            ' ' +
            type_annotations[x.split('\n')[0].lstrip(':param ').rstrip(':')] +
            '\n' +
            '\n'.join(x.split('\n')[1:])
            for x in descriptions
        ]
        result = '\n'.join(result)
        print(result)
