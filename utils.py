#!./venv python
# -*- coding: utf-8 -*-
"""
WISE: Natural Language Platform to Query Knowledge bases
"""
__author__ = "Mohamed Eldesouki"
__copyright__ = "Copyright 2020-29, GINA CODY SCHOOL OF ENGINEERING AND COMPUTER SCIENCE, CONCORDIA UNIVERSITY"
__credits__ = ["Mohamed Eldesouki"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "CODS Lab"
__email__ = "mohamed@eldesouki.ca"
__status__ = "debug"
__created__ = "2020-03-30"

from itertools import chain, product


def get_combination_of_two_lists(list1, list2, directed=False, with_reversed=False):
    lists = [l for l in (list1, list2) if l]

    if len(lists) < 2:
        return set(chain(list1, list2))

    combinations = product(*lists, repeat=1)
    combinations_selected = list()
    combinations_memory = list()

    for comb in combinations:
        pair = set(comb)

        if len(lists) == 2 and len(pair) == 1:
            continue

        if not directed and pair in combinations_memory:
            continue
        combinations_memory.append(pair)
        combinations_selected.append(comb)
    else:
        if with_reversed:
            combinations_reversed = [(comb[1], comb[0]) for comb in combinations_selected if len(lists) == 2]
            combinations_selected.extend(combinations_reversed)

        return set(combinations_selected)
