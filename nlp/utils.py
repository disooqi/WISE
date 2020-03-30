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

from string import punctuation
from nltk.corpus import wordnet as wn


nltk_POS_map = {'VB': wn.VERB, 'VBD': wn.VERB, 'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB,
                'JJ': wn.ADJ,
                'NN': wn.NOUN, 'NNS': wn.NOUN,
                'RB': wn.ADV}
table = str.maketrans('', '', punctuation)


def traverse_tree(subtree):
    positions = list()
    positions.append(subtree['spans'][0]['start'])
    if 'children' not in subtree:
        return positions

    for child in subtree['children']:
        ps = traverse_tree(child)
        positions.extend(ps)
    else:
        return positions


def remove_duplicates(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

