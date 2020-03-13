#!./venv python
# -*- coding: utf-8 -*-
"""
evaluation.py: evaluating WISE online service against QALD-3 benchmark
"""
__author__ = "Mohamed Eldesouki"
__copyright__ = "Copyright 2020, CODS Lab, GINA CODY SCHOOL OF ENGINEERING AND COMPUTER SCIENCE, CONCORDIA UNIVERSITY"
__credits__ = ["Mohamed Eldesouki"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "CODS Lab"
__email__ = "mohamed@eldesouki.ca"
__status__ = "debug"
__created__ = "2020-03-11"

import json
import time
import xml.etree.ElementTree as Et
from xml.dom.minidom import parse
from ganswer import ask_gAnswer
from wise import ask_wise
import gensim.downloader as api


word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
if __name__ == '__main__':
    q3 = 'Did Socrates influence Aristotle?'
    q1 = 'What is the capital of Cameroon?'
    q2 = 'Who wrote Harry Potter?'
    q4 = 'Is Michelle Obama the wife of Barack Obama?'

    # answer = ask_wise(q3, n_max_answer=1000)
    answer = ask_wise(q4, word_vectors, n_max_answer=1000)
    print(answer)


