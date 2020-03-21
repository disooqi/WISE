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
from wise import Wise
from pprint import pprint
# import gensim.downloader as api
from question import Question
from termcolor import colored, cprint

the_39_question_ids = (1, 3, 8, 9, 11, 13, 14, 15, 16, 17, 21, 23, 24, 26, 27, 28, 30, 31, 33, 35, 37, 39, 40, 41, 43,
                       47, 54, 56, 61, 62, 64, 68, 75, 83, 85, 92, 93, 96, 99)
file_name = r"qald6/qald-6-test-multilingual.json"

# word_vectors = api.load("wiki-news-300d-1m")  # load pre-trained word-vectors from gensim-data


if __name__ == '__main__':
    the_39_questions = list()

    with open(file_name) as f:
        qald6_testset = json.load(f)
    dataset_id = qald6_testset['dataset']['id']
    WISE = Wise()
    for question in qald6_testset['questions']:
        question_id = question['id']
        answer_type = question['answertype']
        question_text = ''
        for language_variant_question in question['question']:
            if language_variant_question['language'] == 'en':
                question_text = language_variant_question['string'].strip()
                break
        if question_id in the_39_question_ids:
            st = time.time()
            # question_text = 'Which movies starring Brad Pitt were directed by Guy Ritchie?'
            answer = WISE.ask(question_text=question_text, answer_type=answer_type)
            et = time.time()
            text = colored(f'[{et-st:.2f} sec]', 'yellow', attrs=['reverse', 'blink'])
            cprint(f"{question_text} {text}")












