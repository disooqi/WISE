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
from termcolor import colored, cprint
from itertools import count
import xml.etree.ElementTree as Et

the_39_question_ids = (1, 3, 8, 9, 11, 13, 14, 15, 16, 17, 21, 23, 24, 26, 27, 28, 30, 31, 33, 35, 37, 39, 40, 41, 43,
                       47, 54, 56, 61, 62, 64, 68, 75, 83, 85, 92, 93, 96, 99)
file_name = r"qald6/qald-6-test-multilingual.json"

# word_vectors = api.load("wiki-news-300d-1m")  # load pre-trained word-vectors from gensim-data


if __name__ == '__main__':
    root_element = Et.Element('dataset')
    root_element.set('id', 'dbpedia-test')
    author_comment = Et.Comment(f'created by mohamed@eldesouki.ca')
    root_element.append(author_comment)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    the_39_questions = list()

    with open(file_name) as f:
        qald6_testset = json.load(f)
    dataset_id = qald6_testset['dataset']['id']
    WISE = Wise()
    count39 = count(1)
    for i, question in enumerate(qald6_testset['questions']):
        if question['id'] not in the_39_question_ids:
            continue
        qc = next(count39)
        if qc > 1:
            break
        print(f"== Question count: {i}, ID {question['id']}  == ")
        # if question.attributes["id"].value != '81':
        #     continue

        # question_text = ''
        for language_variant_question in question['question']:
            if language_variant_question['language'] == 'en':
                question_text = language_variant_question['string'].strip()
                break

        st = time.time()
        # question_text = 'Which movies starring Brad Pitt were directed by Guy Ritchie?'
        answers = WISE.ask(question_text=question_text, answer_type=question['answertype'], n_max_answers=10)
        answer = answers[0]

        answer['id'] = question['id']
        answer['answertype'] = question['answertype']
        question_element = Et.SubElement(root_element, 'question', id=str(question['id']))

        Et.SubElement(question_element, 'string', lang="en").text = f'![CDATA[{answer["question"]}]]'
        Et.SubElement(question_element, 'query').text = f"![CDATA[{answer['sparql']}]]"
        # Et.SubElement(question_element, 'sparql').text = f"![CDATA[{answer['sparql']}]]"
        answers = Et.SubElement(question_element, 'answers')
        results = answer.get('results', None)
        if not results:
            continue
        for answer in results['bindings']:
            for k, v in answer.items():
                answer_element = Et.SubElement(answers, 'answer')
                if question['answertype'] == 'resource':
                    Et.SubElement(answer_element, 'uri').text = v["value"]
                elif question['answertype'] == 'number':
                    Et.SubElement(answer_element, 'number').text = v["value"]
                elif question['answertype'] == 'date':
                    Et.SubElement(answer_element, 'date').text = v["value"]
                elif question['answertype'] == 'boolean':
                    Et.SubElement(answer_element, 'boolean').text = v["value"]  # True|False
                elif question['answertype'] == 'string':
                    Et.SubElement(answer_element, 'string').text = v["value"]

        et = time.time()
        text = colored(f'[{et-st:.2f} sec]', 'yellow', attrs=['reverse', 'blink'])
        cprint(f"{question_text} {text}")

    tree = Et.ElementTree(root_element)
    tree.write(f"output/WISE_qald6_{timestr}.xml")















