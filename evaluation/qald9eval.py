#!./venv python
# -*- coding: utf-8 -*-
"""
evaluation.py: evaluating WISE online service against QALD-6 benchmark
"""
__author__ = "Mohamed Eldesouki"
__copyright__ = "Copyright 2020, CODS Lab, GINA CODY SCHOOL OF ENGINEERING AND COMPUTER SCIENCE, CONCORDIA UNIVERSITY"
__credits__ = ["Mohamed Eldesouki"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "CODS Lab"
__email__ = "mohamed@eldesouki.ca"
__status__ = "debug"
__created__ = "2020-05-15"


import json
import time
from src.wise import Wise
from termcolor import colored, cprint
from itertools import count
import xml.etree.ElementTree as Et


file_name = r"qald9/qald-9-test-multilingual.json"

# word_vectors = api.load("wiki-news-300d-1m")  # load pre-trained word-vectors from gensim-data
if __name__ == '__main__':
    root_element = Et.Element('dataset')
    root_element.set('id', 'dbpedia-test')
    author_comment = Et.Comment(f'created by mohamed@eldesouki.ca')
    root_element.append(author_comment)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    the_39_questions = list()
    total_time = 0
    with open(file_name) as f:
        qald6_testset = json.load(f)
    dataset_id = qald6_testset['dataset']['id']
    counter = count(1)
    wise_qald6 = {"dataset": {"id": "qald-9-test-multilingual"}, "questions": []}
    WISE = Wise()
    excluded = ['167']
    start = True
    for i, question in enumerate(qald6_testset['questions']):
        if not start:
            if question['id'] == '167':
                start = True
            continue


        # if question['id'] not in ['167']:
        #     continue
        if question['id'] in excluded:
            continue
        qc = next(counter)

        # question_text = ''
        for language_variant_question in question['question']:
            if language_variant_question['language'] == 'en':
                question_text = language_variant_question['string'].strip()
                # print(question['id'], ', ', f'"{question["query"]["sparql"]}"')

                text = colored(f"[PROCESSING: ] Question count: {qc}, ID {question['id']}  >>> {question_text}", 'blue',
                               attrs=['reverse', 'blink'])
                cprint(f"== {text}  ")
                break

        st = time.time()
        # question_text = 'Which movies starring Brad Pitt were directed by Guy Ritchie?'
        # question_text = 'When did the Boston Tea Party take place and led by whom?'
        # question_text = 'Who was the doctoral supervisor of Albert Einstein?'
        answers = WISE.ask(question_id=question["id"], question_text=question_text, answer_type=question['answertype'],
                           n_max_answers=39, merge_answers=True)
        # answers = ''
        if answers:
            wise_qald6["questions"].append(answers)
        else:
            question['answers'] = [{"head": {"vars": ["uri"]}, "results": {"bindings": []}}]
            wise_qald6["questions"].append(question)

        et = time.time()
        total_time = total_time + (et - st)
        text = colored(f'[DONE!! in {et-st:.2f} SECs]', 'green', attrs=['bold', 'reverse', 'blink', 'dark'])
        # cprint(f"== {text} ==")

    text1 = colored(f'total_time = [{total_time:.2f} sec]', 'yellow', attrs=['reverse', 'blink'])
    text2 = colored(f'avg time = [{total_time / qc:.2f} sec]', 'yellow', attrs=['reverse', 'blink'])
    cprint(f"== QALD 9 Statistics : {qc} questions, Total Time == {text1}, Average Time == {text2} ")

    with open(f'output/WISE_result_{timestr}.json', encoding='utf-8', mode='w') as rfobj:
        json.dump(wise_qald6, rfobj)
        rfobj.write('\n')

