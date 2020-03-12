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
__date__ = "2020-03-10"

import json
import time
import xml.etree.ElementTree as Et
from xml.dom.minidom import parse
from ganswer import ask_gAnswer
from wise import ask_wise


def handle_question(question):
    # question_en = question.getElementsByTagName('string')
    for q in question.getElementsByTagName('string'):
        # if element.getAttribute('A') == "1":
        if q.getAttribute('lang') == "en":
            # print([n for n in q.childNodes if n.nodeType == q.CDATA_SECTION_NODE][0])
            the_question = q.firstChild.data
            # answer = ask_gAnswer(the_question, n_max_answer=1000, n_max_sparql=1000)
            answer = ask_wise(the_question, n_max_answer=1000)
            return answer
            # print(q.firstChild.data, type(q.firstChild.data))


def handle_dbpedia_questions(dbpedia_questions):
    questions = dbpedia_questions.getElementsByTagName("question")
    root_element = Et.Element('dataset')
    root_element.set('id', 'dbpedia-test')

    author_comment = Et.Comment(f'created by mohamed@eldesouki.ca')
    root_element.append(author_comment)
    with open('output/WISE_result_00.jsons', encoding='utf-8', mode='w') as rfobj:
        for i, question in enumerate(questions):
            if question.attributes["id"].value != '99':
                continue
            answer = json.loads(handle_question(question))
            answer['id'] = question.attributes["id"].value
            answer['answertype'] = question.attributes['answertype'].value

            # response_json = json.loads(line.strip())
            question_element = Et.SubElement(root_element, 'question', id=question.attributes["id"].value)
            # with open(f'{question.attributes["id"].value}', mode='w') as fo:
            #     fo.write(handle_question_2(question))
            # continue
            # question_element.set('id', response_json['id'])
            Et.SubElement(question_element, 'string', lang="en").text = f'![CDATA[{answer["question"]}]]'
            answers = Et.SubElement(question_element, 'answers')
            results = answer.get('results', None)
            # break
            if not results:
                # print(answer)
                continue
            # bindings = results.get('bindings', None)
            for answer in results['bindings']:
                for k, v in answer.items():
                    answer_element = Et.SubElement(answers, 'answer')
                    if question.attributes['answertype'].value == 'resource':
                        Et.SubElement(answer_element, 'uri').text = f'http://dbpedia.org/resource/{v["value"][1:-1]}'
                    elif question.attributes['answertype'].value == 'number':
                        Et.SubElement(answer_element, 'number').text = v["value"][1:-1]
                    elif question.attributes['answertype'].value == 'date':
                        Et.SubElement(answer_element, 'date').text = v["value"][1:-1]
                    elif question.attributes['answertype'].value == 'boolean':
                        Et.SubElement(answer_element, 'boolean').text = v["value"][1:-1]  # True|False
                    elif question.attributes['answertype'].value == 'string':
                        Et.SubElement(answer_element, 'string').text = v["value"][1:-1]
            json.dump(answer, rfobj)
            rfobj.write('\n')
            # time.sleep(5)
            # if i == 1:
            #     break
        else:
            tree = Et.ElementTree(root_element)
            tree.write("output/WISE_00.xml")


if __name__ == '__main__':
    with parse(r'qald3/dbpedia-test-questions.xml') as dom:
        handle_dbpedia_questions(dom)
