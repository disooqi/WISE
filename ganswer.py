#!./env python
# -*- coding: utf-8 -*-
"""
client.py: python client for gAnswer System through Web service.
"""
import requests

__author__ = "Mohamed Eldesouki"
__copyright__ = "Copyright 2020, CODS Lab, GINA CODY SCHOOL OF ENGINEERING AND COMPUTER SCIENCE, CONCORDIA UNIVERSITY"
__credits__ = ["Mohamed Eldesouki"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "CODS Lab"
__email__ = "mohamed@eldesouki.ca"
__status__ = "debug"
__date__ = "2020-01-22"


def ask_gAnswer(question, ip='206.12.92.208', port='9999', n_max_answer=10, n_max_sparql=10):
    # question = 'Who is the wife of Donald Trump?'
    payload = f'{{maxAnswerNum:{n_max_answer}, maxSparqlNum:{n_max_sparql}, question:{question}}}'
    r = requests.get(f"http://{ip}:{port}/gSolve/?data={payload}")

    return r.text


if __name__ == '__main__':
    pass
