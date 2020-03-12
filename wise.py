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
__date__ = "2020-03-05"

from pprint import pprint
import os
import re
import json
import operator
import requests
import numpy as np
from urllib.parse import urlparse
from bert_serving.client import BertClient
from sparqls import (make_keyword_unordered_search_query_with_type, make_top_predicates_subj_query,
                     make_top_predicates_obj_query, evaluate_SPARQL_query, construct_answers_query)
# import rdflib # to construct rdf graph and return its equavilant SPARQL query
# import NetworkX


coreNLP_server_url = 'http://localhost:9000/?properties={"annotators": "ner,openie", "outputFormat": "json"}'
bert_server = BertClient(port=5555, port_out=5556)

def ask_wise(question, n_max_answer=10):
    """WISE pipeline"""
    print(question)
    question_type = detect_question_type(question)
    sentence = rephrase_question(question, question_type)
    entities, relations = extract_relations_and_entities(sentence)
    sbj_uri, prd_uri = construct_information_from_KB(entities=entities, relations=relations)

    answers_query = construct_answers_query(sbj_uri, prd_uri, limit=100)
    # print("answers_query", answers_query)
    result_text = evaluate_SPARQL_query(answers_query, fmt='application/json')
    result = json.loads(result_text)
    print(result_text)
    print('\/'*80)

    return f'{{"question": "{question}"}}'


def detect_question_type(question_text):
    question_types = ['yesno', 'list', 'count']
    return question_types[0]


def rephrase_question(question_text, question_type):
    return question_text


def extract_relations_and_entities(question):
    # print(f"Recognize triples from Question test using Stanford OpenIE")

    entities = list()
    relations = list()
    corenlp_output = requests.post(coreNLP_server_url, data=question).json()

    if corenlp_output['sentences'][0]['entitymentions']:
        for entity in corenlp_output['sentences'][0]['entitymentions']:
            entities.append(entity['text'])
        else:
            if not entities:
                return [], []
    if corenlp_output['sentences'][0]['openie']:
        for triple in corenlp_output['sentences'][0]['openie']:
            if triple['subject'] in entities and triple['object'] in entities:
                relations.append((triple['relation'], triple['subject'], triple['object']))
                break
            elif triple['subject'] in entities:
                relations.append((triple['relation'], triple['subject']))
                break
            elif triple['object'] in entities:
                relations.append((triple['relation'], triple['object']))
                break
        # else:
        #     print(question)
        #     pprint(f"Entities {entities} do not appear in the OpenIE triples {corenlp_output['sentences'][0]['openie']}")
        #     print()

    return entities, relations


def construct_information_from_KB(entities=None, relations=None):
    answer = ''

    predicate_URIs = list()
    predicate_names = list()
    assert len(relations) <= 1
    if not relations:
        return '', ''
    # if len(relations[0]) == 3:
    #     prd, sbj, obj = relations[0]
    #     sbj_query = make_keyword_unordered_search_query_with_type(sbj)
    #     obj_query = make_keyword_unordered_search_query_with_type(obj)
    #     sbj_result = json.loads(evaluate_SPARQL_query(sbj_query))
    #     obj_result = json.loads(evaluate_SPARQL_query(obj_query))
    #
    #     sbj_uris, sbj_names = extract_resource_name(sbj_result['results']['bindings'])
    #     obj_uris, obj_names = extract_resource_name(obj_result['results']['bindings'])
    #
    #     sbj_name_vecs = bert_server.encode(sbj_names)
    #     obj_name_vecs = bert_server.encode(obj_names)
    #
    #     sbj_vec = bert_server.encode([sbj])[0]
    #     obj_vec = bert_server.encode([obj])[0]
    #
    #     # compute normalized dot product as score
    #     sbj_scores = np.sum(sbj_vec * sbj_name_vecs, axis=1) / np.linalg.norm(sbj_name_vecs, axis=1)
    #     obj_scores = np.sum(obj_vec * obj_name_vecs, axis=1) / np.linalg.norm(obj_name_vecs, axis=1)
    #
    #     sbj_URIs_with_scores = list(zip(sbj_uris, sbj_scores))
    #     obj_URIs_with_scores = list(zip(obj_uris, obj_scores))
    #
    #     sbj_URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
    #     obj_URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
    #
    #     # print(sbj_URIs_with_scores[:1], sbj)
    #     # print(obj_URIs_with_scores[:1], obj)
    #
    #     sbj_predicate_sparql = make_top_predicates_subj_query(sbj_URIs_with_scores[0][0], limit=100)
    #     obj_predicate_sparql = make_top_predicates_subj_query(obj_URIs_with_scores[0][0], limit=100)
    #     predicate_sbj_result = json.loads(evaluate_SPARQL_query(sbj_predicate_sparql))
    #     predicate_obj_result = json.loads(evaluate_SPARQL_query(obj_predicate_sparql))
    #
    #     for binding in predicate_obj_result['results']['bindings']:
    #         pprint(binding)

    if len(relations[0]) == 2:
        prd, sbj = relations[0]
        sbj_query = make_keyword_unordered_search_query_with_type(sbj)
        sbj_result = json.loads(evaluate_SPARQL_query(sbj_query))
        sbj_uris, sbj_names = extract_resource_name(sbj_result['results']['bindings'])
        sbj_name_vecs = bert_server.encode(sbj_names)
        sbj_vec = bert_server.encode([sbj])[0]
        # compute normalized dot product as score
        sbj_scores = np.sum(sbj_vec * sbj_name_vecs, axis=1) / np.linalg.norm(sbj_name_vecs, axis=1)
        sbj_URIs_with_scores = list(zip(sbj_uris, sbj_scores))
        sbj_URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
        # print(sbj_URIs_with_scores[:1], sbj)
        sbj_predicate_sparql = make_top_predicates_subj_query(sbj_URIs_with_scores[0][0], limit=100)
        predicate_sbj_result = json.loads(evaluate_SPARQL_query(sbj_predicate_sparql))

        for binding in predicate_sbj_result['results']['bindings']:
            predicate_URI = binding['p']['value']
            predicate_URIs.append(predicate_URI)
            uri_path = urlparse(predicate_URI).path
            predicate_name = os.path.basename(uri_path)
            p = re.compile(r'(_|\([^()]*\))')
            predicate_name = p.sub(' ', predicate_name)
            p2 = re.compile(r'([a-z0-9])([A-Z])')
            predicate_name = p2.sub(r"\1 \2", predicate_name)
            predicate_names.append(predicate_name)
        else:
            predicate_name_vecs = bert_server.encode(predicate_names)
            relational_phrase_vec = bert_server.encode([prd])[0]
            # compute normalized dot product as score
            score = np.sum(relational_phrase_vec * predicate_name_vecs, axis=1) / np.linalg.norm(predicate_name_vecs,
                                                                                                 axis=1)
            topk_idx = np.argsort(score)[::-1][:10]
            for idx in topk_idx:

                print(f'> {score[idx]} {predicate_URIs[idx]} {predicate_names[idx]}')
                return sbj_URIs_with_scores[0][0], predicate_names[idx]
    else:
        return '', ''


def extract_resource_name(result_bindings):
    resource_names = list()
    resource_URIs = list()
    for binding in result_bindings:
        resource_URI = binding['uri']['value']
        resource_URIs.append(resource_URI)
        uri_path = urlparse(resource_URI).path
        resource_name = os.path.basename(uri_path)
        p = re.compile(r'(_|\([^()]*\))')
        resource_name = p.sub(' ', resource_name)
        resource_names.append(resource_name)
    return resource_URIs, resource_names


if __name__ == '__main__':
    pass
