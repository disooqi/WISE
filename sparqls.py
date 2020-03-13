#!./venv python
# -*- coding: utf-8 -*-
"""
http://vos.openlinksw.com/owiki/wiki/VOS/VOSSparqlProtocol#SPARQL%20Service%20Endpoint
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

import requests


KEYWORD_SEARCH_QUERY_VAR_URI = "uri"
KEYWORD_SEARCH_QUERY_VAR_PRED = "pred"
KEYWORD_SEARCH_QUERY_VAR_OBJ = "obj"
KEYWORD_SEARCH_QUERY_VAR_LABEL = "label"
KEYWORD_SEARCH_QUERY_VAR_TYPE = "type"
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
RDF_SAMEAS = "<http://www.w3.org/2002/07/owl#sameAs>"
RDFS_LABEL = "<http://www.w3.org/2000/01/rdf-schema#label>"
RDFS_CONCEPT = "<http://www.w3.org/2004/02/skos/core#Concept>"
RDFS_PROPERTY = "<http://www.w3.org/2002/07/owl#ObjectProperty>"

some_text = 'Barack Obama'
resource_URI = 'http://dbpedia.org/resource/United_States_Secretary_of_the_Interior'


def make_keyword_search_query_with_type(keywords_string, limit=100):
    return f"select ?uri ?label ?type where {{ ?uri ?p ?label . ?label  <bif:contains> '{keywords_string}' . " \
           f"{{ select ?uri (MIN(STR(?auxType)) as ?type) where {{  ?uri {RDF_TYPE}  ?auxType  " \
           f"filter (?auxType != {RDFS_CONCEPT}) }} group by  ?uri }} }} LIMIT {limit}"


def make_keyword_unordered_search_query_with_type(keywords_string, limit=500):
    kws = keywords_string.strip().replace(' ', ' AND ')
    return f"select  ?uri  ?label " \
           f"where {{ ?uri ?p  ?label . ?label  <bif:contains> '{kws}' . }}  LIMIT {limit}"

# def make_keyword_unordered_search_query_with_type(keywords_string, limit=1000):
#     kws = keywords_string.strip().replace(' ', ' AND ')
#     return f"select  ?uri  ?label  ?type  " \
#            f"where {{ ?uri ?p  ?label . ?label  <bif:contains> '{kws}' . " \
#            f"optional {{ select  ?uri  (MIN(STR(?auxType)) as  ?type) " \
#            f"where {{  ?uri  {RDF_TYPE} ?auxType filter (?auxType !=  {RDFS_CONCEPT}) }} " \
#            f"group by  ?uri  }} }} LIMIT {limit}"


def make_top_predicates_subj_query(uri, limit=1000):
    return f"select distinct ?p where {{ <{uri}> ?p ?o . }}  LIMIT {limit}"


def make_top_predicates_obj_query(uri, limit=1000):
    return f"select ?p ?p2 where {{ ?s ?p <{uri}> . optional {{ ?s ?p2 ?o }} }} LIMIT {limit}"


def construct_yesno_answers_query(sbj_uri, prd_uri, obj_uri):
    # UNION {{ <{obj_uri}> <{prd_uri}> <{sbj_uri}> }} }}
    return f"ASK {{ <{sbj_uri}> <{prd_uri}> <{obj_uri}> }}"


def construct_answers_query(sub_uri, pred_uri, limit=1000):

    return f"select ?o where {{ <{sub_uri}> <{pred_uri}> ?o . }}  LIMIT {limit}"


def evaluate_SPARQL_query(query, fmt='application/json'):
    payload = {
        'default-graph-uri': '',
        'query': query,
        'format': fmt,  # application/rdf+xml
        'CXML_redir_for_subjs': '121',
        'CXML_redir_for_hrefs': '',
        'timeout': '30000',
        'debug': 'on',
        'run': '+Run+Query+',
    }

    query_response = requests.get(f'https://dbpedia.org/sparql', params=payload)
    return query_response.text


if __name__ == '__main__':
    # payload = {
    #     'default-graph-uri': '',
    #     'query': make_top_predicates_obj_query(resource_URI),
    #     'format': 'application/rdf',  # application/rdf+xml
    #     'CXML_redir_for_subjs': '121',
    #     'CXML_redir_for_hrefs': '',
    #     'timeout': '30000',
    #     'debug': 'on',
    #     'run': '+Run+Query+',
    # }
    #
    # x = requests.get(f'https://dbpedia.org/sparql', params=payload)
    # print(x.text)
    pass
