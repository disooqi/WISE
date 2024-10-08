#!./venv python
# -*- coding: utf-8 -*-
"""
http://vos.openlinksw.com/owiki/wiki/VOS/VOSSparqlProtocol#SPARQL%20Service%20Endpoint
http://vos.openlinksw.com/owiki/wiki/VOS/VOSSparqlProtocol#HTTP%20Response%20Formats
"""
__author__ = "Mohamed Eldesouki"
__copyright__ = "Copyright 2020-29, GINA CODY SCHOOL OF ENGINEERING AND COMPUTER SCIENCE, CONCORDIA UNIVERSITY"
__credits__ = ["Mohamed Eldesouki", "Essam Mansour"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "CODS Lab"
__email__ = "cods@eldesouki.ca"
__status__ = "debug"
__date__ = "2020-03-05"

import requests
import logging

formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
logger2 = logging.getLogger("SPARQL logger")
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger2.addHandler(sh)
logger2.setLevel(logging.DEBUG)

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

__all__ = ['_make_keyword_unordered_search_query_with_type', '_make_top_predicates_sbj_query',
           '_make_top_predicates_obj_query', '_evaluate_SPARQL_query',
           '_sparql_query_to_get_predicates_when_subj_and_obj_are_known']


def _make_keyword_search_query_with_type(keywords_string: str, limit=100):
    return f"select ?uri ?label ?type where {{ ?uri ?p ?label . ?label  <bif:contains> '{keywords_string}' . " \
           f"{{ select ?uri (MIN(STR(?auxType)) as ?type) where {{  ?uri {RDF_TYPE}  ?auxType  " \
           f"filter (?auxType != {RDFS_CONCEPT}) }} group by  ?uri }} }} LIMIT {limit}"


def _make_keyword_unordered_search_query_with_type(keywords_string: str, limit=500):
    kws = ' AND '.join(keywords_string.strip().split())
    return f"select distinct ?uri  ?label " \
           f"where {{ ?uri ?p  ?label . ?label  <bif:contains> '{kws}' . }}  LIMIT {limit}"


# def make_keyword_unordered_search_query_with_type(keywords_string, limit=1000):
#     kws = keywords_string.strip().replace(' ', ' AND ')
#     return f"select  ?uri  ?label  ?type  " \
#            f"where {{ ?uri ?p  ?label . ?label  <bif:contains> '{kws}' . " \
#            f"optional {{ select  ?uri  (MIN(STR(?auxType)) as  ?type) " \
#            f"where {{  ?uri  {RDF_TYPE} ?auxType filter (?auxType !=  {RDFS_CONCEPT}) }} " \
#            f"group by  ?uri  }} }} LIMIT {limit}"


def _make_top_predicates_sbj_query(uri, limit=1000):
    return f"select distinct ?p where {{ <{uri}> ?p ?o . }}  LIMIT {limit}"


def _sparql_query_to_get_predicates_when_subj_and_obj_are_known(subj_uri, obj_uri, limit=1000):
    return f"select distinct ?p where {{ <{subj_uri}> ?p <{obj_uri}> . }}  LIMIT {limit}"


def _make_top_predicates_obj_query(uri, limit=1000):
    # return f"select ?p ?p2 where {{ ?s ?p <{uri}> . optional {{ ?s ?p2 ?o }} }} LIMIT {limit}"
    return f"select distinct ?p where {{ ?s ?p <{uri}> . }} LIMIT {limit}"


def _construct_yesno_answers_query(sbj_uri, prd_uri, obj_uri):
    return f"ASK {{ <{sbj_uri}> <{prd_uri}> <{obj_uri}> }}"


def _construct_yesno_answers_query2(sbj_uri, prd_uris, obj_uri):
    disj = list()
    for prd_uri in prd_uris:
        disj.append(f"{{ <{obj_uri}> <{prd_uri}> <{sbj_uri}> }}")
    return f"ASK {{ {' UNION '.join(disj)} }}"


def _construct_answers_query(sub_uri, pred_uri, limit=1000):
    return f"select ?o where {{ <{sub_uri}> <{pred_uri}> ?o . }}  LIMIT {limit}"


def _evaluate_SPARQL_query(query: str, fmt='application/json'):
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
    # logger2.debug(f"[STATUS CODE FOR SPARQL EVAL:] {query_response.status_code}")
    if query_response.ok:
        return query_response.text
    else:
        raise

    # if query_response.status_code in [414]:
    #     return '{"head":{"vars":[]}, "results":{"bindings": []}, "status":414 }'


def _process_SPARQL_query_result(query_response: requests.models.Response):
    pass


if __name__ == '__main__':
    pass
