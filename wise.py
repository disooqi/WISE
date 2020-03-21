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
__created__ = "2020-03-05"

from pprint import pprint
import os
import re
import json
import operator
import requests
import logging
import numpy as np
from string import punctuation
from collections import defaultdict
from itertools import count, product
from functools import reduce
from statistics import mean
from urllib.parse import urlparse
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
# from bert_serving.client import BertClient
from sparqls import (make_keyword_unordered_search_query_with_type, make_top_predicates_subj_query,
                     make_top_predicates_obj_query, evaluate_SPARQL_query, construct_answers_query,
                     construct_yesno_answers_query, construct_yesno_answers_query2)
from allennlp.predictors.predictor import Predictor
# import rdflib # to construct rdf graph and return its equavilant SPARQL query
# import NetworkX
from question import Question
import embeddings_client as w2v

formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

# LOGGER 1 for Info, Warning and Errors
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('wise.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

# LOGGER 2 for DEBUGGING
logger2 = logging.getLogger("Dos logger")
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger2.addHandler(sh)
logger2.setLevel(logging.DEBUG)

# coreNLP_server_url = 'http://localhost:9000/?properties={"annotators": "ner,openie", "outputFormat": "json"}'
# # bert_server = BertClient(port=5555, port_out=5556)

# ner_predictor2 = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
# oie_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")

coreNLP_server_url = ''
oie_predictor = ''

table = str.maketrans('', '', punctuation)


class Wise:
    """A Natural Language Platform For Querying RDF-Based Graphs

            Usage::
            Usage::

                >>> from wise import Wise
                >>> my_wise = Wise(semantic_afinity_server='127.0.0.1:9600', n_max_answers=10)

            :param semantic_afinity_server: A string, IP and Port for the semantic similarity server of the
            form ``127.0.0.1:9600``.
            :param n_max_answers: An int, the maximum number of result items return by WISE.
            :rtype: A :class:`Wise <Wise>`
            """
    ner = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")
    parser = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    def __init__(self, semantic_afinity_server=None, n_max_answers: int = 100):
        self._ss_server = semantic_afinity_server
        self._n_max_answers = n_max_answers
        self._current_question = None
        self.n_max_Vs = 2
        self.n_max_Es = 3

    @property
    def question(self):
        return self._current_question

    @question.setter
    def question(self, value):
        if value not in Question.answer_types:
            raise ValueError(f"Question should has one of the following types {Question.answer_types}")
        self._current_question = value

    def ask(self, question_text: str, question_id: int = 0, answer_type: str = None, n_max_answers: int =None):
        """WISE pipeline

        Usage::

            >>> from wise import Wise
            >>> my_wise = Wise()
            >>> my_wise.ask("What is the longest river?")


        """
        question = Question(question_text=question_text, question_id=question_id, answer_type=answer_type)
        logger.info(f'\n{"<NEW QUESTION>" * 10}\n[Question:] {question.text},\n')
        self._current_question = question
        self._n_max_answers = n_max_answers if n_max_answers else self._n_max_answers

        self.detect_question_and_answer_type()
        self.rephrase_question()
        self.process_question()
        self.find_possible_noun_phrases_and_relations()
        self.extract_possible_V_and_E()
        self.construct_star_queries()
        self.merge_star_queries()
        # self.evaluate_star_queries()

        answer = ''
        return answer

    def detect_question_and_answer_type(self):
        # question_text = question_text.lower()
        # if question_text.startswith('Who'):
        #     question_text = re.sub('Who', 'Mohamed', question_text)

        properties = ['name', 'capital', 'country']
        # what is the name
        # what country

        if self.question.text.lower().startswith('who was'):
            self.question.question_type = 'person'
            self.question.answer_type = 'resource'
        elif self.question.text.lower().startswith('who is '):
            self.question.question_type = 'person'
            self.question.answer_type = 'resource'
        elif self.question.text.lower().startswith('who are '):
            self.question.question_type = 'person'
            self.question.answer_type = 'list'
        elif self.question.text.lower().startswith('who '):  # Who [V]
            self.question.question_type = 'person'
            self.question.answer_type = 'resource'  # of list
        elif self.question.text.lower().startswith('how many '):
            self.question.question_type = 'count'
            self.question.answer_type = 'number'
        elif self.question.text.lower().startswith('how much '):
            self.question.question_type = 'price'
            self.question.answer_type = 'number'
        elif self.question.text.lower().startswith('when did '):
            self.question.question_type = 'date'
            self.question.answer_type = 'date'
        elif self.question.text.lower().startswith('in which '):  # In which [NNS], In which city
            self.question.question_type = 'person'
            self.question.answer_type = 'resource'  # of list
        elif self.question.text.lower().startswith('which '):  # which [NNS], which actors
            self.question.question_type = 'person'
            self.question.answer_type = 'list'  # of list
        elif self.question.text.lower().startswith('where '):  # where do
            self.question.question_type = 'person'
            self.question.answer_type = 'resource'  # of list
        elif self.question.text.lower().startswith('show '):  # Show ... all
            self.question.question_type = 'person'
            self.question.answer_type = 'list'  # of list
        else:
            pass  # 11,13,75

        logger.info(f'[QUESTION TYPE:] {self.question.question_type}, [ANSWER TYPE:] {self.question.answer_type},\n')

    def rephrase_question(self):
        if self.question.text.lower().startswith('who was'):
            pass
        logger.info(f'[Question Reformulation (Not Impl yet):] {self.question.text},\n')

    def process_question(self):
        parse_components = self.__class__._parse_sentence(self.question.text)
        self.question.parse_components = self.__class__._regroup_named_entities(parse_components)

    def find_possible_noun_phrases_and_relations(self):
        s, pred, o = list(), list(), list()
        # for t in question.parse_components:
        #     pprint(t)
        # TODO look at what other tags from dep parser are considered to identify subject and objects
        positions = list(zip(*self.question.parse_components))[4]
        sbj_has_NE, obj_has_NE = False, False
        for i, w, h, d, p, pos, t in self.question.parse_components:
            if w.lower() in ['how', 'who', 'when', 'what', 'which', 'where']:
                continue
            if w in punctuation or w.lower() in STOPWORDS:
                # TODO: "they" has an indication that the answer is list of people
                continue
            w = w.translate(table)
            if 'subj' in d:
                if t != "O":
                    sbj_has_NE = True
                s.append((i, w, h, d, p, pos, t))
            if 'obj' in d:
                if t != "O":
                    obj_has_NE = True
                o.append((i, w, h, d, p, pos, t))
            if 'subj' not in d and 'obj' not in d:
                pred.append((i, w, h, d, p, pos, t))
            # if t != "O":
            #     print(f"NAMED-ENTITIES: {w}")

        else:
            if sbj_has_NE:
                s = filter(lambda x: x[6] != 'O', s)
            if obj_has_NE:
                o = filter(lambda x: x[6] != 'O', o)
            s = list(map(lambda x: x[1], s))
            o = list(map(lambda x: x[1], o))
            # reduce()
            self.question.possible_subjects = s
            self.question.possible_objects = o

            one_relation = list()
            relations = list()
            idx = count()
            prev_position = None
            for i, w, h, d, p, pos, t in pred:
                c = next(idx)
                if c == 0:
                    prev_position = p
                    one_relation.append(w)
                    continue

                if abs(positions.index(p)-positions.index(prev_position)) == 1:
                    one_relation.append(w)
                else:
                    relations.append(' '.join(one_relation))
                    one_relation.clear()
                    one_relation.append(w)

                prev_position = p
            else:
                rr = ' '.join(one_relation)
                if rr:
                    relations.append(rr)

            self.question.possible_predicates = relations
            logger.info(f'[POSSIBLE SUBJs:]{s}\n')  # TODO rename to possible noun phrase subject
            logger.info(f'[POSSIBLE OBJs:]{o}\n')  # TODO rename to possible noun phrase subject
            logger.info(f'[POSSIBLE RELATIONS:]{relations}\n')

    def extract_possible_V_and_E(self):
        def compute_semantic_similarity_between_single_word_and_word_list(word, word_list):
            scores = list()
            score = 0.0
            for w in word_list:
                try:
                    score = w2v.n_similarity(word.lower().split(), w.lower().split())
                except KeyError:
                    score = 0.0
                finally:
                    scores.append(score)
            else:
                return scores

        noun_phrases = set(self.question.possible_subjects + self.question.possible_objects)
        vertices = defaultdict(dict)
        self.question.noun_phrases = dict()
        for noun_phrase in noun_phrases:
            v_query = make_keyword_unordered_search_query_with_type(noun_phrase, limit=100)
            v_result = json.loads(evaluate_SPARQL_query(v_query))
            v_uris, v_names = self.__class__.extract_resource_name(v_result['results']['bindings'])

            v_scores = compute_semantic_similarity_between_single_word_and_word_list(noun_phrase, v_names)
            v_URIs_with_scores = list(zip(v_uris, v_scores))
            v_URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
            self.question.noun_phrases[noun_phrase] = v_URIs_with_scores[:self.n_max_Vs]
            for v_URI, score in v_URIs_with_scores[:self.n_max_Vs]:
                vertices[v_URI]['score'] = score
                vertices[v_URI]['noun_phrase'] = noun_phrase  # noun phrase not v (please change the variable name)
                vertices[v_URI]['relations'] = dict()

                prd_sparql = make_top_predicates_subj_query(v_URI, limit=100)
                prd_result = json.loads(evaluate_SPARQL_query(prd_sparql))

                prd_uris, prd_names = self.__class__.extract_predicate_names(prd_result['results']['bindings'])

                # TODO you could compute the star query weight from here
                for relation in self.question.possible_predicates:
                    prd_scores = compute_semantic_similarity_between_single_word_and_word_list(relation, prd_names)
                    prd_URIs_with_scores = list(zip(prd_uris, prd_scores))
                    prd_URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
                    vertices[v_URI]['relations'][relation] = prd_URIs_with_scores[:self.n_max_Es]

        logger.info(f'[VERTICES AND EDGES:] {vertices},\n')
        self.question.VsEs = vertices
        logger.info(f'[STAR QUERY COMPONENTS:] {vertices},\n')

    def construct_star_queries(self):
        Vs = self.question.VsEs
        prd_sets = list()
        var_counter = count(1)
        for v_URI, v in Vs.items():
            star_queries = list()
            Es = v['relations'].values()
            possible_prd_combinations = product(*Es)
            for comb in possible_prd_combinations:
                comb_dict = dict(comb)
                star_query_triple_patterns = list()
                prd_set = set(comb_dict.keys())
                if prd_set in prd_sets:
                    continue
                for p_URI in comb_dict.keys():
                    # star_query_triple_patterns.append(f"<{v_URI}> <{p_URI}> ?O{next(var_counter)}")
                    star_query_triple_patterns.append((v_URI, p_URI, f'?O{next(var_counter)}'))
                else:
                    prd_sets.append(prd_set)
                    # triple_patterns = ' . '.join(star_query_triple_patterns)
                    # if not triple_patterns.strip():
                    #     continue
                    # star_query = (f"select * where  {{ {triple_patterns} }}", mean(comb_dict.values()))

                    data_points = comb_dict.values()
                    if data_points:
                        star_queries.append((star_query_triple_patterns, mean(data_points)))
                    else:
                        print("<<<< There is no data points >>>>", star_query_triple_patterns)
            else:
                star_queries.sort(key=operator.itemgetter(1), reverse=True)
                self.question.VsEs[v_URI]['star_queries'] = star_queries
                logger.info(f'[STAR QUERIES <{v_URI}>:] {star_queries},\n')
                # In merging the star queries should come from different Named Entities not from different Vs
        # else:
        #     print(self.question.VsEs)
            # self.question star_queries

    def merge_star_queries(self):
        noun_phrases = self.question.noun_phrases
        possible_star_queries_different_noun_phrases = dict()

        for noun_phrase, v_URIs in noun_phrases.items():
            alternative_star_queries_for_same_NP = dict()
            for v_URI, score in v_URIs:
                alternative_star_queries_for_same_NP[v_URI] = self.question.VsEs[v_URI]['star_queries']
            else:
                possible_star_queries_different_noun_phrases[noun_phrase] = alternative_star_queries_for_same_NP

        # else:
        #     print(possible_star_queries_different_noun_phrases)

        # list of different v_URIs for each noun phrase
        triple_lists = possible_star_queries_different_noun_phrases.values()
        j = list()
        for l in triple_lists:
            j.append(l.keys())
        else:
            f = list(product(*j))
            # print(f)

        final_queries = list()
        x = count(0)
        for y in f:  # for each possible merge
            rpt = set()
            star_queries_from_all_entities = list()
            for v_URI in y:  # get the star_queries for for each v in y
                if v_URI in rpt:
                    continue
                rpt.add(v_URI)
                star_queries = self.question.VsEs[v_URI]['star_queries']
                star_queries_from_all_entities.extend(star_queries)
            else:
                query = Wise._check_if_any_two_star_queries_share_a_predicate(star_queries_from_all_entities)
                final_queries.append(query)
                logger2.debug(query)
                logger.info(f"[FINAL QUERY ({next(x)}):] {query}")

        # for v_URI, v in Vs.items():
        #     noun_phrases[]

    def evaluate_star_queries(self):
        pass

    @staticmethod
    def _check_if_any_two_star_queries_share_a_predicate(star_queries: list):
        prds = defaultdict(list)
        for triples, score in star_queries.copy():
            for triple in triples:
                prds[triple[1]].append((triple[0], triple[2]))

        star_queries_after_merge = list()
        for star_query_triples, score in star_queries.copy():
            star_query = list()
            for triple in star_query_triples:
                flag = False
                for obj_uri in prds[triple[1]]:
                    if obj_uri[0] != triple[0]:
                        flag = True
                        star_query.append(f"<{triple[0]}> <{triple[1]}> <{obj_uri}>")
                else:
                    if not flag:
                        star_query.append(f"<{triple[0]}> <{triple[1]}> {triple[2]}")
            else:
                star_queries_after_merge.append(f"{{ {' . '.join(star_query)} }}")
        else:
            return f"SELECT * WHERE {{ {' UNION '.join(star_queries_after_merge)} }}"

    @classmethod
    def _parse_sentence(cls, sentence: str):
        allannlp_ner_output = cls.ner.predict(sentence=sentence)
        allannlp_dep_output = cls.parser.predict(sentence=sentence)

        words = allannlp_ner_output['words']
        ner_tags = allannlp_ner_output['tags']
        pos_tags = allannlp_dep_output['pos']
        dependencies = allannlp_dep_output['predicted_dependencies']
        heads = allannlp_dep_output['predicted_heads']
        # d = reformat_allennlp_ner_output(ner_tags, words)

        positions = traverse_tree(allannlp_dep_output['hierplane_tree']['root'])
        positions.sort()
        words_info = list(zip(range(1, len(words) + 1), words, heads, dependencies, positions, pos_tags, ner_tags))
        logger.info(f'[QUESTION PARSE COMPONENTS:] {words_info},\n')

        return words_info

    @staticmethod
    def _regroup_named_entities(parse_components):
        l2 = list()
        entity = list()
        flag = False
        tag = ''

        head = None
        dep = None
        poss = list()
        position = None
        for i, w, h, d, p, pos, t in parse_components:
            if t.startswith('B-'):
                flag = True
                tag = t[2:]
                if 'obj' in d or 'subj' in d:
                    head, dep = h, d
                position = p
                poss.append(pos)
                entity.append(w)
            elif flag and t.startswith('I-'):
                if 'obj' in d or 'subj' in d:
                    head, dep = h, d
                poss.append(pos)
                entity.append(w)
            elif flag and t.startswith('L-'):
                if 'obj' in d or 'subj' in d:
                    head, dep = h, d
                poss.append(pos)
                entity.append(w)
                l2.append((i, ' '.join(entity), head, dep, position, ' '.join(poss), tag))
                entity.clear()
                flag = False
            elif t.startswith('U-'):
                l2.append((i, w, h, d, p, pos, t[2:]))
            else:
                l2.append((i, w, h, d, p, pos, t))
        else:
            logger.info(f'[QUESTION PARSE COMPONENTS WITH REGROUPED NAMED ENTITIES:] {l2},\n')
            return l2

    @staticmethod
    def extract_resource_name(result_bindings):
        resource_names = list()
        resource_URIs = list()
        for binding in result_bindings:
            resource_URI = binding['uri']['value']
            uri_path = urlparse(resource_URI).path
            resource_name = os.path.basename(uri_path)
            resource_name = re.sub(r'(:|_|\(|\))', ' ', resource_name)
            # resource_name = re.sub(r'^Category:', '', resource_name)
            # TODO: check for URI validity
            if not resource_name.strip():
                continue
            resource_URIs.append(resource_URI)
            resource_names.append(resource_name)
        return resource_URIs, resource_names

    @staticmethod
    def extract_predicate_names(result_bindings):
        predicate_URIs = list()
        predicate_names = list()
        for binding in result_bindings:
            predicate_URI = binding['p']['value']
            uri_path = urlparse(predicate_URI).path
            predicate_name = os.path.basename(uri_path)
            p = re.compile(r'(_|\([^()]*\))')
            predicate_name = p.sub(' ', predicate_name)
            p2 = re.compile(r'([a-z0-9])([A-Z])')
            predicate_name = p2.sub(r"\1 \2", predicate_name)
            if not predicate_name.strip():
                continue
            predicate_names.append(predicate_name)
            predicate_URIs.append(predicate_URI)
        return predicate_URIs, predicate_names


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


#
# def extract_relations_and_entities(question):
#     entities = list()
#     relations = list()
#     corenlp_output = requests.post(coreNLP_server_url, data=question).json()
#     allannlp_ner_output = ner_predictor.predict(sentence=question)
#     allannlp_oie_output = oie_predictor.predict(sentence=question)
#
#     # seq = ('from', 'O'), ('Bruce Springsteen', 'PERSON'), ('released', 'O'), ....
#     seq = reformat_allennlp_ner_output(allannlp_ner_output['tags'], allannlp_ner_output['words'])
#
#     # AllenNLP entities
#     for mwe in seq:
#         if mwe[1] != 'O':
#             entities.append(mwe[0])
#     else:
#         if not entities:
#             return [], []
#
#     # Stanford entities
#     # if corenlp_output['sentences'][0]['entitymentions']:
#     #     for entity in corenlp_output['sentences'][0]['entitymentions']:
#     #         entities.append(entity['text'])
#     #     else:
#     #         if not entities:
#     #             return [], []
#
#     if corenlp_output['sentences'][0]['openie']:
#         for triple in corenlp_output['sentences'][0]['openie']:
#             if triple['subject'] in entities and triple['object'] in entities:
#                 relations.append((triple['relation'], triple['subject'], triple['object']))
#                 break
#             elif triple['subject'] in entities:
#                 relations.append((triple['relation'], triple['subject']))
#                 break
#             elif triple['object'] in entities:
#                 relations.append((triple['relation'], triple['object']))
#                 break
#         # else:
#         #     print(question)
#         #     pprint(f"Entities {entities} do not appear in the OpenIE triples {corenlp_output['sentences'][0]['openie']}")
#         #     print()
#
#     return entities, relations


# def extract_relations_and_entities2(question):
#     entities = list()
#     relations = list()
#     allannlp_ner_output = ner_predictor.predict(sentence=question)
#     allannlp_oie_output = oie_predictor.predict(sentence=question)
#
#     # seq = ('from', 'O'), ('Bruce Springsteen', 'PERSON'), ('released', 'O'), ....
#     seq = reformat_allennlp_ner_output(allannlp_ner_output['tags'], allannlp_ner_output['words'])
#
#     # AllenNLP entities
#     for mwe in seq:
#         if mwe[1] != 'O':
#             entities.append(mwe[0])
#     else:
#         if not entities:
#             return [], []
#
#     for triple in allannlp_oie_output['verbs']:
#         ff = triple['description']
#
#         print(type(ff))
#
#     return entities, relations
#
# def construct_information_from_KB(entities=None, relations=None):
#     answer = ''
#
#     assert len(relations) <= 1
#     if not relations:
#         return '', ''
#     if len(relations[0]) == 3:
#         prd, sbj, obj = relations[0]
#         prd = remove_stopwords(prd.lower())
#
#         sbj_query = make_keyword_unordered_search_query_with_type(sbj, limit=100)
#         obj_query = make_keyword_unordered_search_query_with_type(obj, limit=100)
#         sbj_result = json.loads(evaluate_SPARQL_query(sbj_query))
#         obj_result = json.loads(evaluate_SPARQL_query(obj_query))
#
#         sbj_uris, sbj_names = extract_resource_name(sbj_result['results']['bindings'])
#         obj_uris, obj_names = extract_resource_name(obj_result['results']['bindings'])
#
#         # sbj_name_vecs = bert_server.encode(sbj_names)
#         # obj_name_vecs = bert_server.encode(obj_names)
#
#         # sbj_vec = bert_server.encode([sbj])[0]
#         # obj_vec = bert_server.encode([obj])[0]
#
#         # compute normalized dot product as score
#         # sbj_scores = np.sum(sbj_vec * sbj_name_vecs, axis=1) / np.linalg.norm(sbj_name_vecs, axis=1)
#         # obj_scores = np.sum(obj_vec * obj_name_vecs, axis=1) / np.linalg.norm(obj_name_vecs, axis=1)
#
#         sbj_scores = []
#         for sbj_name in sbj_names:
#             try:
#                 sim = w2v.n_similarity(sbj.lower().split(), sbj_name.lower().split())
#             except KeyError:
#                 sim = 0.0
#             finally:
#                 sbj_scores.append(sim)
#
#         obj_scores = []
#         for obj_name in obj_names:
#             try:
#                 sim = w2v.n_similarity(obj.lower().split(), obj_name.lower().split())
#             except KeyError:
#                 sim = 0.0
#             finally:
#                 obj_scores.append(sim)
#
#         sbj_URIs_with_scores = list(zip(sbj_uris, sbj_scores))
#         obj_URIs_with_scores = list(zip(obj_uris, obj_scores))
#
#         sbj_URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
#         obj_URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
#
#         logger.info(f'[subject resources:] {sbj_URIs_with_scores[:5]},\n'
#                     f'[object resources:] {obj_URIs_with_scores[:5]}\n')
#         # print(sbj_URIs_with_scores[:1], sbj)
#         # print(obj_URIs_with_scores[:1], obj)
#
#         sbj_predicate_sparql = make_top_predicates_subj_query(sbj_URIs_with_scores[0][0], limit=100)
#         obj_predicate_sparql = make_top_predicates_subj_query(obj_URIs_with_scores[0][0], limit=100)
#         predicate_sbj_result = json.loads(evaluate_SPARQL_query(sbj_predicate_sparql))
#         predicate_obj_result = json.loads(evaluate_SPARQL_query(obj_predicate_sparql))
#
#         prd_sbj_uris, prd_sbj_names = extract_predicate_names(predicate_sbj_result['results']['bindings'])
#         prd_obj_uris, prd_obj_names = extract_predicate_names(predicate_obj_result['results']['bindings'])
#
#         # prd_sbj_name_vecs = bert_server.encode(prd_sbj_names)
#         # prd_obj_name_vecs = bert_server.encode(prd_obj_names)
#
#         # relational_phrase_vec = bert_server.encode([prd])[0]
#         # compute normalized dot product as score
#         # prd_sbj_score = np.sum(relational_phrase_vec * prd_sbj_name_vecs, axis=1) / np.linalg.norm(prd_sbj_name_vecs, axis=1)
#         # prd_obj_score = np.sum(relational_phrase_vec * prd_obj_name_vecs, axis=1) / np.linalg.norm(prd_obj_name_vecs, axis=1)
#
#         prd_sbj_score = []
#         for prd_sbj_name in prd_sbj_names:
#             try:
#                 sim = w2v.n_similarity(prd.lower().split(), prd_sbj_name.lower().split())
#             except KeyError:
#                 sim = 0.0
#             finally:
#                 prd_sbj_score.append(sim)
#
#         prd_obj_score = []
#         for prd_obj_name in prd_obj_names:
#             try:
#                 sim = w2v.n_similarity(prd.lower().split(), prd_obj_name.lower().split())
#             except KeyError:
#                 sim = 0.0
#             finally:
#                 prd_obj_score.append(sim)
#
#         prd_sbj_URIs_with_scores = list(zip(prd_sbj_uris, prd_sbj_score))
#         prd_obj_URIs_with_scores = list(zip(prd_obj_uris, prd_obj_score))
#
#         prd_sbj_URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
#         prd_obj_URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
#
#         logger.info(f'[subject predicates:] {prd_sbj_URIs_with_scores[:10]},\n'
#                     f'[object predicates:] {prd_obj_URIs_with_scores[:10]}\n')
#
#         return list(zip(*prd_sbj_URIs_with_scores))[0][:5], list(zip(*prd_obj_URIs_with_scores))[0][:5], \
#                list(zip(*sbj_URIs_with_scores))[0][:5], list(zip(*obj_URIs_with_scores))[0][:5]
#
#
#     if len(relations[0]) == 2:
#         prd, sbj = relations[0]
#         sbj_query = make_keyword_unordered_search_query_with_type(sbj)
#         sbj_result = json.loads(evaluate_SPARQL_query(sbj_query))
#         sbj_uris, sbj_names = extract_resource_name(sbj_result['results']['bindings'])
#         # sbj_name_vecs = bert_server.encode(sbj_names)
#         # sbj_vec = bert_server.encode([sbj])[0]
#         # compute normalized dot product as score
#         # sbj_scores = np.sum(sbj_vec * sbj_name_vecs, axis=1) / np.linalg.norm(sbj_name_vecs, axis=1)
#
#         sbj_scores = []
#         for sbj_name in sbj_names:
#             try:
#                 sim = w2v.n_similarity(sbj.lower().split(), sbj_name.lower().split())
#             except KeyError:
#                 sim = 0.0
#             finally:
#                 sbj_scores.append(sim)
#
#         sbj_URIs_with_scores = list(zip(sbj_uris, sbj_scores))
#         sbj_URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
#         # print(sbj_URIs_with_scores[:1], sbj)
#
#         sbj_predicate_sparql = make_top_predicates_subj_query(sbj_URIs_with_scores[0][0], limit=100)
#         predicate_sbj_result = json.loads(evaluate_SPARQL_query(sbj_predicate_sparql))
#
#         prd_sbj_uris, prd_sbj_names = extract_predicate_names(predicate_sbj_result['results']['bindings'])
#
#         prd_sbj_score = []
#         for prd_sbj_name in prd_sbj_names:
#             try:
#                 sim = w2v.n_similarity(prd.lower().split(), prd_sbj_name.lower().split())
#             except KeyError:
#                 sim = 0.0
#             finally:
#                 prd_sbj_score.append(sim)
#
#         prd_sbj_URIs_with_scores = list(zip(prd_sbj_uris, prd_sbj_score))
#         prd_sbj_URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
#
#
#     else:
#         return '', ''
#


if __name__ == '__main__':
    pass
