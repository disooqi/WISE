#!./venv python
# -*- coding: utf-8 -*-
"""
WISE: Natural Language Platform to Query Knowledge bases
"""
__author__ = "Mohamed Eldesouki"
__copyright__ = "Copyright 2020-29, GINA CODY SCHOOL OF ENGINEERING AND COMPUTER SCIENCE, CONCORDIA UNIVERSITY"
__credits__ = ["Mohamed Eldesouki", "Essam Mansour"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "CODS Lab"
__email__ = "cods@eldesouki.ca"
__status__ = "debug"
__created__ = "2020-03-05"

import os
import re
import json
import operator
import logging
from collections import defaultdict
from itertools import count, product, zip_longest
from urllib.parse import urlparse
from .sparqls import *
from .question import Question
from .nlp.utils import remove_duplicates
from . import embeddings_client as w2v, utils

formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

# LOGGER 1 for Info, Warning and Errors
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('wise.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# LOGGER 2 for DEBUGGING
logger2 = logging.getLogger("Dos logger")
# if not logger.handlers:
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger2.addHandler(sh)
logger2.setLevel(logging.DEBUG)

__all__ = ['Wise']


class Wise:
    """A Natural Language Platform For Querying RDF-Based Graphs

            Usage::

                >>> from wise import Wise
                >>> my_wise = Wise(semantic_affinity_server='127.0.0.1:9600', n_max_answers=10)

            :param semantic_affinity_server: A string, IP and Port for the semantic similarity server of the
            form ``127.0.0.1:9600``.
            :param n_max_answers: An int, the maximum number of result items return by WISE.
            :rtype: A :class:`Wise <Wise>`
            """

    def __init__(self, semantic_affinity_server=None, n_max_answers: int = 100):
        self._ss_server = semantic_affinity_server
        self._n_max_answers = n_max_answers  # this should affect the number of star queries to be executed against TS
        self.__question = None
        self.n_max_Vs = 2
        self.n_max_Es = 3
        self.v_uri_scores = None

    def ask(self, question_text: str, question_id: int = 0, answer_type: str = None, n_max_answers: int = None,
            answer_format: str = 'qald', merge_answers: bool = False, n_max_query_eval: int = 10):
        """WISE pipeline

        Usage::

            >>> from wise import Wise
            >>> my_wise = Wise()
            >>> my_wise.ask("What is the longest river?")

        :param n_max_query_eval:
        :param question_id:
        :param answer_type:
        :param merge_answers:
        :param question_text: A string, the question to be answered by WISE.
        :param n_max_answers: An int, the maximum number of result items return by WISE.
        :param answer_format: A string, format of answers return by WISE. values: "qald", "list", "string",
        "bool", "number".
        :rtype: A :class:`dict <dict>`
        """
        self.v_uri_scores = defaultdict(float)
        self.__question = Question(question_id=question_id, question_text=question_text)

        if answer_type:
            self.__question.answer_datatype = answer_type
        self._n_max_answers = n_max_answers if n_max_answers else self._n_max_answers
        self.__detect_question_and_answer_type()
        self.__rephrase_question()
        # if no named entity you should return here
        if len(self.__question.query_graph) == 0 or not self.__question.query_graph.edges:
            logger.info("[NO Named-entity or NO Relation Detected]")
            return []
        self.__extract_possible_V_and_E()
        self.__generate_star_queries()
        self.__evaluate_star_queries()

        answers = self.__question.format_answers(merge_answers)[:self._n_max_answers]

        logger.info(f"\n\n\n{'#' * 120}")
        return answers

    def __detect_question_and_answer_type(self):
        # question_text = question_text.lower()
        # if question_text.startswith('Who'):
        #     question_text = re.sub('Who', 'Mohamed', question_text)

        properties = ['name', 'capital', 'country']
        # what is the name
        # what country

        if self.__question.text.lower().startswith('who was'):
            self.__question.answer_type = 'person'
            self.__question.answer_datatype = 'resource'
            # self.question.add_entity('var', question_type=self.question.answer_type)
        elif self.__question.text.lower().startswith('who is '):
            self.__question.answer_type = 'person'
            self.__question.answer_datatype = 'resource'
        elif self.__question.text.lower().startswith('who are '):
            self.__question.answer_type = 'person'
            self.__question.answer_datatype = 'list'
        elif self.__question.text.lower().startswith('who '):  # Who [V]
            self.__question.answer_type = 'person'
            self.__question.answer_datatype = 'resource'  # of list
        elif self.__question.text.lower().startswith('how many '):
            self.__question.answer_type = 'count'
            self.__question.answer_datatype = 'number'
        elif self.__question.text.lower().startswith('how much '):
            self.__question.answer_type = 'price'
            self.__question.answer_datatype = 'number'
        elif self.__question.text.lower().startswith('when did '):
            self.__question.answer_type = 'date'
            self.__question.answer_datatype = 'date'
        elif self.__question.text.lower().startswith('in which '):  # In which [NNS], In which city
            self.__question.answer_type = 'person'
            self.__question.answer_datatype = 'resource'  # of list
        elif self.__question.text.lower().startswith('which '):  # which [NNS], which actors
            self.__question.answer_type = 'person'
            self.__question.answer_datatype = 'list'  # of list
        elif self.__question.text.lower().startswith('where '):  # where do
            self.__question.answer_type = 'person'
            self.__question.answer_datatype = 'resource'  # of list
        elif self.__question.text.lower().startswith('show '):  # Show ... all
            self.__question.answer_type = 'person'
            self.__question.answer_datatype = 'list'  # of list
        else:
            pass  # 11,13,75

    def __rephrase_question(self):
        if self.__question.text.lower().startswith('who was'):
            pass

        # logger.info(f'[Question Reformulation (Not Impl yet):] {self.question.text},\n')

    def __extract_possible_V_and_E(self):
        for entity in self.__question.query_graph:
            if entity == 'var':
                self.__question.query_graph.add_node(entity, uris=[], answers=[])
                continue
            entity_query = _make_keyword_unordered_search_query_with_type(entity, limit=100)

            try:
                # TODO: ISSUE #1
                entity_result = json.loads(_evaluate_SPARQL_query(entity_query))
            except:
                logger.error(f"Error at 'extract_possible_V_and_E' method with v_query value of {entity_query} ")
                continue
            # TODO: What if V = {} for some NE; should we remove it from graph or keep it and deal with graph as it is?!
            uris, names = self.__class__.__extract_resource_name(entity_result['results']['bindings'])
            scores = self.__compute_semantic_similarity_between_single_word_and_word_list(entity, names)

            URIs_with_scores = list(zip(uris, scores))
            URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
            self.v_uri_scores.update(URIs_with_scores)
            URIs_sorted = list(zip(*URIs_with_scores))[0]
            URIs_chosen = remove_duplicates(URIs_sorted)[:self.n_max_Vs]
            self.__question.query_graph.nodes[entity]['uris'].extend(URIs_chosen)
        # else:
        #     logger2.debug(f"[NODES]")

        # Find E for all relations
        for (source, destination, key, relation) in self.__question.query_graph.edges(data='relation', keys=True):
            source_URIs = self.__question.query_graph.nodes[source]['uris']
            destination_URIs = self.__question.query_graph.nodes[destination]['uris']
            combinations = utils.get_combination_of_two_lists(source_URIs, destination_URIs, with_reversed=False)

            uris, names = list(), list()
            for comb in combinations:
                if source == 'var' or destination == 'var':
                    URIs_false, names_false = self.__get_predicates_and_their_names(subj=comb)
                    URIs_true, names_true = self.__get_predicates_and_their_names(obj=comb)
                else:
                    v_uri_1, v_uri_2 = comb
                    URIs_false, names_false = self.__get_predicates_and_their_names(v_uri_1, v_uri_2)
                    URIs_true, names_true = self.__get_predicates_and_their_names(v_uri_2, v_uri_1)
                URIs_false = list(zip_longest(URIs_false, [False], fillvalue=False))
                URIs_true = list(zip_longest(URIs_true, [True], fillvalue=True))
                uris.extend(URIs_false + URIs_true)
                names.extend(names_false + names_true)
            else:
                URIs_chosen = self.__get_chosen_URIs_for_relation(relation, uris, names)
                self.__question.query_graph[source][destination][key]['uris'].extend(URIs_chosen)
        else:
            logger.info(f"[GRAPH NODES WITH URIs:] {self.__question.query_graph.nodes(data=True)}")
            logger.info(f"[GRAPH EDGES WITH URIs:] {self.__question.query_graph.edges(data=True)}")

    @staticmethod
    def __compute_semantic_similarity_between_single_word_and_word_list(word, word_list):
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

    def __get_chosen_URIs_for_relation(self, relation: str, uris: list, names: list):
        if not uris:
            return uris

        scores = self.__class__.__compute_semantic_similarity_between_single_word_and_word_list(relation, names)
        # (uri, True) ===>  (uri, True, score)
        l1, l2 = list(zip(*uris))
        URIs_with_scores = list(zip(l1, l2, scores))
        URIs_with_scores.sort(key=operator.itemgetter(2), reverse=True)
        # self.uri_scores.update(URIs_with_scores)
        return remove_duplicates(URIs_with_scores)[:self.n_max_Es]

    def __generate_star_queries(self):
        possible_triples_for_all_relations = list()
        for source, destination, key, relation_uris in self.__question.query_graph.edges(data='uris', keys=True):
            source_URIs = self.__question.query_graph.nodes[source]['uris']
            destination_URIs = self.__question.query_graph.nodes[destination]['uris']
            node_uris = source_URIs if destination == 'var' else destination_URIs

            possible_triples_for_single_relation = utils.get_combination_of_two_lists(node_uris, relation_uris)
            possible_triples_for_all_relations.append(possible_triples_for_single_relation)
        else:
            for star_query in product(*possible_triples_for_all_relations):
                score = sum([self.v_uri_scores[subj] + predicate[2] for subj, predicate in star_query])

                triple = [f'?var <{predicate[0]}> <{v_uri}>' if predicate[1] else f'<{v_uri}> <{predicate[0]}> ?var'
                          for v_uri, predicate in star_query]

                query = f"SELECT * WHERE {{ {' . '.join(triple)} }}"
                self.__question.add_possible_answer(sparql=query, score=score)

    def __evaluate_star_queries(self):
        self.__question.possible_answers.sort(reverse=True)
        for i, possible_answer in enumerate(self.__question.possible_answers[:self._n_max_answers]):  # _n_max_eval
            logger.info(f"[EVALUATING SPARQL:] {possible_answer.sparql}")
            result = _evaluate_SPARQL_query(possible_answer.sparql)
            possible_answer.update_answers_element(result)

    @staticmethod
    def __extract_resource_name(result_bindings):
        resource_names = list()
        resource_URIs = list()
        for binding in result_bindings:
            resource_URI = binding['uri']['value']
            uri_path = urlparse(resource_URI).path
            resource_name = os.path.basename(uri_path)
            dir_name = os.path.dirname(uri_path)
            if resource_name.startswith('Category:') or not dir_name.endswith('/resource'):
                continue
            resource_name = re.sub(r'(:|_|\(|\))', ' ', resource_name)
            # resource_name = re.sub(r'^Category:', '', resource_name)
            # TODO: check for URI validity
            if not resource_name.strip():
                continue
            resource_URIs.append(resource_URI)
            resource_names.append(resource_name)
        return resource_URIs, resource_names

    @staticmethod
    def __extract_resource_name_from_uri(uri: str):
        resource_URI = uri
        uri_path = urlparse(resource_URI).path
        resource_name = os.path.basename(uri_path)
        resource_name = re.sub(r'(:|_|\(|\))', ' ', resource_name)
        # resource_name = re.sub(r'^Category:', '', resource_name)
        # TODO: check for URI validity
        return resource_URI, resource_name

    @staticmethod
    def __extract_predicate_names(result_bindings):
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

    @staticmethod
    def __get_predicates_and_their_names(subj=None, obj=None):
        if subj and obj:
            q = _sparql_query_to_get_predicates_when_subj_and_obj_are_known(subj, obj, limit=100)
            uris, names = Wise.__execute_sparql_query_and_get_uri_and_name_lists(q)
        elif subj:
            q = _make_top_predicates_sbj_query(subj, limit=100)
            uris, names = Wise.__execute_sparql_query_and_get_uri_and_name_lists(q)
        elif obj:
            q = _make_top_predicates_obj_query(obj, limit=100)
            uris, names = Wise.__execute_sparql_query_and_get_uri_and_name_lists(q)
        else:
            raise Exception

        return uris, names

    @staticmethod
    def __execute_sparql_query_and_get_uri_and_name_lists(q):
        result = json.loads(_evaluate_SPARQL_query(q))
        return Wise.__extract_predicate_names(result['results']['bindings'])


if __name__ == '__main__':
    my_wise = Wise()
    my_wise.ask(question_text='Which movies starring Brad Pitt were directed by Guy Ritchie?', n_max_answers=1)
