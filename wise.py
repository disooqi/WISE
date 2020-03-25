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
import logging
from string import punctuation
from collections import defaultdict
from itertools import count, product, chain, starmap, zip_longest
from statistics import mean
from urllib.parse import urlparse
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from sparqls import (make_keyword_unordered_search_query_with_type, make_top_predicates_sbj_query,
                     make_top_predicates_obj_query, evaluate_SPARQL_query, construct_answers_query,
                     construct_yesno_answers_query, construct_yesno_answers_query2,
                     sparql_query_to_get_predicates_when_subj_and_obj_are_known)
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



# ner_predictor2 = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
# oie_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")

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
    ner1 = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")
    ner2 = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    parser = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    def __init__(self, semantic_afinity_server=None, n_max_answers: int = 100):
        self._ss_server = semantic_afinity_server
        self._n_max_answers = n_max_answers  # this should affect the number of star queries to be executed against TS
        self._current_question = None
        self.n_max_Vs = 2
        self.n_max_Es = 3
        self.v_uri_scores = defaultdict(float)

    @property
    def question(self):
        return self._current_question

    @question.setter
    def question(self, value: str):
        self._current_question = Question(question_text=value)

    def ask(self, question_text: str, question_id: int = 0, answer_type: str = None, n_max_answers: int =None):
        """WISE pipeline

        Usage::

            >>> from wise import Wise
            >>> my_wise = Wise()
            >>> my_wise.ask("What is the longest river?")


        """
        self.question = question_text
        # self.question.id = question_id

        if answer_type:
            self.question.answer_datatype = answer_type
        logger.info(f'\n{"<>" * 200}\n[Question:] {self.question.text},\n')
        self._n_max_answers = n_max_answers if n_max_answers else self._n_max_answers
        self.detect_question_and_answer_type()
        self.rephrase_question()
        self.process_question()
        self.find_possible_noun_phrases_and_relations()
        # if no named entity you should return here
        if not self.question.graph.nodes:
            return []
        self.extract_possible_V_and_E()
        self.generate_star_queries()
        self.evaluate_star_queries()

        answers = [answer.json() for answer in self.question.possible_answers[:n_max_answers]]

        logger.info(f"[FINAL ANSWERS:] {self.question.sparqls}")

        return answers

    def detect_question_and_answer_type(self):
        # question_text = question_text.lower()
        # if question_text.startswith('Who'):
        #     question_text = re.sub('Who', 'Mohamed', question_text)

        properties = ['name', 'capital', 'country']
        # what is the name
        # what country

        if self.question.text.lower().startswith('who was'):
            self.question.answer_type = 'person'
            self.question.answer_datatype = 'resource'
            self.question.add_entity('var', question_type=self.question.answer_type)
        elif self.question.text.lower().startswith('who is '):
            self.question.answer_type = 'person'
            self.question.answer_datatype = 'resource'
        elif self.question.text.lower().startswith('who are '):
            self.question.answer_type = 'person'
            self.question.answer_datatype = 'list'
        elif self.question.text.lower().startswith('who '):  # Who [V]
            self.question.answer_type = 'person'
            self.question.answer_datatype = 'resource'  # of list
        elif self.question.text.lower().startswith('how many '):
            self.question.answer_type = 'count'
            self.question.answer_datatype = 'number'
        elif self.question.text.lower().startswith('how much '):
            self.question.answer_type = 'price'
            self.question.answer_datatype = 'number'
        elif self.question.text.lower().startswith('when did '):
            self.question.answer_type = 'date'
            self.question.answer_datatype = 'date'
        elif self.question.text.lower().startswith('in which '):  # In which [NNS], In which city
            self.question.answer_type = 'person'
            self.question.answer_datatype = 'resource'  # of list
        elif self.question.text.lower().startswith('which '):  # which [NNS], which actors
            self.question.answer_type = 'person'
            self.question.answer_datatype = 'list'  # of list
        elif self.question.text.lower().startswith('where '):  # where do
            self.question.answer_type = 'person'
            self.question.answer_datatype = 'resource'  # of list
        elif self.question.text.lower().startswith('show '):  # Show ... all
            self.question.answer_type = 'person'
            self.question.answer_datatype = 'list'  # of list
        else:
            pass  # 11,13,75

        logger.info(f'[QUESTION TYPE:] {self.question.answer_type}, [ANSWER TYPE:] {self.question.answer_datatype},\n')

    def rephrase_question(self):
        if self.question.text.lower().startswith('who was'):
            pass
        logger.info(f'[Question Reformulation (Not Impl yet):] {self.question.text},\n')

    def process_question(self):
        parse_components = self.__class__._parse_sentence(self.question.text)
        self.question.parse_components = self.__class__._regroup_named_entities(parse_components)
        ne_extra = self.__class__.ner2.predict(sentence=self.question.text)
        self.question.ne_extra = self._get_named_entities(ne_extra['words'], ne_extra['tags'])

    def find_possible_noun_phrases_and_relations(self):
        s, pred, o = list(), list(), list()
        # TODO look at what other tags from dep parser are considered to identify subject and objects
        positions = list(zip(*self.question.parse_components))[4]
        #  i = word index, w = word_text, h = Dep_head, d
        for i, w, h, d, p, pos, t in self.question.parse_components:
            if w.lower() in ['how', 'who', 'when', 'what', 'which', 'where']:
                continue
            if w in punctuation or w.lower() in STOPWORDS:
                # TODO: "they" has an indication that the answer is list of people
                continue
            w = w.translate(table)

            if t != "O":
                s.append((i, w, h, d, p, pos, t))
            elif 'subj' in d or 'obj' in d:
                self.question.add_possible_answer_type(w)
            else:
                pred.append((i, w, h, d, p, pos, t))

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

        # TODO: This is a hack you need find a better way
        # if not relations:
        #     relations.append(self.question.answer_type[0])

        # # TODO: This is a hack you need find a better way
        # if not s and not o:
        #     s.extend(self.question.ne_extra)
        #     for rel in relations:
        #         if not s:  # in case no relation
        #             return
        #         for e in s:
        #             if rel in e:
        #                 continue
        #         else:
        #             self.question.add_relation(e, 'var', relation=rel)
        #     else:
        #         return

        for i, entity, h, d, p, pos, t in s + o:
            self.question.add_entity(entity, pos=pos, entity_type=t)
            for relation in relations:
                self.question.add_relation(entity, 'var', relation=relation, uris=[])

        logger2.debug(f"SUBJs: {self.question.graph.nodes}")
        logger2.debug(f"RELATIONS: {list(self.question.graph.edges.data('relation'))}")
        logger.info(f'[GRAPH:] {self.question.entities} <===>  {self.question.relations}\n')

    def extract_possible_V_and_E(self):
        for entity in self.question.entities:
            if entity == 'var':
                self.question.add_entity_properties(entity, uris=[])
                continue
            entity_query = make_keyword_unordered_search_query_with_type(entity, limit=100)

            try:
                entity_result = json.loads(evaluate_SPARQL_query(entity_query))
            except:
                logger.error(f"Error at 'extract_possible_V_and_E' method with v_query value of {entity_query} ")
                continue

            uris, names = self.__class__.extract_resource_name(entity_result['results']['bindings'])
            scores = self.__compute_semantic_similarity_between_single_word_and_word_list(entity, names)

            URIs_with_scores = list(zip(uris, scores))
            URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
            self.v_uri_scores.update(URIs_with_scores)
            URIs_sorted = list(zip(*URIs_with_scores))[0]
            URIs_chosen = remove_duplicates(URIs_sorted)[:self.n_max_Vs]

            self.question.add_entity_properties(entity, uris=URIs_chosen)

            logger.info(f"[URIs for Entity '{entity}':] {URIs_chosen}")

        # Find E for all relations
        for (source, destination, relation) in self.question.graph.edges.data('relation'):
            source_URIs = self.question.graph.nodes[source]['uris']
            destination_URIs = self.question.graph.nodes[destination]['uris']
            combinations = get_combination_of_two_lists(source_URIs, destination_URIs, with_reversed=False)

            uris, names = list(), list()
            if destination == 'var':  # 'var' always comes in the destination part
                for uri in combinations:
                    URIs_false, names_false = self._get_predicates_and_their_names(subj=uri)
                    URIs_true, names_true = self._get_predicates_and_their_names(obj=uri)
                    URIs_false = list(zip_longest(URIs_false, [False], fillvalue=False))
                    URIs_true = list(zip_longest(URIs_true, [True], fillvalue=True))
                    URIs_chosen = self.__get_chosen_URIs(relation, URIs_false+URIs_true, names_false+names_true)
                    self.question.graph[source][destination]['uris'].extend(URIs_chosen)
                    # self.question.add_relation_properties(source, destination, uris=URIs_chosen)
            else:
                for v_uri_1, v_uri_2 in combinations:
                    URIs_false, names_false = self._get_predicates_and_their_names(v_uri_1, v_uri_2)
                    URIs_true, names_true = self._get_predicates_and_their_names(v_uri_2, v_uri_1)
                    URIs_false = list(zip_longest(URIs_false, [False], fillvalue=False))
                    URIs_true = list(zip_longest(URIs_true, [True], fillvalue=True))
                    URIs_chosen = self.__get_chosen_URIs(relation, URIs_false+URIs_true, names_false+names_true)
                    self.question.graph[source][destination]['uris'].extend(URIs_chosen)
                    # self.question.add_relation_properties(source, destination, uris=URIs_chosen)

            # scores = compute_semantic_similarity_between_single_word_and_word_list(relation, names)
            # URIs_with_scores = list(zip(uris, scores))
            # URIs_with_scores.sort(key=operator.itemgetter(1), reverse=True)
            # self.uri_scores.update(URIs_with_scores)
            # URIs_sorted = list(zip(*URIs_with_scores))[0]
            # URIs_chosen = remove_duplicates(URIs_sorted)[:self.n_max_Es]



            logger.info(f"[URIs for RELATION '{relation}':] {URIs_chosen}")

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

    def __get_chosen_URIs(self, relation: str, uris: list, names: list):
        scores = self.__class__.__compute_semantic_similarity_between_single_word_and_word_list(relation, names)
        # (uri, True) ===>  (uri, True, score)
        l1, l2 = list(zip(*uris))
        URIs_with_scores = list(zip(l1, l2, scores))
        URIs_with_scores.sort(key=operator.itemgetter(2), reverse=True)
        # self.uri_scores.update(URIs_with_scores)
        return remove_duplicates(URIs_with_scores)[:self.n_max_Es]

    def generate_star_queries(self):
        possible_triples_for_all_relations = list()
        for source, destination, relation_uris in self.question.graph.edges.data('uris'):
            source_URIs = self.question.graph.nodes[source]['uris']
            destination_URIs = self.question.graph.nodes[destination]['uris']
            node_uris = source_URIs if destination == 'var' else destination_URIs

            possible_triples_for_single_relation = get_combination_of_two_lists(node_uris, relation_uris)
            possible_triples_for_all_relations.append(possible_triples_for_single_relation)
        else:
            for star_query in product(*possible_triples_for_all_relations):
                score = sum([self.v_uri_scores[subj]+predicate[2] for subj, predicate in star_query])

                triple = [f'?var <{predicate[0]}> <{v_uri}>' if predicate[1] else f'<{v_uri}> <{predicate[0]}> ?var'
                          for v_uri, predicate in star_query]

                query = f"SELECT * WHERE {{ {' . '.join(triple)} }}"
                self.question.add_possible_answer(question=self.question.text, sparql=query, score=score)

    def evaluate_star_queries(self):
        self.question.possible_answers.sort(reverse=True)
        qc = count(1)
        sparqls = list()
        for i, possible_answer in enumerate(self.question.possible_answers[:self._n_max_answers]):
            result = evaluate_SPARQL_query(possible_answer.sparql)
            # logger2.debug(f"[RAW RESULT FROM VIRTUOSO:] {result}")
            try:
                v_result = json.loads(result)
                possible_answer.update(results=v_result['results'], vars=v_result['head']['vars'])
                answers = list()
                for binding in v_result['results']['bindings']:
                    answer = self.__class__.extract_resource_name_from_uri(binding['var']['value'])[0]
                    answers.append(answer)

                    # for var, v in binding.items():
                    #     uri, name = self.__class__.extract_resource_name_from_uri(v['value'])
                else:
                    if v_result['results']['bindings']:
                        logger.info(f"[POSSIBLE ANSWER {next(qc)}:] {answers}")
                        sparqls.append(possible_answer.sparql)

            except:
                print(f" >>>>>>>>>>>>>>>>>>>> What the hell [{result}] <<<<<<<<<<<<<<<<<<")
        else:
            self.question.sparqls = sparqls

    @classmethod
    def _parse_sentence(cls, sentence: str):
        allannlp_ner_output = cls.ner1.predict(sentence=sentence)
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
    def _get_named_entities(words, tags):
        named_entities = list()
        entity = list()
        for w, t in zip(words, tags):
            if t.startswith('B-'):
                entity.append(w)
            elif t.startswith('I-'):
                entity.append(w)
            elif t.startswith('L-'):
                entity.append(w)
                named_entities.append(' '.join(entity))
                entity.clear()
            elif t.startswith('U-'):
                named_entities.append(w)

        else:
            logger.info(f'[NAMED ENTITIES FROM EXTRA RECOGNIZER:] {named_entities},\n')
            return named_entities

    @staticmethod
    def _regroup_named_entities(parse_components):
        l2 = list()
        entity = list()
        flag = False
        tag = ''

        head = None
        h_d = list()
        dep = None
        poss = list()
        position = None
        for i, w, h, d, p, pos, t in parse_components:
            if t.startswith('B-'):
                flag = True
                tag = t[2:]
                if 'obj' in d or 'subj' in d:
                    head, dep = h, d
                h_d.append((i, h, d))
                position = p
                poss.append(pos)
                entity.append(w)
            elif flag and t.startswith('I-'):
                if 'obj' in d or 'subj' in d:
                    head, dep = h, d
                h_d.append((i, h, d))
                poss.append(pos)
                entity.append(w)
            elif flag and t.startswith('L-'):
                if 'obj' in d or 'subj' in d:
                    head, dep = h, d
                h_d.append((i, h, d))
                entity_idxs = list(zip(*h_d))[0]
                if not head and not dep:
                    for _, _h, _d in h_d:
                        if h not in entity_idxs:
                            head, dep = _h, _d
                            break
                    else:
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
    def extract_resource_name_from_uri(uri: str):
        resource_URI = uri
        uri_path = urlparse(resource_URI).path
        resource_name = os.path.basename(uri_path)
        resource_name = re.sub(r'(:|_|\(|\))', ' ', resource_name)
        # resource_name = re.sub(r'^Category:', '', resource_name)
        # TODO: check for URI validity
        return resource_URI, resource_name

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

    @staticmethod
    def _get_predicates_and_their_names(subj=None, obj=None):
        if subj and obj:
            q = sparql_query_to_get_predicates_when_subj_and_obj_are_known(subj, obj, limit=100)
            uris, names = Wise.execute_sparql_query_and_get_uri_and_name_lists(q)
        elif subj:
            q = make_top_predicates_sbj_query(subj, limit=100)
            uris, names = Wise.execute_sparql_query_and_get_uri_and_name_lists(q)
        elif obj:
            q = make_top_predicates_obj_query(obj, limit=100)
            uris, names = Wise.execute_sparql_query_and_get_uri_and_name_lists(q)
        else:
            raise Exception

        return uris, names

    @staticmethod
    def execute_sparql_query_and_get_uri_and_name_lists(q):
        result = json.loads(evaluate_SPARQL_query(q))
        return Wise.extract_predicate_names(result['results']['bindings'])


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


def remove_duplicates(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def get_combination_of_two_lists(list1, list2, directed=False, with_reversed=False):
    lists = [l for l in (list1, list2) if l]

    if len(lists) < 2:
        return set(chain(list1, list2))

    combinations = product(*lists, repeat=1)
    combinations_selected = list()
    combinations_memory = list()

    for comb in combinations:
        pair = set(comb)

        if len(lists) == 2 and len(pair) == 1:
            continue

        if not directed and pair in combinations_memory:
            continue
        combinations_memory.append(pair)
        combinations_selected.append(comb)
    else:
        if with_reversed:
            combinations_reversed = [(comb[1], comb[0]) for comb in combinations_selected if len(lists) == 2]
            combinations_selected.extend(combinations_reversed)

        return set(combinations_selected)


if __name__ == '__main__':
    print(get_combination_of_two_lists([50,2,3], [50,3,70], directed=True, with_reversed=True))
