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

import json
import logging
import copy
from functools import reduce
from collections import defaultdict
import networkx as nx
from .nlp.relation import RelationLabeling
from transitions.core import MachineError
from .nlp.utils import traverse_tree, table, punctuation
from .nlp.models import ner, parser

logger = logging.getLogger(__name__)
if not logger.handlers:
    file_handler = logging.FileHandler('wise.log')
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


class Question:
    types = ('person', 'price', 'count', 'date')  # it should be populated by the types of ontology
    datatypes = ('number', 'date', 'string', 'boolean', 'resource', 'list')

    def __init__(self, question_text: str, question_id: int = None, answer_datatype: str = None):
        self.tokens = list()
        self._id = question_id
        self._question_text = question_text
        self.query_graph = nx.MultiGraph()
        self._answer_type = list()
        self._answer_datatype = answer_datatype
        self._parse_components = None
        self._possible_answers = list()

        self.__process()

    def add_possible_answer(self, **kwargs):
        # bisect.insort(self._possible_answers, Answer(**kwargs))  # it is not going to work because some answers are
        # inserted without score at first

        q = {"language": "en", "string": self._question_text, "keywords": ""}
        self._possible_answers.append(Answer(_id=self._id, question=q, answertype=self._answer_datatype, **kwargs))

    def format_answers(self, merge_answers):
        # TODO: you should remove duplicates when you merge
        answers = list(filter(lambda a: a['answers'], map(lambda a: a.as_dict, self._possible_answers)))
        if not merge_answers or not answers:
            return answers
        answer_template = copy.deepcopy(answers[0])
        answer_template['answers'][0]['results']['bindings'].clear()
        for answer in answers:
            answer_template['answers'][0]['results']['bindings'].extend(answer['answers'][0]['results']['bindings'])

        else:
            return answer_template

    @property
    def possible_answers(self):
        return self._possible_answers

    def add_possible_answer_type(self, ontology_type: str):
        self._answer_type.append(ontology_type)

    @property
    def answer_type(self):
        return self._answer_type

    @answer_type.setter
    def answer_type(self, value):
        self._answer_type.append(value)

    @property
    def answer_datatype(self):
        return self._answer_datatype

    @answer_datatype.setter
    def answer_datatype(self, value):
        if value not in Question.datatypes:
            raise ValueError(f"Question should has one of the following types {Question.datatypes}")
        self._answer_datatype = value

    @property
    def id(self):
        return self._id

    @property
    def text(self):
        return self._question_text

    def __process(self):
        self.__parse_sentence()
        self.__regroup_named_entities()
        self.__find_possible_entities_and_relations()

    def __parse_sentence(self):
        allannlp_ner_output = ner.predict(sentence=self._question_text)
        allannlp_dep_output = parser.predict(sentence=self._question_text)

        words = allannlp_ner_output['words']
        ner_tags = allannlp_ner_output['tags']
        pos_tags = allannlp_dep_output['pos']
        dependencies = allannlp_dep_output['predicted_dependencies']
        heads = allannlp_dep_output['predicted_heads']
        # d = reformat_allennlp_ner_output(ner_tags, words)

        positions = traverse_tree(allannlp_dep_output['hierplane_tree']['root'])
        positions.sort()
        words_info = list(zip(range(1, len(words) + 1), words, heads, dependencies, positions, pos_tags, ner_tags))

        for i, w, h, d, p, pos, t in words_info:
            self.tokens.append({'index': i, 'token': w, 'head': h, 'dependency': d, 'position': p,
                                'pos-tag': pos, 'ne-tag': t})

    def __regroup_named_entities(self):
        l2 = list()
        entity = list()
        tag = ''

        head = None
        h_d = list()
        dep = None
        poss = list()
        position = None
        for token in self.tokens:
            if token['ne-tag'].startswith('B-'):
                tag = token['ne-tag'][2:]
                position = token['position']
                if 'obj' in token['dependency'] or 'subj' in token['dependency']:
                    head, dep = token['head'], token['dependency']
                h_d.append((token['index'], token['head'], token['dependency']))
                poss.append(token['pos-tag'])
                entity.append(token['token'])
            elif token['ne-tag'].startswith('I-'):
                if 'obj' in token['dependency'] or 'subj' in token['dependency']:
                    head, dep = token['head'], token['dependency']
                h_d.append((token['index'], token['head'], token['dependency']))
                poss.append(token['pos-tag'])
                entity.append(token['token'])
            elif token['ne-tag'].startswith('L-'):
                if 'obj' in token['dependency'] or 'subj' in token['dependency']:
                    head, dep = token['head'], token['dependency']
                h_d.append((token['index'], token['head'], token['dependency']))
                entity_idxs = list(zip(*h_d))[0]
                if not head and not dep:
                    for _, _h, _d in h_d:
                        if token['head'] not in entity_idxs:
                            head, dep = _h, _d
                            break
                    else:
                        head, dep = token['head'], token['dependency']
                poss.append(token['pos-tag'])
                entity.append(token['token'])
                l2.append((token['index'], ' '.join(entity), head, dep, position, ' '.join(poss), tag))
                entity.clear()
            elif token['ne-tag'].startswith('U-'):
                l2.append((token['index'], token['token'], token['head'], token['dependency'], token['position'],
                           token['pos-tag'], token['ne-tag'][2:]))
            else:
                l2.append((token['index'], token['token'], token['head'], token['dependency'], token['position'],
                           token['pos-tag'], token['ne-tag']))
        else:
            self.tokens.clear()
            for i, w, h, d, p, pos, t in l2:
                self.tokens.append({'index': i, 'token': w, 'head': h, 'dependency': d, 'position': p, 'pos-tag': pos,
                                    'ne-tag': t})
            else:
                logger.info(f"[NAMED-ENTITIES:] {self.tokens}")

    def __find_possible_entities_and_relations(self):
        s, pred, o = list(), list(), list()
        relations_ignored = ['has', 'have', 'had', 'be', 'is', 'are', 'was', 'were', 'do', 'did', 'does',
                             'much', 'many', 'give', 'show', '']
        relation_labeling = RelationLabeling()
        # positions = [token['position'] for token in self.question.tokens]
        #  i = word index, w = word_text, h = Dep_head, d
        for token in self.tokens:
            if token['token'].lower() in ['how', 'who', 'when', 'what', 'which', 'where']:
                continue
            if token['token'] in punctuation:
                # TODO: "they" has an indication that the answer is list of people
                continue
            token['token'] = token['token'].translate(table)

            try:
                pos = token["pos-tag"] if token['ne-tag'] == 'O' else 'NE'
                tok = token['token']
                eval(f'relation_labeling.{pos.replace("$", "_")}("{tok}", "{token["pos-tag"]}")')
            except AttributeError as ae:
                relation_labeling.flush_relation()
            except MachineError as me:
                print(f"MachineError: {me}")
                relation_labeling.flush_relation()
            else:
                pass
            finally:
                pass

            # if token['token'].lower() in STOPWORDS:
            #     # TODO: "they" has an indication that the answer is list of people
            #     continue

            if token['ne-tag'] != "O":
                s.append((token['index'], token['token'], token['head'], token['dependency'], token['position'],
                          token['pos-tag'], token['ne-tag']))
            elif 'subj' in token['dependency'] or 'obj' in token['dependency']:
                self.add_possible_answer_type(token['token'])
        else:
            relation_labeling.flush_relation()
            relations = list(filter(lambda x: x.lower() not in relations_ignored, relation_labeling.relations))

        for i, entity, h, d, p, pos, t in s + o:
            # TODO: This for-loop does not consider relation between two named entities
            if entity.startswith('the '):
                entity = entity[4:]
            self.query_graph.add_node(entity, pos=pos, entity_type=t, uris=[])
            for relation in relations:
                relation_key = self.query_graph.add_edge(entity, 'var', relation=relation, uris=[])

        logger.info(f"[NODES:] {s + o}")
        logger.info(f"[RELATIONS:] {relations}")


class Answer:
    __slots__ = ('question', '__answer')

    def __init__(self, _id: int = 0, answertype: str = None, aggregation: bool = False, onlydbo: bool = False,
                 hybrid: bool = False, question: dict = None, sparql: str = None, score: float = 0):
        # keys = ["id", "answertype", "aggregation", "onlydbo", "hybrid", "question", "query", "answers"]
        self.question = [question]

        self.__answer = {
            "id": _id,
            "answertype": answertype,
            "aggregation": aggregation,
            "onlydbo": onlydbo,
            "hybrid": hybrid,
            "question": self.question,
            "score": score,
            "query": {"sparql": sparql},
            "answers": []
        }
        # print(self.__answer)

    def __lt__(self, other):
        return self.__answer['score'] < other.__answer['score']

    def update_answers_element(self, triple_store_response):

        # TODO case 1: what if response got no answer
        result = json.loads(triple_store_response)
        answer = defaultdict(dict)
        try:
            answer['head']['vars'] = result['head']['vars']
            if not result['results']['bindings']:
                return
            answer['results']['bindings'] = result['results']['bindings']
        except KeyError:
            return

        self.__answer['answers'].append(answer)

    @property
    def sparql(self):
        return self.__answer['query']['sparql']

    @property
    def as_dict(self):
        return self.__answer

    def __del__(self):
        # The destructor
        # This magic method is not called when "del" used, it is called when the refcount reaches 0
        pass


if __name__ == '__main__':
    pass
