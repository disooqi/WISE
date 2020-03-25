import json
import networkx as nx
import bisect


class Question:
    types = ('person', 'price', 'count', 'date')  # it should be populated by the types of ontology
    datatypes = ('number', 'date', 'string', 'boolean', 'resource', 'list')

    def __init__(self, question_text, question_id=None, answer_datatype=None):
        self._id = question_id
        self._question_text = question_text
        self.graph = nx.DiGraph()
        self._answer_type = list()
        self._answer_datatype = answer_datatype
        self._parse_components = None
        self._possible_answers = list()

    def add_possible_answer(self, **kwargs):
        # bisect.insort(self._possible_answers, Answer(**kwargs))  # it is not going to work because some answers are
        # inserted without score at first
        self._possible_answers.append(Answer(**kwargs))

    @property
    def possible_answers(self):
        return self._possible_answers

    def add_entity(self, named_entity, **kwargs):
        self.graph.add_node(named_entity, **kwargs)

    def add_entity_properties(self, named_entity, **kwargs):
        for key, value in kwargs.items():
            self.graph.nodes[named_entity][key] = value

    def add_relation(self, source, destination, **kwargs):
        self.graph.add_edge(source, destination, **kwargs)

    def add_relation_properties(self, source, destination, **kwargs):
        self.graph.add_edge(source, destination, **kwargs)

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

    @property
    def parse_components(self):
        return self._parse_components

    @parse_components.setter
    def parse_components(self, value):
        self._parse_components = value

    @property
    def entities(self):
        return list(self.graph.nodes)

    @property
    def relations(self):
        return list(self.graph.edges)


class Answer:
    def __init__(self, **kwargs):
        self._answer = dict({
            "question": None,
            # "question_id": kwargs['question_id'],  # question_id
            "results": None,  # here are the bindings returned from the triple store
            "status": None,  # same as the http request status, and actually it does not make sense and I might remove
            "vars": None,
            "sparql": None
        })
        for key, value in kwargs.items():
            self._answer[key] = value

    def __lt__(self, other):
        return self.score < other.score

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self._answer[key] = value

    def json(self):
        return self._answer

    @property
    def sparql(self):
        return self._answer['sparql']

    @property
    def score(self):
        return self._answer['score']

    @sparql.setter
    def sparql(self, value):
        self._answer['sparql'] = value


if __name__ == '__main__':
    pass
