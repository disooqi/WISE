import json
import networkx as nx


class Question:
    question_types = ('person', 'price', 'count', 'date')
    answer_types = ('number', 'date', 'string', 'boolean', 'resource', 'list')

    def __init__(self, question_text, question_id=None, answer_type=None):
        self._id = question_id
        self._question_text = question_text
        self._query_graph = nx.DiGraph()
        self._question_type = None
        self._answer_type = answer_type
        self._parse_components = None
        self._possible_answers = list()

    def add_possible_answer(self, **kwargs):
        self._possible_answers.append(Answer(**kwargs))

    @property
    def possible_answers(self):
        return self._possible_answers

    def add_relation(self):
        pass

    def add_entity(self):
        pass

    def add_entity_property(self):
        pass

    def add_relation_property(self):
        pass

    @property
    def question_type(self):
        return self._question_type

    @question_type.setter
    def question_type(self, value):
        if value not in Question.question_types:
            raise ValueError(f"Question should has one of the following types {Question.question_types}")
        self._question_type = value

    @property
    def answer_type(self):
        return self._answer_type

    @answer_type.setter
    def answer_type(self, value):
        if value not in Question.answer_types:
            raise ValueError(f"Question should has one of the following types {Question.answer_types}")
        self._answer_type = value

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


class Answer:
    def __init__(self, **kwargs):
        # self._sparql = kwargs.get('sparql', None)
        # self._question = kwargs.get('question', None)
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

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self._answer[key] = value

    def json(self):
        return self._answer



    @property
    def sparql(self):
        return self._answer['sparql']

    @sparql.setter
    def sparql(self, value):
        self._answer['sparql'] = value


if __name__ == '__main__':
    pass
