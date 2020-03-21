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
        self._answer_sparql = None
        self._parse_components = None

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


if __name__ == '__main__':
    pass
