# -*- coding: utf-8 -*-
"""
WISE: Natural Language Platform to Query Knowledge bases
"""
__author__ = "Mohamed Eldesouki"
__copyright__ = "Copyright 2020-29, GINA CODY SCHOOL OF ENGINEERING AND COMPUTER SCIENCE, CONCORDIA UNIVERSITY"
__credits__ = ["Mohamed Eldesouki"]
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "mohamed@eldesouki.ca"
__created__ = "2020-07-10"

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import DataRequired, Length, NumberRange


class SearchForm(FlaskForm):
    question = StringField('Ask A Question?', validators=[DataRequired(), Length(min=2, max=300)])
    n_answers = IntegerField('Max', validators=[DataRequired(), NumberRange(min=1, max=100)], default=1)
    search = SubmitField('Look for Answer')
