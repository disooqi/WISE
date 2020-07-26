from flask import Flask
from .proxy import ReverseProxied

from .wise import *
from . import nlp as _nlp, sparqls as _sparqls
__all__ = ['Wise']

app = Flask(__name__)
app.wsgi_app = ReverseProxied(app.wsgi_app)
app.config['SECRET_KEY'] = 'thesecretkey'

from . import routes
