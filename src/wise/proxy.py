#!./venv python
# -*- coding: utf-8 -*-
"""
WISE: Natural Language Platform to Query Knowledge bases
"""
__author__ = "Mohamed Eldesouki"
__copyright__ = "Copyright 2020-29, GINA CODY SCHOOL OF ENGINEERING AND COMPUTER SCIENCE, CONCORDIA UNIVERSITY"
__credits__ = ["Mohamed Eldesouki"]
__maintainer__ = "CODS Lab"
__email__ = "wise@cods.encs.concordia.ca"
__created__ = "2020-05-30"


class ReverseProxied(object):
    """Wrap the application in this middleware and configure the
    front-end server to add these headers, to let you quietly bind
    this to a URL other than / and to an HTTP scheme that is
    different than what is used locally.

    In nginx:
    location /WISE {
        proxy_pass http://unix:/work/wise-website/wise.sock;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Scheme $scheme;
        proxy_set_header X-Script-Name /WISE;
        }

    :param app: the WSGI application
    """
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        script_name = environ.get('HTTP_X_SCRIPT_NAME', '')
        if script_name:
            environ['SCRIPT_NAME'] = script_name
            path_info = environ['PATH_INFO']
            if path_info.startswith(script_name):
                environ['PATH_INFO'] = path_info[len(script_name):]

        scheme = environ.get('HTTP_X_SCHEME', '')
        if scheme:
            environ['wsgi.url_scheme'] = scheme
        return self.app(environ, start_response)

'''
The references of this solution were here:
https://web.archive.org/web/20190128010140/http://flask.pocoo.org/snippets/35/
https://stackoverflow.com/questions/36258506/gunicorn-with-gaiohttp-worker-always-returning-404-with-flask-app
https://stackoverflow.com/questions/18967441/add-a-prefix-to-all-flask-routes
https://medium.com/@varunchitale/a-simple-python-api-using-flask-with-nginx-setup-on-aws-ec2-4a380ceaf006
'''