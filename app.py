# -*- coding: utf-8 -*-
"""
WISE: Natural Language Platform to Query Knowledge bases
"""
__author__ = "Mohamed Eldesouki"
__copyright__ = "Copyright 2020-29, GINA CODY SCHOOL OF ENGINEERING AND COMPUTER SCIENCE, CONCORDIA UNIVERSITY"
__credits__ = ["Mohamed Eldesouki"]
__maintainer__ = "CODS Lab"
__email__ = "wise@eldesouki.ca"
__created__ = "2020-05-30"

from src.wise import app


if __name__ == '__main__':
    # app.run(port=5000, host='0.0.0.0', debug=True, static_files = {'/static': '/path/to/static'})
    app.run(port=5000, host='0.0.0.0', debug=True)
