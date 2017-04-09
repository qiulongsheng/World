# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:19:42 2017

@author: Atlantis
"""

from flask import Flask
app = Flask(__name__)

#@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True)