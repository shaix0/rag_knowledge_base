"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__)

import rag_knowledge_base.views
