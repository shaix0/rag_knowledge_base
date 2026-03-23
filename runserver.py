"""
This script runs the rag_knowledge_base application using a development server.
"""

from os import environ
from rag_knowledge_base import create_app

app = create_app()

if __name__ == '__main__':
    # app = create_app()
    debug = True
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
