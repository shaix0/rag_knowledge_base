"""
The flask application package.
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config
# app = Flask(__name__)

# import rag_knowledge_base.views

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)

    # from .config.quiz_config import MODE_CONFIG
    # app.config['MODE_CONFIG'] = MODE_CONFIG

    from .blueprints.main_bp import main_bp
    from .blueprints.questions_bp import questions_bp
    from .blueprints.quiz_bp import quiz_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(questions_bp)
    app.register_blueprint(quiz_bp)

    return app
