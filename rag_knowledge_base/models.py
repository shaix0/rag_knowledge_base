from . import db
from sqlalchemy.dialects.postgresql import JSONB

class Question(db.Model):
    __tablename__ = 'questions'

    id = db.Column(db.Integer, primary_key=True)
    question_text = db.Column(db.Text, nullable=False)
    options = db.Column(JSONB, nullable=False)
    answers = db.Column(JSONB, nullable=False)
    source = db.Column(db.String)
    page_number = db.Column(db.Integer)
    error_count = db.Column(db.Integer, default=0)

    def to_dict(self):
        return {
            "id": self.id,
            "question_text": self.question_text,
            "options": self.options,
            "answers": self.answers,
            "source": self.source,
            "page_number": self.page_number,
            "error_count": self.error_count
        }

    def __repr__(self):
        return f"<Question {self.id}: {self.question_text[:30]}...>"