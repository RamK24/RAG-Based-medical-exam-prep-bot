from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()

class Docs(db.Model):
    doc_id = db.Column(db.Integer, primary_key=True)
    textbook = db.Column(db.String(100), nullable=False)
    content = db.Column(db.String(10000), nullable=False)

    def __repr__(self):
        return '<id %r>' %self.doc_id