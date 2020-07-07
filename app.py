from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy  # add
from datetime import datetime  # add

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///train.db'  # add
db = SQLAlchemy(app)  # add


# add
class train_data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ACTION = db.Column(db.String(80), nullable=False)
    RESOURCE = db.Column(db.String(80), nullable=False)
    MGR_ID = db.Column(db.String(80), nullable=False)
    ROLE_ROLLUP_1 = db.Column(db.String(80), nullable=False)
    ROLE_ROLLUP_2 = db.Column(db.String(80), nullable=False)
    ROLE_DEPTNAME = db.Column(db.String(80), nullable=False)
    ROLE_TITLE = db.Column(db.String(80), nullable=False)
    ROLE_FAMILY_DESC = db.Column(db.String(80), nullable=False)
    ROLE_CODE = db.Column(db.String(80), nullable=False)

@app.route('/')
def index():
    # return "Hello World!"
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)




