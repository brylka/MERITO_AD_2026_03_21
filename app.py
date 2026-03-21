import joblib
from flask import Flask, render_template

model = joblib.load('model.joblib')
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)