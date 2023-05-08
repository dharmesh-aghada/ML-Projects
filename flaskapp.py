# Importing Libraries
from flask import Flask, request, render_template
import numpy as np
import pickle

# Creating Flask App
flaskapp = Flask(__name__)

# Loading Pickle file of Model
model = pickle.load(open('irismodel.pkl', 'rb'))


@flaskapp.route('/')
def homepage():
    return render_template('index.html')


@flaskapp.route('/predict', methods=['POST'])
def predict():
    features_float = [float(x) for x in request.form.values()]
    features = [np.array(features_float)]
    prediction = model.predict(features)
    return render_template('index.html', prediction_text='The flower species is {}'.format(prediction))


# Running Flask App
if __name__ == '__main__':
    flaskapp.run(debug=True)