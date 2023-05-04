import numpy as np
import pickle
from flask import Flask, request, render_template
from waitress import serve

model = pickle.load(open('../models/model_rf', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    print(array_features)
    # Predict features
    prediction = model.predict(array_features)
    pred = "{:.2f}".format(*prediction)
    # pred = str(prediction)
    print(pred)

    return render_template('index.html', result=pred)

if __name__ == '__main__':
    serve(app, host="127.0.0.1", port=8080)