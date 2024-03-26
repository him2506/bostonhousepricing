from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickled model
with open('regmodel.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract input values from JSON data
    input_values = [float(data[f'input_{i+1}']) for i in range(13)]

    # Make prediction using the model
    prediction = model.predict(np.array(input_values).reshape(1, -1))

    # Prepare JSON response
    response = {'prediction': prediction[0]}

    # Return JSON response
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
