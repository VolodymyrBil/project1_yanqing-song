import pickle

from flask import Flask
from flask import request
from flask import jsonify


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('death')


@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    X = dv.transform([patient])

    y_pred = model.predict_proba(X)[0, 1]
    death = y_pred >= 0.5

    result = {'death_event_probability': float(y_pred), 'death_event': bool(death)}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
