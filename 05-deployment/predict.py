import pickle
from flask import Flask, request, jsonify

with open('dv.bin', 'rb') as dv:
    dv = pickle.load(dv)
with open('model1.bin', 'rb') as model:
    model = pickle.load(model)


app = Flask('subscription')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    subscription = y_pred >= 0.5

    result = {
        'subscription_probability': float(y_pred),
        'subscription': bool(subscription)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)