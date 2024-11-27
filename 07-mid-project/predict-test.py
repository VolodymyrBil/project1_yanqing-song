import requests


url = 'http://localhost:9696/predict'
patient = {
    'age': 65,
    'anaemia': 0,
    'creatinine_phosphokinase': 1688,
    'diabetes': 0,
    'ejection_fraction': 38,
    'high_blood_pressure': 0,
    'platelets': 263360,
    'serum_creatinine': 1.1,
    'serum_sodium': 138,
    'sex': 1,
    'smoking': 1,
    'time': 250,
}
response = requests.post(url, json=patient).json()
print(response)


if response['death_event']:
    print('The patient may not survive')
else:
    print('The patient may survive')
