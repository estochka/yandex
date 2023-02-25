import pickle

import pandas as pd
from catboost import CatBoostClassifier
from flask import Flask, request, render_template

app = Flask(__name__)
with open('prep.pkl', "rb") as f:
    prep = pickle.load(f)

cat_model = CatBoostClassifier()
cat_model.load_model('cat_model.bin')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    data = []
    columns = ['internet_service', 'online_security', 'online_backup', 'device_protection', 'tech_support',
               'streaming_tv', 'streaming_movies', 'multiple_lines', 'gender', 'senior_citizen', 'partner',
               'dependents', 'begin_date', 'end_date', 'type', 'paperless_billing', 'payment_method',
               'monthly_charges', 'total_charges']

    for i in range(1, 20):
        data.append(request.form[str(i)])
    try:
        data = pd.DataFrame([data], columns=columns)
        data[['monthly_charges', 'total_charges']] = data[['monthly_charges', 'total_charges']].astype('float')
        data = prep.transform(data)

        result = round(cat_model.predict_proba(data)[0, 1], 3)
    except:
        result = 'неверный ввод даты'
    return render_template('predict.html', result=result)




