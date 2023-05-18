import os
import json
from flask import request, jsonify
import requests
from app import app
import pandas as pd
from time import time, ctime
from algo import infer
from config import DATA_PATH, MODEL_PATH, MODEL_FILE_NAME
from config import logging


@app.route("/")
def welcome():
    return "Welcome"


@app.route('/api/v1/predict_conf', methods=["GET", "POST"])
def predict_conf():
    start_time = ctime(time())
    
    try:
        sepal_length = float(request.args.get("sepal_length"))
        if sepal_length == "":
            return jsonify({"request_timestamp": start_time, "status_code": 200, "message":"`sepal_length` can't be empty!"})
    except Exception:
        pass

    try:
        sepal_width = float(request.args.get("sepal_width"))
        if sepal_width == "":
            return jsonify({"request_timestamp": start_time, "status_code": 200, "message":"`sepal_width` can't be empty!"})
    except Exception:
        pass

    try:
        petal_length = float(request.args.get("petal_length"))
        if petal_length == "":
            return jsonify({"request_timestamp": start_time, "status_code": 200, "message":"`petal_length` can't be empty!"})
    except Exception:
        pass

    try:
        petal_width = float(request.args.get("petal_width"))
        if petal_width == "":
            return jsonify({"request_timestamp": start_time, "status_code": 200, "message":"`petal_width` can't be empty!"})
    except Exception:
        pass

    logging.info("all parameters captured from user")

    sample_features = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred_proba = infer(model_path=MODEL_PATH, model_fname=MODEL_FILE_NAME, sample=sample_features)
    
    logging.info("generating output")
    output = {"request_timestamp":start_time, "status_code":0}
    for idx, item in enumerate(pred_proba):
        output[f"sample input {idx+1}"] = {'conf. setosa':item[0], 'conf. versicolor':item[1], 'conf. virginica':item[2]}

    with open(os.path.join(DATA_PATH, 'output_history.json'), 'a+') as output_file:
        json.dump(output, output_file)

    return jsonify(output)


@app.route('/api/v1/predict_conf_file', methods=["POST"])
def predict_conf_file():
    start_time = ctime(time())
    df = pd.read_csv(requests.files.get('sample_file'))
    logging.info("all parameters received")


    pred_proba_all = infer(model_path=MODEL_PATH, model_fname=MODEL_FILE_NAME, sample=df)
    
    logging.info("generating output")
    output = {"request_timestamp":start_time, "status_code":0}
    for idx, item in enumerate(pred_proba_all):
        output[f"sample input {idx+1}"] = {'conf. setosa':item[0], 'conf. versicolor':item[1], 'conf. virginica':item[2]}
    
    with open(os.path.join(DATA_PATH, 'output_history.json'), 'a+') as output_file:
        json.dump(output, output_file)

    return jsonify(output)