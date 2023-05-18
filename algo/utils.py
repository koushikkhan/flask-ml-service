import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from config import logging
import xml.etree.ElementTree as ET


# utility functions
def load_data(data_path, data_fname):
    dat = pd.read_csv(os.path.join(data_path, data_fname))
    return dat

def label_to_id(label):
    label_to_id_map = {"setosa":0, "versicolor":1, "virginica":2}
    return label_to_id_map[label]

def id_to_label(id):
    id_to_label_map = {0:"setosa", 1:"versicolor", 2:"virginica"}
    return id_to_label_map[id]

def train_model(data_path, data_fname, model_path, model_fname):
    data = load_data(data_path=data_path, data_fname=data_fname)
    data["species"] = data["species"].apply(lambda x: label_to_id(x))
    features = data.iloc[:,0:4].to_numpy()
    target = data.iloc[:, -1].to_numpy()

    print(f"feature dimension: {np.shape(features)} target dimension: {np.shape(target)}")
    
    # instantiating estimator. Choosing K=1
    model = RandomForestClassifier()

    # build model
    model.fit(X=features, y=target)

    # save model
    with open(os.path.join(model_path, model_fname), 'wb') as model_out:
        pickle.dump(model, model_out)
    print("model file has been saved!")

    return

def infer(model_path, model_fname, sample):
    # load model
    with open(os.path.join(model_path, model_fname), 'rb') as model_in:
        model = pickle.load(model_in)
    logging.info(f"{model_fname} loaded for inference")
    
    pred_proba = model.predict_proba(sample)
    logging.info(f"prediction complete")
    return pred_proba