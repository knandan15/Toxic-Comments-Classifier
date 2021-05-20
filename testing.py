#from flask import Flask, render_template, url_for, request, jsonify      
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#for BERT
import pandas as pd
import argparse
import os
from detoxify import Detoxify




def run(model_name):
    """Loads model from checkpoint or from model name and runs inference on the input_obj.
    Displays results as a pandas DataFrame object.
    If a dest_file is given, it saves the results to a txt file.
    """
    text = ['stai zitto, tu sei un bugiardo']
    if model_name is not None:
        res = Detoxify(model_name).predict(text)
    # else:
    #     res = Detoxify(checkpoint=from_ckpt).predict(text)

    res_df = pd.DataFrame(res, index=[text] if isinstance(text, str) else text).round(5)
    print(res_df)
    

    return res


# Load the TF-IDF vocabulary specific to the category
with open(r"toxic_vect.pkl", "rb") as f:
    tox = pickle.load(f)

with open(r"severe_toxic_vect.pkl", "rb") as f:
    sev = pickle.load(f)

with open(r"obscene_vect.pkl", "rb") as f:
    obs = pickle.load(f)

with open(r"insult_vect.pkl", "rb") as f:
    ins = pickle.load(f)

with open(r"threat_vect.pkl", "rb") as f:
    thr = pickle.load(f)

with open(r"identity_hate_vect.pkl", "rb") as f:
    ide = pickle.load(f)

# Load the pickled RDF models
with open(r"toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open(r"severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open(r"obscene_model.pkl", "rb") as f:
    obs_model  = pickle.load(f)

with open(r"insult_model.pkl", "rb") as f:
    ins_model  = pickle.load(f)

with open(r"threat_model.pkl", "rb") as f:
    thr_model  = pickle.load(f)

with open(r"identity_hate_model.pkl", "rb") as f:
    ide_model  = pickle.load(f)

data=['fuck you']
vect = tox.transform(data)
pred_tox = tox_model.predict_proba(vect)[:,1]

vect = sev.transform(data)
pred_sev = sev_model.predict_proba(vect)[:,1]

vect = obs.transform(data)
pred_obs = obs_model.predict_proba(vect)[:,1]

vect = thr.transform(data)
pred_thr = thr_model.predict_proba(vect)[:,1]

vect = ins.transform(data)
pred_ins = ins_model.predict_proba(vect)[:,1]

vect = ide.transform(data)
pred_ide = ide_model.predict_proba(vect)[:,1]

print('Toxic: ',pred_tox[0])
print('Severe: ',pred_sev[0])
print('Obscene: ',pred_obs[0])
print('Threat: ',pred_thr[0])
print('Insult: ',pred_ins[0])
print('Identity hate: ',pred_ide[0])

run('multilingual')