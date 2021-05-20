import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle

#for BERT
import pandas as pd
import argparse
import os
from detoxify import Detoxify

st.title('TOXIC COMMENTS CLASSFIER')    

st.write("""
# Explore different classifiers
""")

user_input = st.text_input('\nEnter Comment: ')


classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Random Forest', 'BERT')
)
#BERT CODE
def predict_bert(model_name,user_input):
    """Loads model from checkpoint or from model name and runs inference on the input_obj.
    Displays results as a pandas DataFrame object.
    If a dest_file is given, it saves the results to a txt file.
    """
    text = [user_input]
    if model_name is not None:
        res = Detoxify(model_name).predict(text)
    # else:
    #     res = Detoxify(checkpoint=from_ckpt).predict(text)

    res_df = pd.DataFrame(res, index=[text] if isinstance(text, str) else text).round(5)
    print(res_df)  
    return res

# RF CODE 
# Load the TF-IDF vocabulary specific to the category
if classifier_name=='Random Forest':
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

def predict_rf(cmmt):
    
    # Take a string input from user
    #user_input = request.form['text']
    data = [cmmt]

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

    return pred_tox[0], pred_sev[0], pred_obs[0], pred_thr[0], pred_ins[0], pred_ide[0]

if classifier_name=='Random Forest':
    t1,t2,t3,t4,t5,t6 = predict_rf(user_input)
    st.write('Toxic: ',t1)
    st.write('Severe Toxic: ',t2)
    st.write('Obscene: ',t3)
    st.write('Insult: ',t4)
    st.write('Threat: ',t5)
    st.write('Identity Hate: ',t6)
elif classifier_name=='BERT':
    res_bert=predict_bert('original',user_input)
    st.write('Toxic: ',res_bert["toxicity"][0])
    st.write('Severe Toxic: ',res_bert["severe_toxicity"][0])
    st.write('Obscene: ',res_bert["obscene"][0])
    st.write('Insult: ',res_bert["threat"][0])
    st.write('Threat: ',res_bert["insult"][0])
    st.write('Identity Hate: ',res_bert["identity_hate"][0])




# def get_dataset(name):
#     data = None
#     if name == 'Iris':
#         data = datasets.load_iris()
#     elif name == 'Wine':
#         data = datasets.load_wine()
#     else:
#         data = datasets.load_breast_cancer()
#     X = data.data
#     y = data.target
#     return X, y

# X, y = get_dataset(dataset_name)
# st.write('Shape of dataset:', X.shape)
# st.write('number of classes:', len(np.unique(y)))

# def add_parameter_ui(clf_name):
#     params = dict()
#     if clf_name == 'SVM':
#         C = st.sidebar.slider('C', 0.01, 10.0)
#         params['C'] = C
#     elif clf_name == 'KNN':
#         K = st.sidebar.slider('K', 1, 15)
#         params['K'] = K
#     else:
#         max_depth = st.sidebar.slider('max_depth', 2, 15)
#         params['max_depth'] = max_depth
#         n_estimators = st.sidebar.slider('n_estimators', 1, 100)
#         params['n_estimators'] = n_estimators
#     return params

# params = add_parameter_ui(classifier_name)

# def get_classifier(clf_name, params):
#     clf = None
#     if clf_name == 'SVM':
#         clf = SVC(C=params['C'])
#     elif clf_name == 'KNN':
#         clf = KNeighborsClassifier(n_neighbors=params['K'])
#     else:
#         clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
#             max_depth=params['max_depth'], random_state=1234)
#     return clf

# clf = get_classifier(classifier_name, params)
# #### CLASSIFICATION ####

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# acc = accuracy_score(y_test, y_pred)

# st.write(f'Classifier = {classifier_name}')
# st.write(f'Accuracy =', acc)

