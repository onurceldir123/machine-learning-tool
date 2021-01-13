from sklearn.pipeline import Pipeline
import joblib
import seaborn as sns
import seaborn as sns2
import pandas as pd
import matplotlib.pyplot as plt, mpld3
import matplotlib.pyplot as plt2
import plotly
import plotly.graph_objs as go
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score
import time
import numpy as np
from matplotlib import pyplot
import os
import category_encoders as ce
import io
from matplotlib.pyplot import axes
def calc_correlations(corr_param):
    plt2.clf()
    corr_matrix = corr_param.corr()
    num_cols = corr_matrix._get_numeric_data()
    sns2.heatmap(corr_matrix)
    plt2.legend(loc=0)
    bytes_image = io.BytesIO()
    plt2.savefig(bytes_image, format='png')   
    bytes_image.seek(0)
    return bytes_image


def load_dataset(param):
    # load the dataset as a pandas DataFrame
    data = param
    col_names = data.columns[0:len(data.columns)-1]
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:, -1]
    # format all fields as string
    X = X.astype(str)
    return X, y, col_names


# prepare input data
def prepare_inputs(X):
    oe = OrdinalEncoder()
    oe.fit(X)
    X = oe.transform(X)
    return X


# prepare target
def prepare_targets(y):
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    return y


# feature selection
def select_features(X, y):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X, y)
    X = fs.transform(X)
    return fs

# feature selection
def select_features2(X, y):
    fs2 = SelectKBest(score_func=mutual_info_classif, k='all')
    fs2.fit(X, y)
    X = fs2.transform(X)
    return fs2

# feature selection
def select_features3(X, y):
    fs3 = SelectKBest(f_classif, k='all')
    fs3.fit(X, y)
    X = fs3.transform(X)
    return fs3


def calc_chi2(chi2_param): 
    time.sleep(2)
    plt.clf()
    # load the dataset
    X, y, col_names = load_dataset(chi2_param)
    # prepare input data
    X = prepare_inputs(X)
    # prepare output data
    y = prepare_targets(y)
    # feature selection
    fs = select_features(X,y)  
    sns.barplot(x=col_names, y=fs.scores_)
    plt.xticks(rotation=90)
    plt.legend(loc=0)
    bytes_image1 = io.BytesIO()
    plt.savefig(bytes_image1, format='png')
    plt.clf()
    bytes_image1.seek(0)
    return bytes_image1

def calc_mutual(mutual_param):
    time.sleep(3)
    plt.clf()
    # load the dataset
    X, y, col_names = load_dataset(mutual_param)
    # prepare input data
    X= prepare_inputs(X)
    # prepare output data
    y = prepare_targets(y)
    # feature selection
    fs2 = select_features2(X, y)
    sns.barplot(x=col_names, y=fs2.scores_)
    plt.xticks(rotation=90)
    plt.legend(loc=0)
    bytes_image2 = io.BytesIO()
    plt.savefig(bytes_image2, format='png')
    plt.clf()
    bytes_image2.seek(0)
    return bytes_image2

def calc_f(f_param):
    time.sleep(4)
    plt.clf()
    # load the dataset
    X, y, col_names = load_dataset(f_param)
    # split into train and test sets
    # prepare input data
    X = prepare_inputs(X)
    # prepare output data
    y = prepare_targets(y)
    # feature selection
    fs3 = select_features3(X, y)
    sns.barplot(x=col_names, y=fs3.scores_)
    plt.xticks(rotation=90)
    plt.legend(loc=0)
    bytes_image3 = io.BytesIO()
    plt.savefig(bytes_image3, format='png')
    plt.clf()
    bytes_image3.seek(0)
    return bytes_image3


def data_information(df):
    types = []
    names = []
    values = []
    counts =[]
    dict_list = []
    missing_values = []
    num_categorical = 0
    for i in range(len(df.columns)):
        if df[df.columns[i]].dtypes == object:
            types.append('categorical')
            names.append(df.columns[i])
            values.append(df[df.columns[i]].value_counts().keys().tolist())
            counts.append(df[df.columns[i]].value_counts().tolist())
            dict_list.append(dict(zip(values[i], counts[i]))) 
            num_categorical=num_categorical+1
            missing_values.append(df[df.columns[i]].isnull().sum())  
        elif df[df.columns[i]].dtypes == bool:
            types.append('categorical')
            names.append(df.columns[i])
            values.append(df[df.columns[i]].value_counts().keys().tolist())
            counts.append(df[df.columns[i]].value_counts().tolist())
            dict_list.append(dict(zip(values[i], counts[i])))
            num_categorical=num_categorical+1 
            missing_values.append(df[df.columns[i]].isnull().sum())    
        else:
            types.append('continuous')
            names.append(df.columns[i])
            values.append(df[df.columns[i]].describe().keys().tolist())
            counts.append(df[df.columns[i]].describe().tolist())
            dict_list.append(dict(zip(values[i], counts[i])))
            missing_values.append(df[df.columns[i]].isnull().sum())  
    return types, names, values, counts, dict_list, num_categorical, missing_values

