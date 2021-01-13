import os
from flask import *
from werkzeug.utils import secure_filename
from calculations import calc_correlations, calc_chi2, calc_mutual, calc_f, data_information
import pandas as pd
from functools import wraps, update_wrapper
from MyData import MyClass as mc
from MyData import datainfo as dt
from MyData import Algorithms as algo_class
from algorithms import algorithm_apply, create_model
from datetime import datetime
import numpy as np
import joblib
from bigml import model
from mongo import loadmodelinfo, save_model, savetomongodb
import pickle
UPLOAD_FOLDER = './files/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import api
@app.route('/')
def hello():
    return render_template("index.html")
    
@app.route('/test')
def test():
    return render_template("test.html")

@app.route('/fileuploaded', methods=['POST', 'GET'])
def fileuploaded():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        mc.filename = filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        data = mc.readcsv(filename)
        #data = data.replace({'?': np.nan})
        data = pd.DataFrame(data=data)       
        mc.my_dataframe = data       
        dt.d_types, dt.d_names, dt.d_values, dt.d_counts, dt.d_dict_list, dt.d_num_categorical, dt.d_missing_values = data_information(mc.my_dataframe)
        mc.my_dataframe = mc.my_dataframe.dropna()
        return render_template("fileuploaded.html",
                                name=file.filename, 
                                column_names=mc.my_dataframe.columns.values,
                                row_data=list(mc.my_dataframe.head(n=400).values.tolist()), 
                                zip=zip,
                                d_types = dt.d_types, 
                                d_names=dt.d_names, 
                                d_dict_list = dt.d_dict_list, 
                                d_len = len(dt.d_counts),
                                d_missing_values = dt.d_missing_values
                                )


@app.route('/correlation/', methods=['POST', 'GET'])
def correlation():
    return render_template("correlation.html",  col_names = mc.my_dataframe.columns[:len(mc.my_dataframe.columns)-1])



@app.route('/algorithms/', methods=['POST', 'GET'])
def algorithms():
    if request.method == 'POST':
        mc.selected_cols = request.form.getlist('feature_names')
        return render_template("algorithms.html", 
                                algos = algo_class.names)

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        mc.selected_algo = request.form.get('algorithm')
        mc.percent = request.form.get('percent')
        print(mc.percent)
        mc.uniques, mc.types, mc.col_uniques, mc.col_types = algorithm_apply(mc.selected_cols, mc.my_dataframe)
        mc.model, mc.encoder, mc.accuracy = create_model(mc.selected_algo, mc.my_dataframe, mc.percent, mc.selected_cols)
        filename = mc.filename
        save_model(mc)
        return render_template("predict.html", 
                            feature_names = mc.selected_cols, 
                            uniques = mc.uniques, 
                            types= mc.types, 
                            length = len(mc.selected_cols), 
                            accuracy = round(float(mc.accuracy*100),1),
                            id = mc.model_id)





@app.route('/prediction/predict', methods=['POST', 'GET'])
def prediction_with():
    if request.method == 'POST': 
        input_data = []
        for i in range(len(mc.selected_cols)):
            if request.form.get("select" + str(i)) == 'True':
                input_data.append(True)
            elif request.form.get("select" + str(i)) == 'False':
                input_data.append(False)
            else:              
                input_data.append(request.form.get("select" + str(i)))
        predictdf = pd.DataFrame(columns=mc.selected_cols)
        predictdf.loc[0] = input_data
        predictdf = mc.encoder.transform(predictdf)
        new_predict = mc.model.predict(predictdf)[0]
        return render_template("predicted.html", 
                            feature_names = mc.selected_cols, 
                            uniques = mc.uniques, 
                            types= mc.types, 
                            length = len(mc.selected_cols),
                            accuracy = round(float(mc.accuracy*100),1),
                            new_predict = new_predict,
                            id= mc.model_id)


@app.route('/plots/breast_cancer_data/correlation_matrix', methods=['GET', 'POST'])
def correlation_matrix():
    corr_graph = None
    corr_graph = calc_correlations(mc.my_dataframe) 
    return send_file(corr_graph,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/plots/breast_cancer_data/chi_2', methods=['GET', 'POST'])
def calculate_chi2():
    #this method helps load plots faster
    chi2_graph = None
    chi2_graph = calc_chi2(mc.my_dataframe)
    return send_file(chi2_graph,
                     attachment_filename='plot1.png',
                     mimetype='image/png')

@app.route('/plots/breast_cancer_data/mutual_information', methods=['GET', 'POST'])
def calculate_mut():
    #this method helps load plots faster
    mutual_graph = None
    mutual_graph = calc_mutual(mc.my_dataframe)
    return send_file(mutual_graph,
                     attachment_filename='plot2.png',
                     mimetype='image/png')


@app.route('/plots/breast_cancer_data/f_scores', methods=['GET', 'POST'])
def calculate_f():
    #this method helps load plots faster
    f_graph = None
    f_graph = calc_f(mc.my_dataframe)
    return send_file(f_graph,
                     attachment_filename='plot3.png',
                     mimetype='image/png')

@app.route('/loadmodel')
def loadmodel():
    return render_template("loadmodel.html", isdefined = None)

@app.route('/prediction_l', methods=['POST', 'GET'])
def prediction_l():
    if request.method == 'POST':
        id = request.form.get('load_id')
        mc.uniques, mc.selected_cols, mc.types, mc.accuracy,  mc.model_id, mc.model_name, mc.model, mc.encoder = loadmodelinfo(id)
        mc.model_name = mc.model_name.replace('files/', '')
        return render_template("predict.html", 
                            feature_names = mc.selected_cols, 
                            uniques = mc.uniques, 
                            types= mc.types, 
                            length = len(mc.selected_cols), 
                            accuracy = round(float(mc.accuracy*100),1),
                            id = mc.model_id,
                            model_name =  mc.model_name,
                            isdefined = None)

@app.route('/prediction_l/predict', methods=['POST', 'GET'])
def prediction_l_with():
    if request.method == 'POST': 
        input_data = []
        for i in range(len(mc.selected_cols)):
            if request.form.get("select" + str(i)) == 'True':
                input_data.append(True)
            elif request.form.get("select" + str(i)) == 'False':
                input_data.append(False)
            else:              
                input_data.append(request.form.get("select" + str(i)))
        predictdf = pd.DataFrame(columns=mc.selected_cols)
        predictdf.loc[0] = input_data
        predictdf = mc.encoder.transform(predictdf)
        new_predict = mc.model.predict(predictdf)[0]
        
        return render_template("predicted.html", 
                            feature_names = mc.selected_cols, 
                            uniques = mc.uniques, 
                            types= mc.types, 
                            length = len(mc.selected_cols),
                            accuracy = round(float(mc.accuracy*100),1),
                            new_predict = new_predict,
                            id= mc.model_id,
                            model_name =  mc.model_name,
                            isdefined = None)

@app.route('/api', methods=['GET'])
def api_doc():
    return render_template('api.html')



