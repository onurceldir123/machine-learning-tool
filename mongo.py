from flask import Flask, send_file, render_template
from pprint import pprint
import pymongo
from bson.son import SON
import numpy as np
from bson.objectid import ObjectId
import pickle
import json
import category_encoders as ce
from pandas import read_csv
import pandas as pd
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["model_db"]
mycol = mydb["models"] 

def savetomongodb(selected_cols, types, uniques, filename, accuracy):
    uniques = uniques.tolist()
    types = types.tolist()
    my_col = {"name": filename, "column_names": selected_cols, "features": uniques, "types": types, "accuracy score": accuracy}
    _id = mycol.insert(my_col)
    return _id

def save_model(mc):
    filename = mc.filename
    mc.model_id = savetomongodb(mc.selected_cols, mc.types, mc.uniques, filename, mc.accuracy)   
    filename = "models/" + str(mc.model_id) +'.sav'
    pickle.dump(mc.model, open(filename, 'wb'))
    filename = "encoders/" + str(mc.model_id) +'.pkl'
    pickle.dump(mc.encoder, open(filename, 'wb'))

def loadmodelinfo(my_id):
    myquery = { "_id": ObjectId(my_id)  }
    mydoc = mycol.find(myquery)
    for x in mydoc:
        names = x
    my_features = x["features"]
    my_names = x["column_names"]
    my_types = x["types"]
    my_accuracy = x["accuracy score"]
    my_model_name = x["name"]
    my_model, my_encoder = load_model(str(my_id))   
    return my_features, my_names, my_types, my_accuracy, ObjectId(my_id),my_model_name, my_model, my_encoder



def load_model(my_id):
    filename = 'models/' + my_id +'.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    encoder_path = "encoders/" +  my_id +'.pkl'
    loaded_encoder = pickle.load(open(encoder_path, 'rb'))
    return loaded_model, loaded_encoder


