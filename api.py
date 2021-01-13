#!flask/bin/python
from flask import Flask, jsonify
from flask import make_response
from flask import request
from flask_restful import reqparse, abort, Api, Resource
import pymongo
from json import dumps
from bson.objectid import ObjectId
import pickle
import pandas as pd
from flask_cors import CORS, cross_origin
from datetime import datetime
from app import app
CORS(app)
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["model_db"]
mycol = mydb["models"] 
print("Connected to database")

@app.route('/api/getall', methods=['GET'])
def get_products():
    output = []
    for s in mycol.find():
        output.append({
            '_id': str(s['_id']),
            'names': s['name'],
            'column_names': s['column_names'],
            'features': s['features'],
            'types': s['types'],
            'accuracy score': s['accuracy score'] 
        })
    return jsonify({'result' : output})



@app.route('/api/<string:my_id>', methods=['GET'])
def get_product(my_id):
    myquery = { "_id": ObjectId(my_id) }
    s = mycol.find_one(myquery)
    output = {
            '_id': str(s['_id']),
            'names': s['name'],
            'column_names': s['column_names'],
            'features': s['features'],
            'types': s['types'],
            'accuracy score': s['accuracy score'] 
        }
    return jsonify({
        'message': "Model fetched successfully!",
        'result' : output
        }) 

def saveToDb(data, pred, _id):
    model_col = mydb[_id]
    data['prediction'] = pred
    data['date'] = datetime.now()
    model_col.insert(data)

@app.route('/api/predict/<string:my_id>', methods=['POST'])
def predict(my_id):
    myquery = { "_id": ObjectId(my_id) }
    s = mycol.find_one(myquery)
    output = {
            '_id': str(s['_id']),
            'names': s['name'],
            'column_names': s['column_names'],
            'features': s['features'],
            'types': s['types'],
            'accuracy score': s['accuracy score'] 
        }
    input_data = request.json
    filename = 'models/' + my_id +'.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    encoder_path = "encoders/" +  my_id +'.pkl'
    loaded_encoder = pickle.load(open(encoder_path, 'rb'))

    predictdf = pd.DataFrame(columns=output['column_names'])
    predictdf.loc[0] = input_data
    predictdf = loaded_encoder.transform(predictdf)
    prediction = loaded_model.predict(predictdf)[0]
    print("\n")
    print(predictdf)
    prediction = str(prediction)
    print("Your prediction is: ", prediction)
    print("\n")
    saveToDb(dict(input_data), prediction, output["_id"])
    return jsonify({
        'message': "Prediction was successfully done.",
        'result' : prediction,
        'input' : input_data
        })
    
@app.route('/api/', methods=['POST'])
def post_products():
    #print(request.json['name'])
    model_id = mycol.insert(request.json)
    print("New model added!")
    return jsonify({
        'result' : "Model added successfully!",
        '_id' : str(model_id)
    })

@app.route('/api/modeldata/<string:my_id>', methods=['GET'])
def get_modeldata(my_id):
    output = []
    model_col = mydb[my_id]
    cursor = model_col.find({},{'_id': 0})
    for document in cursor:
        print(document)
        output.append(document)
    return make_response(jsonify(output), 200)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'HTTP 404 Error': 'The content you looks for does not exist. Please check your request.'}), 404)
 
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE, OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True)#!flask/bin/python