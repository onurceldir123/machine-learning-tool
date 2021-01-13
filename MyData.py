import pandas as pd
from pandas import read_csv

class MyClass:
    def __init__(self):
        self.my_dataframe = my_dataframe
        self.col_names = col_names
        self.col_length = col_length
        self.selected_cols = selected_cols
        self.selected_algo = selected_algo
        self.percent = percent
        self.uniques = uniques
        self.types = types
        self.encoder = encoder
        self.model = model
        self.accuracy = accuracy
        self.filename = filename
        self.model_id = model_id
        self.model_name = model_name
    def predict(self, input):
        self.model.predict()
    @staticmethod
    def readcsv(filename):
        filename = "files/" + filename
        data = pd.read_csv(filename)        
        return data
class datainfo:
    def __init__(self):
        self.d_types = d_types
        self.d_names = d_names
        self.d_values = dict_values
        self.d_counts = d_counts
        self.d_dict_list = d_dict_list
        self.d_num_categorical = d_num_categorical
        self.d_missing_values = d_missing_values
    

class Algorithms:
    names = [
            'Logistic Regression',
            'Random Forest',
            'Support Vector Machine',
            'Decision Tree Classifier',
            'Gaussian Naive Bayes',
            'K-Nearest Neighbour',]
    def __init__(self):      
        self.selected_algo = selected_algo



    
