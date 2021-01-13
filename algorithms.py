import pandas as pd 
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder, scale
from calculations import load_dataset
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import naive_bayes
def algorithm_apply(cols_selected, data_frame):
        col_unique = []
        col_type = []    
        for i in range(len(cols_selected)):
            if data_frame[cols_selected[i]].dtypes == object:
                #print("%s datatype: %s --> categories: %s"  %(cols_selected[i],"Categorical", data_frame[cols_selected[i]].unique()))
                col_unique.append(data_frame[cols_selected[i]].unique().tolist())
                col_type.append('categorical')
            elif data_frame[cols_selected[i]].dtypes == bool:
                #print("%s datatype: %s "  %(cols_selected[i],"categorical"))
                col_unique.append( data_frame[cols_selected[i]].unique().tolist())
                col_type.append('categorical') 
            else:
                #print("%s datatype: %s "  %(cols_selected[i],"categorical"))
                col_unique.append([str(data_frame[cols_selected[i]].min()), str(data_frame[cols_selected[i]].max())])
                col_type.append('numeric') 
        matrix = cols_selected, col_unique, col_type
        df_matrix = pd.DataFrame(matrix)
        df_matrix = df_matrix.transpose()
        return df_matrix[1], df_matrix[2], col_unique, col_type

def create_model(algo_selected, data_frame, percent, cols_selected):
    X = data_frame[cols_selected]
    y = data_frame[data_frame.columns[len(data_frame.columns)-1]]

    encoder = ce.OrdinalEncoder()
    encoder.fit(X)
    X = encoder.transform(X)
    
    percent = float(percent)*(0.01)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-percent, random_state = 2)
    X_train = np.array(X_train).astype(np.float)
    X_test = np.array(X_test).astype(np.float)


    if algo_selected=='Logistic Regression':
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        return model, encoder, accuracy
    elif algo_selected=='Random Forest':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        return model, encoder, accuracy
    elif algo_selected=='Support Vector Machine':
        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        return model, encoder, accuracy
    elif algo_selected=='Decision Tree Classifier':
        model = tree.DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        return model, encoder, accuracy
    elif algo_selected=='Gaussian Naive Bayes':
        model = naive_bayes.GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        return model, encoder, accuracy
    elif algo_selected=='K-Nearest Neighbour':
        model = KNeighborsClassifier(3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        return model, encoder, accuracy
    