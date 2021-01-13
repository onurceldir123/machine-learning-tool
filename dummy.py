import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder, scale
from calculations import load_dataset
from sklearn.model_selection import train_test_split
import category_encoders as ce

# Read in the CSV file and convert "?" to NaN
df = pd.read_csv("files/breast-cancer.csv")
type_counts = []
num_categorical = 0
missing_values = []
for i in range(len(df.columns)):
    if df[df.columns[i]].dtypes == object:
        type_counts.append(df[df.columns[i]].value_counts())
        num_categorical=num_categorical+1   
        missing_values.append(df[df.columns[i]].isnull().sum())   
    elif df[df.columns[i]].dtypes == bool:
        type_counts.append(df[df.columns[i]].value_counts())
        num_categorical=num_categorical+1   
        missing_values.append(df[df.columns[i]].isnull().sum())    
    else:
        type_counts.append(df[df.columns[i]].describe())
        missing_values.append(df[df.columns[i]].isnull().sum())    

print("-------------------")
print(len(df.columns))
print(num_categorical)
print(len(type_counts)-num_categorical)
print(type_counts[0].name)

print(df.isnull().sum())
"""
y = df[df.columns[len(df.columns)-1]]
print(X[0:1])

encoder = ce.OrdinalEncoder()
encoder.fit(X)
X = encoder.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
X_train = np.array(X_train).astype(np.float)
X_test = np.array(X_test).astype(np.float)
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
reg_rf = RandomForestClassifier()
reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)
print("forest : %s" % metrics.accuracy_score(y_test, y_pred))

from sklearn.svm import SVC
reg_svc = SVC()
reg_svc.fit(X_train, y_train)
y_pred = reg_svc.predict(X_test)
print("svc : %s" % metrics.accuracy_score(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
reg_knn = KNeighborsClassifier(3)
reg_knn.fit(X_train, y_train)
y_pred = reg_knn.predict(X_test)
print("Knn : %s" % metrics.accuracy_score(y_test, y_pred))


from sklearn import tree
reg_tree = tree.DecisionTreeClassifier()
reg_tree.fit(X_train, y_train)
y_pred = reg_tree.predict(X_test)
print("decision tree : %s" % metrics.accuracy_score(y_test, y_pred))

from sklearn import naive_bayes
reg_gaus = naive_bayes.GaussianNB()
reg_gaus.fit(X_train, y_train)
y_pred = reg_gaus.predict(X_test)
print("bayes : %s" % metrics.accuracy_score(y_test, y_pred))

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
reg_quad = QuadraticDiscriminantAnalysis()
reg_quad.fit(X_train, y_train)
y_pred = reg_quad.predict(X_test)
print("quad : %s" % metrics.accuracy_score(y_test, y_pred))"""

#test data
"""
test0 = ['outlook', 'temperature', 'humidity', 'windy']
testdf = pd.DataFrame(columns=df.columns[0:len(df.columns)-1])
testdf.loc[0] = test0

testdf = encoder.transform(testdf)
print(testdf)

print(reg_rf.predict(testdf))
print(reg_svc.predict(testdf))
print(reg_knn.predict(testdf))
print(reg_tree.predict(testdf))
print(reg_gaus.predict(testdf))
print(reg_quad.predict(testdf))"""
