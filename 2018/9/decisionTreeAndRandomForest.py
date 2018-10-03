import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

spaceData = pd.read_csv( './2018/9/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
spaceData.columns = ['objid','ra','dec','u','g','r','i',
                  'z','run','rerun','camcol','field','specobjid','class','redshift','plate','mjd','fiberid']
print ("Dataset Length: ", len(spaceData)) 
print ("Dataset Shape: ", spaceData.shape)
# Printing the dataset obseravtions 
print (spaceData.head(5)) 

y = spaceData.pop('class')
print(y)
x = spaceData

# 分割
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
print("DescisionTree accuracy: ")
print(clf_gini.score(X_test, y_test))

DecisionTreePicture = DecisionTreeClassifier(max_depth=10)
DecisionTreePicture.fit(X_train, y_train)
dot_data = export_graphviz(DecisionTreePicture, out_file=None) 
graph = graphviz.Source(dot_data)
print(graph)

rf = RandomForestClassifier(criterion='gini', n_estimators=1000,min_samples_split=12,min_samples_leaf=1,oob_score=True,random_state=100,n_jobs=-1) 
rf = RandomForestClassifier(n_estimators=3)
rf.fit(X_train, y_train)
print("RandomForest accuracy: ")
print(rf.score(X_test, y_test))

estimator = rf.estimators_[2]
dot_data = export_graphviz(estimator, out_file=None) 
graph = graphviz.Source(dot_data)
print(graph)