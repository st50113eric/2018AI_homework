import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
spaceData = pd.read_csv( './2018/9/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
print ("Dataset Length: ", len(spaceData)) 
print ("Dataset Shape: ", spaceData.shape) 
print (spaceData.head(5)) # Printing the dataset obseravtions 
x = pd.DataFrame(spaceData['data'], column=spaceData['objid','ra','dec','u','g','r','i','z'])
y = pd.DataFrame("target_names: "+ str(spaceData["class"]))
 
# processData = spaceData[['objid','ra','dec','u','g','r','i',
#                   'z','run','rerun','camcol','field','specobjid', 'class','redshift','plate','mjd','fiberid']]
# x = processData[['objid','ra','dec','u','g','r','i',
#                   'r','i','z']]
# y = processData[['class']]
# for i in range(0, len(processData)):
#     if (y['class'][i]) == "GALAXY":
#         y['class'][i] = 2
#     if (y['class'][i]) == "STAR":
#         y['class'][i] = 1
#     else:
#         y['class'][i] = 0
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state=0)
# tree.fit(X_train,y_train)
