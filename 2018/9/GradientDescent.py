import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# https://www.mockaroo.com/
train = np.loadtxt('./2018/9/ResearchRandomizer.csv',delimiter=',',dtype='int',skiprows=1)
df=pd.DataFrame(train)
print(df)
train_x = train[:,0]
train_y = train[:,1]
plt.scatter(train_x, train_y)
plt.subplot(2,1,1)

#標準化
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x-mu) / sigma
train_z = standardize(train_x)
print(train_z)
plt.scatter(train_z, train_y)
plt.subplot(2,1,2)
plt.show()