import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = np.loadtxt('ResearchRandomizer.csv',delimiter=',',dtype='int',skiprows=1)
df=pd.DataFrame(train)
print(df)
train_x = train[:,0]
train_y = train[:,1]
plt.scatter(train_x, train_y)
plt.show()