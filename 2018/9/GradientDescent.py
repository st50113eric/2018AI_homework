import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# https://www.mockaroo.com/ # CSV排序完成
train = np.loadtxt('./2018/9/ResearchRandomizer.csv',delimiter=',',dtype='int',skiprows=1)
df=pd.DataFrame(train)
print(df)
train_x = train[:,0]
train_y = train[:,1]
plt.scatter(train_x, train_y)
plt.show()
#標準化
mu = train_x.mean()
# 平均值
sigma = train_x.std()
def standardize(x):
    return (x-mu) / sigma
train_z = standardize(train_x)
print(train_z)
plt.scatter(train_z, train_y)
plt.show()

#參考初始化 #隨機樣本位於[0, 1)中
theta0 = np.random.rand()
theta1 = np.random.rand()

#線性回歸函數後的預測函數
def f(x):
    return theta0 + theta1 * x

#cost function
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

#學習率
ETA = 1e-3

#誤差的差分
diff = 1

#更新回數
count = 0

#參數調校更新直到誤差差分小於0.01
error = E(train_z,train_y)
while diff > 1e-2  :
    #更新結果暫時儲存
    tmp_theta0 = theta0 - ETA * np.sum((f(train_z)-train_y))
    tmp_theta1 = theta1 - ETA * np.sum((f(train_z)-train_y) * train_z)
    
    #更新參數
    theta0 = tmp_theta0
    theta1 = tmp_theta1
    
    #計算與"前"一項誤差的差分
    current_error = E(train_z,train_y)
    diff = error - current_error
    error = current_error
    
    #運算過程
    count += 1
    log = '{}次: theta0 = {:.3f}, theta1 = {:.3f}, 差分 = {:.4f}'
    print(log.format(count,theta0,theta1,diff))

result_x = np.linspace(-2, 2, 1000).reshape(-1, 1)
result_y = f(result_x)
plt.plot(result_x, result_y, 'r-')
plt.scatter(train_z, train_y)
plt.show()