# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:05:08 2022

@author: prajw
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models import Network
from models import FCLayer
from models import ActivationLayer
from models import tanh, tanh_prime
from models import mse, mse_prime
from models import sigmoid, sigmoid_prime
from models import relu, relu_prime
import matplotlib.pyplot as plt
import seaborn as sns

#placeholder for data (1500x2)
X = pd.read_csv('data/Regression/UnivariateData/4.csv', header=None)
#read data from data/Classification/LS_Group04/Class*.txt


# y = [0 for i in range(1500)]
# for i in range(500, 1000):
#     y[i] = 1

# for i in range(1000, 1500):
#     y[i] = 2

# def vectorized_result(j):
#     e = np.zeros((3, 1))
#     e[j] = 1.0
#     return e
y = X.values[:,-1]

ym = y.min()
yM = y.max()

y = (y-ym)/(yM-ym)

X = X.values[:,:-1]
X[:,0] = (X[:,0] - min(X[:,0]))/(max(X[:,0]) -min(X[:,0]))
# X[:,1] = (X[:,1] - min(X[:,1]))/(max(X[:,1]) -min(X[:,1]))

# X = [np.reshape(x, (2, 1)) for x in X]
X = X.reshape(-1,1,1)
# y = np.array([vectorized_result(i) for i in y])
y = y.reshape(-1,1,1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=7)

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.25, random_state=7)



#%%

nets = [[1,4,1],
        [1,8,1],
        [1,16,1],
        [1,32,1]]

#%%
# y = np.array(y).reshape(-1,1)

acc_t = []


for i in nets:
# network
    net = Network()
    net.add(FCLayer(i[0], i[1]))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))
    net.add(FCLayer(i[1], i[2]))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    # train
    net.use(mse, mse_prime)
    net.fit(X_train, y_train, epochs=100, learning_rate=5)
    
    y_pred_val = np.array(net.predict(X_val))
    acc = np.sum(np.square(y_pred_val-y_val))
    acc_t.append(np.sqrt(acc))

    print('\n')
    print('#'*20)
    print('Validation')
    print('Network: ', i)
    print('Accuracy: ', acc)
    print('#'*20)

    y_pred = np.array(net.predict(X_test))
    acc = np.sum(np.square(y_pred-y_test))

    print('#'*20)
    print('Test')
    print('Network: ', i)
    print('Accuracy: ', acc)
    print('#'*20)

#%%
best_i = np.argmin(acc_t)

net = Network()
net.add(FCLayer(nets[best_i][0], nets[best_i][1]))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(nets[best_i][1], nets[best_i][2]))
net.add(ActivationLayer(sigmoid, sigmoid_prime))

# train
net.use(mse, mse_prime)
net.fit(X_train, y_train, epochs=100, learning_rate=1)

y_pred = net.predict(X_test)
acc = np.sum(np.square(y_pred-y_test))

print("Best Error at {} hidden dims, acc:" .format(nets[best_i][1]), acc)

history = net.error

#plot history
plt.plot(history)
plt.title('Model loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.scatter(y_pred, y_test)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()
