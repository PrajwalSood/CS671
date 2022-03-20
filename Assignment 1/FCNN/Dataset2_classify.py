# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 16:34:17 2022

@author: prajw
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from models import Network
from models import FCLayer
from models import ActivationLayer
from models import tanh, tanh_prime
from models import mse, mse_prime
from models import sigmoid, sigmoid_prime
import matplotlib.pyplot as plt

#placeholder for data (1500x2)
X = pd.read_csv('data/Classification/NLS_Group04.txt', sep=' ', header=None, skiprows = 1).values[:,:2]
#read data from data/Classification/LS_Group04/Class*.txt


y = [0 for i in range(1800)]
for i in range(300, 800):
    y[i] = 1

for i in range(800, 1800):
    y[i] = 2

def vectorized_result(j):
    e = np.zeros((3, 1))
    e[j] = 1.0
    return e

X[:,0] = (X[:,0] - min(X[:,0]))/(max(X[:,0]) -min(X[:,0]))
X[:,1] = (X[:,1] - min(X[:,1]))/(max(X[:,1]) -min(X[:,1]))

# X = [np.reshape(x, (2, 1)) for x in X]
X = X.reshape(-1,1,2)
y = np.array([vectorized_result(i) for i in y])
y = y.reshape(-1,1,3)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=7)

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.25, random_state=7)



#%%

nets = [[2,4,2,3],
        [2,8,4,3],
        [2,16,8,3],
        [2,32,8,3]]

#%%
# y = np.array(y).reshape(-1,1)

acc_t = []

i= 1
for i in nets:
# network
    net = Network()
    net.add(FCLayer(i[0], i[1]))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))
    net.add(FCLayer(i[1], i[2]))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))
    net.add(FCLayer(i[2], i[3]))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    # train
    net.use(mse, mse_prime)
    net.fit(X_train, y_train, epochs=100, learning_rate=1)
    
    y_pred_val = np.argmax(np.array(net.predict(X_val)), axis = -1)
    acc = list(y_pred_val == np.argmax(y_val, axis = -1)).count(True)/len(y_pred_val)
    acc_t.append(acc)

    print('\n')
    print('#'*20)
    print('Validation')
    print('Network: ', i)
    cm = confusion_matrix(np.argmax(y_val, axis = -1), y_pred_val)
    print('Confusion Matrix: \n', cm)
    print('Accuracy: ', acc)
    print('#'*20)

    y_pred = np.argmax(np.array(net.predict(X_test)), axis = -1)
    acc = list(y_pred == np.argmax(y_test, axis = -1)).count(True)/len(y_pred)

    print('#'*20)
    print('Test')
    print('Network: ', i)
    cm = confusion_matrix(np.argmax(y_test, axis = -1), y_pred)
    print('Confusion Matrix: \n', cm)
    print('Accuracy: ', acc)
    print('#'*20)

best_i = np.argmax(acc_t)

net = Network()
net.add(FCLayer(nets[best_i][0], nets[best_i][1]))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(nets[best_i][1], nets[best_i][2]))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(nets[best_i][2], nets[best_i][3]))
net.add(ActivationLayer(sigmoid, sigmoid_prime))


# train
net.use(mse, mse_prime)
net.fit(X_train, y_train, epochs=100, learning_rate=1)

y_pred = np.argmax(np.array(net.predict(X_test)), axis = -1)
acc = list(y_pred == np.argmax(y_test, axis = -1)).count(True)/len(y_pred)

print("Best Acc at {} hidden dims, acc:" .format(nets[best_i][1:3]), acc)

#%%
history = net.error

#plot history
plt.plot(history)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.show()


# Decision Boundary

plt.plot(X[:300,:,0], X[:300,:,1], 'o', color='r')
plt.plot(X[300:800,:,0], X[300:800,:,1], 'o', color='y')
plt.plot(X[800:1800,:,0], X[800:1800,:,1], 'o', color='b')

x_min, x_max = X[:,:,0].min() - 0.5, X[:,:,0].max() + 0.5
y_min, y_max = X[:,:,1].min() - 0.5, X[:,:,1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

Z = np.argmax(np.array(net.predict(np.c_[xx.ravel(), yy.ravel()])), axis = -1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
plt.show()
