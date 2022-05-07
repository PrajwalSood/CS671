import re
import numpy as np
import matplotlib.pyplot as plt
import cv2
from models import Network, ConvLayer, PoolingLayer, FCLayer, ActivationLayer, FlattenLayer
from models import tanh, tanh_prime, mse, mse_prime, sigmoid, sigmoid_prime, relu, relu_prime
import glob

train = glob.glob('data/train/*/*jpg')
train_l = [1,0,0] * 50 + [0,1,0]*50 + [0,0,1]*50
val = glob.glob('data/val/*/*jpg')

test = glob.glob('data/test/*/*jpg')

for i in range(len(train)):
    train[i] = cv2.imread(train[i])
    train[i] = cv2.resize(train[i], (224, 224))
    train[i] = cv2.cvtColor(train[i], cv2.COLOR_BGR2GRAY)
    train[i] = train[i]/255.

train = np.array(train)

for i in range(len(val)):
    val[i] = cv2.imread(val[i])
    val[i] = cv2.resize(val[i], (224, 224))
    val[i] = cv2.cvtColor(val[i], cv2.COLOR_BGR2GRAY)
    val[i] = val[i]/255.

val = np.array(val)

for i in range(len(test)):
    test[i] = cv2.imread(test[i])
    test[i] = cv2.resize(test[i], (224, 224))
    test[i] = cv2.cvtColor(test[i], cv2.COLOR_BGR2GRAY)
    test[i] = test[i]/255.

test = np.array(test)

net = Network()
net.add(ConvLayer((224,224,1), (32,3,3), 5, 1, relu, relu_prime))
net.add(ConvLayer((222,222,32), (64,3,3), 5, 1, relu, relu_prime))
net.add(PoolingLayer((220,220,64), (2,2)))
net.add(FlattenLayer())
net.add(FCLayer(220*220*64, 1024))
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(1024, 3))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.use(mse, mse_prime)
net.fit(train, train_l, epochs=100, learning_rate=1)
