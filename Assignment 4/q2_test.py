import os
from pprint import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
from data_loader import load_CV, data_loader_cv
from tensorflow import keras
import time
from sklearn.metrics import classification_report

###############     for CV dataset

X_train, y_train, X_val, y_val, X_test, y_test = data_loader_cv()

############### Load Rnn

model = tf.keras.models.load_model('models/RNN_CV/model.h5')


####################### train
pred = []
for i in X_train:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_train = np.argmax(y_train, axis = 2)

cf = classification_report(y_train, pred)
print(cf)
with open('metrics/RNN_CV/train_report.txt', 'w') as f:
  f.write(cf) 


################## val
pred = []
for i in X_val:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_val = np.argmax(y_val, axis = 2)

cf = classification_report(y_val, pred)
print(cf)
with open('metrics/RNN_CV/val_report.txt', 'w') as f:
  f.write(cf) 


################## test
pred = []
for i in X_test:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_test = np.argmax(y_test, axis = 2)

cf = classification_report(y_test, pred)
print(cf)
with open('metrics/RNN_CV/test_report.txt', 'w') as f:
  f.write(cf) 

############################ LSTM

############### Load Rnn

model = tf.keras.models.load_model('models/LSTM_CV/model.h5')


####################### train
pred = []
for i in X_train:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_train = np.argmax(y_train, axis = 2)

cf = classification_report(y_train, pred)
print(cf)
with open('metrics/LSTM_CV/train_report.txt', 'w') as f:
  f.write(cf) 


################## val
pred = []
for i in X_val:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_val = np.argmax(y_val, axis = 2)

cf = classification_report(y_val, pred)
print(cf)
with open('metrics/LSTM_CV/val_report.txt', 'w') as f:
  f.write(cf) 


################## test
pred = []
for i in X_test:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_test = np.argmax(y_test, axis = 2)

cf = classification_report(y_test, pred)
print(cf)
with open('metrics/LSTM_CV/test_report.txt', 'w') as f:
  f.write(cf) 