import os
from pprint import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
from data_loader import load_CV2, data_loader_cv
from tensorflow import keras
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

###############     for CV dataset

X_train, y_train, X_val, y_val, X_test, y_test = data_loader_cv()

############### Load Rnn

model = tf.keras.models.load_model('models/RNN_CV2/model.h5')


####################### train
pred = []
for i in X_train:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_train = np.argmax(y_train, axis = 2)

cf = classification_report(y_train, pred)
print(cf)
with open('metrics/RNN_CV2/train_report.txt', 'w') as f:
  f.write(cf) 

cm = confusion_matrix(y_train, pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax) #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa']); ax.yaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa'])
plt.savefig('metrics/RNN_CV2/train_confusion_matrix.png')
plt.show()

################## val
pred = []
for i in X_val:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_val = np.argmax(y_val, axis = 2)

cf = classification_report(y_val, pred)
print(cf)
with open('metrics/RNN_CV2/val_report.txt', 'w') as f:
  f.write(cf) 

cm = confusion_matrix(y_val, pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax) #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa']); ax.yaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa'])
plt.savefig('metrics/RNN_CV2/val_confusion_matrix.png')
plt.show()


################## test
pred = []
for i in X_test:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_test = np.argmax(y_test, axis = 2)

cf = classification_report(y_test, pred)
print(cf)
with open('metrics/RNN_CV2/test_report.txt', 'w') as f:
  f.write(cf) 

cm = confusion_matrix(y_test, pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax) #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa']); ax.yaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa'])
plt.savefig('metrics/RNN_CV2/test_confusion_matrix.png')
plt.show()
############################ LSTM

############### Load LSTM

model = tf.keras.models.load_model('models/LSTM_CV2/model.h5')


####################### train
pred = []
for i in X_train:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_train = np.argmax(y_train, axis = 2)

cf = classification_report(y_train, pred)
print(cf)
with open('metrics/LSTM_CV2/train_report.txt', 'w') as f:
  f.write(cf) 

cm = confusion_matrix(y_train, pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax) #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa']); ax.yaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa'])
plt.savefig('metrics/LSTM_CV2/train_confusion_matrix.png')
plt.show()

################## val
pred = []
for i in X_val:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_val = np.argmax(y_val, axis = 2)

cf = classification_report(y_val, pred)
print(cf)
with open('metrics/LSTM_CV2/val_report.txt', 'w') as f:
  f.write(cf) 

cm = confusion_matrix(y_val, pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax) #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa']); ax.yaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa'])
plt.savefig('metrics/LSTM_CV2/val_confusion_matrix.png')
plt.show()



################## test
pred = []
for i in X_test:
  pred.append(np.argmax(model.predict(i.astype(np.float32))))

y_test = np.argmax(y_test, axis = 2)

cf = classification_report(y_test, pred)
print(cf)
with open('metrics/LSTM_CV2/test_report.txt', 'w') as f:
  f.write(cf) 

cm = confusion_matrix(y_test, pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax) #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa']); ax.yaxis.set_ticklabels(['hI', 'ne', 'ni', 'nii', 'pa'])
plt.savefig('metrics/LSTM_CV2/test_confusion_matrix.png')
plt.show()