import numpy as np
import pandas as pd
import glob
from sklearn.utils import shuffle
import tqdm
from sklearn.preprocessing import OneHotEncoder

def load_hw(sequence_length=75):
    seq, labels = [], []

    files = glob.glob('data/handwriting/*/train/*.txt')
    files = shuffle(files)
    enc = OneHotEncoder(sparse=False)

    for i in files:
        data = pd.read_csv(i, sep=' ', header=None)
        X = data.iloc[:, 1:-1].values.reshape(-1,2).astype(object)
        
        seq.append(np.expand_dims(X, axis = 0))
        labels.append(i.split('\\')[-3])
    # labels = enc.fit_transform(np.array(labels).reshape(-1,1))
    
    for i in range(len(labels)):
        if labels[i]=='a':
            labels[i]=np.array([[1,0,0,0,0]]).astype(object)
        elif labels[i]=='ai':
            labels[i]=np.array([[0,1,0,0,0]]).astype(object)
        elif labels[i]=='bA':
            labels[i]=np.array([[0,0,1,0,0]]).astype(object)
        elif labels[i]=='chA':
            labels[i]=np.array([[0,0,0,1,0]]).astype(object)
        elif labels[i]=='dA':
            labels[i]=np.array([[0,0,0,0,1]]).astype(object)
    
    X_train, y_train = seq, np.array(labels)
    #repeat the same process for test data
    seq, labels = [], []
    files = glob.glob('data/handwriting/*/dev/*.txt')
    files = shuffle(files)

    for i in files:
        data = pd.read_csv(i, sep=' ', header=None)
        X = data.iloc[:, 1:-1].values.reshape(-1,2).astype(object)
        seq.append(np.expand_dims(X, axis = 0))
        labels.append(i.split('\\')[-3])
    for i in range(len(labels)):
        if labels[i]=='a':
            labels[i]=np.array([[1,0,0,0,0]]).astype(object)
        elif labels[i]=='ai':
            labels[i]=np.array([[0,1,0,0,0]]).astype(object)
        elif labels[i]=='bA':
            labels[i]=np.array([[0,0,1,0,0]]).astype(object)
        elif labels[i]=='chA':
            labels[i]=np.array([[0,0,0,1,0]]).astype(object)
        elif labels[i]=='dA':
            labels[i]=np.array([[0,0,0,0,1]]).astype(object)
    #   print(labels)
    #   labels = enc.transform(np.array(labels).reshape(-1,1))
    X_test, y_test = seq, np.array(labels)

    return X_train[:int(len(X_train)*0.8)], y_train[:int(len(X_train)*0.8)], X_train[int(len(X_train)*0.8):], y_train[int(len(X_train)*0.8):], X_test, y_test

def data_loader_cv():
    seq, labels = [], []
    files = glob.glob('data/CV/*/train/*.mfcc')
    files = shuffle(files)
    enc = OneHotEncoder(sparse=False)

    for i in files:
        data = pd.read_csv(i, sep=' ', header=None)
        X = data.iloc[:, 1:-1].values.astype(object)
        seq.append(np.expand_dims(X, axis = 0))
        labels.append(i.split('\\')[-3])
    # labels = enc.fit_transform(np.array(labels).reshape(-1,1))
    
    for i in range(len(labels)):
        if labels[i]=='hI':
            labels[i]=np.array([[1,0,0,0,0]]).astype(object)
        elif labels[i]=='ne':
            labels[i]=np.array([[0,1,0,0,0]]).astype(object)
        elif labels[i]=='ni':
            labels[i]=np.array([[0,0,1,0,0]]).astype(object)
        elif labels[i]=='nii':
            labels[i]=np.array([[0,0,0,1,0]]).astype(object)
        elif labels[i]=='pa':
            labels[i]=np.array([[0,0,0,0,1]]).astype(object)
    
    X_train, y_train = seq, np.array(labels)

    seq, labels = [], []
    files = glob.glob('data/CV/*/test/*.mfcc')
    files = shuffle(files)
    enc = OneHotEncoder(sparse=False)

    for i in files:
        data = pd.read_csv(i, sep=' ', header=None)
        X = data.iloc[:, 1:-1].values.astype(object)
        seq.append(np.expand_dims(X, axis = 0))
        labels.append(i.split('\\')[-3])
    # labels = enc.fit_transform(np.array(labels).reshape(-1,1))
    
    for i in range(len(labels)):
        if labels[i]=='hI':
            labels[i]=np.array([[1,0,0,0,0]]).astype(object)
        elif labels[i]=='ne':
            labels[i]=np.array([[0,1,0,0,0]]).astype(object)
        elif labels[i]=='ni':
            labels[i]=np.array([[0,0,1,0,0]]).astype(object)
        elif labels[i]=='nii':
            labels[i]=np.array([[0,0,0,1,0]]).astype(object)
        elif labels[i]=='pa':
            labels[i]=np.array([[0,0,0,0,1]]).astype(object)
    
    X_test, y_test = seq, np.array(labels)

    return X_train[:int(len(X_train)*0.8)], y_train[:int(len(X_train)*0.8)], X_train[int(len(X_train)*0.8):], y_train[int(len(X_train)*0.8):], X_test, y_test