from gc import callbacks
from unittest import result
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from data_loader import get_data, get_data_all, only_img
from models import fcnn, autoencoder1, autoencoder3
from utils import plot_historyae, visualise_pred_batch
from os.path import exists

def ae2clf(model_path, layers=1):
  '''
  Load an autoencoder model, if layers = 1, take first 2 layers and add a dense classification layer with 5 neurons, else take first 3 layers and add classification layer with 5 neurons
  '''
  model = tf.keras.models.load_model(model_path)
  if layers == 1:
    input_layer = tf.keras.layers.Input(shape=(28,28,1), name = "input")
    x = tf.keras.layers.Flatten()(input_layer)
    enc = model.layers[2]
    x = enc(x)
    x = tf.keras.layers.Dense(32, activation='sigmoid', name = "clf1")(x)
    x = tf.keras.layers.Dense(5, activation='softmax', name = "output")(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    model.layers[2].trainable = False
  
  elif layers == 3:
    input_layer = tf.keras.layers.Input(shape=(28,28,1), name = "input")
    x = tf.keras.layers.Flatten()(input_layer)
    enc = model.layers[2]
    enc.trainable = False
    x = enc(x)
    enc = model.layers[4]
    enc.trainable = False
    x = enc(x)
    x = tf.keras.layers.Dense(32, activation='sigmoid', name = "clf1")(x)
    x = tf.keras.layers.Dense(5, activation='softmax', name = "output")(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    model.layers[2].trainable = False
    model.layers[3].trainable = False
  
  return model

ae2clf('Assignment 2/ae/ae3_[64, 16, 64].h5', layers=3).summary()
ae2clf('Assignment 2/ae/ae1_[16].h5', layers=1).summary()

