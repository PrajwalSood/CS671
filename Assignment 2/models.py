from ast import mod
import xdrlib
import tensorflow as tf
import numpy as np
import pandas as pd

def fcnn(layersize):
  input_layer = tf.keras.layers.Input(shape=(28,28,1), name = "input")
  x = tf.keras.layers.Flatten()(input_layer)
  x = tf.keras.layers.Dense(layersize[0], activation='relu', name = "hidden_layer1")(x)
  x = tf.keras.layers.Dense(layersize[1], activation='relu', name = "hidden_layer2")(x)
  x = tf.keras.layers.Dense(layersize[2], activation='relu', name = "hidden_layer3")(x)
  x = tf.keras.layers.Dense(5, activation='softmax', name = "output")(x)
  model = tf.keras.models.Model(inputs=input_layer, outputs=x)
  return model

def autoencoder1(layersize, encoder_only = False, weights = None):
  input_layer = tf.keras.layers.Input(shape=(28,28,1), name = "input")
  x = tf.keras.layers.Flatten()(input_layer)
  hl = tf.keras.layers.Dense(layersize[0], activation='relu', name = "hidden_layer1")(x)
  hl = tf.keras.layers.Dropout(0.25)(hl)
  o = tf.keras.layers.Dense(784, activation='sigmoid', name = "output")(hl)
  o = tf.keras.layers.Reshape((28,28,1))(o)
  if encoder_only:
    model = tf.keras.models.Model(inputs=input_layer, outputs=hl)
  else:
    model = tf.keras.models.Model(inputs=input_layer, outputs=o)
  return model

def autoencoder3(layersize, encoder_only = False):
  input_layer = tf.keras.layers.Input(shape=(28,28,1), name = "input")
  x = tf.keras.layers.Flatten()(input_layer)
  hl1 = tf.keras.layers.Dense(layersize[0], activation='relu', name = "hidden_layer1")(x)
  hl1 = tf.keras.layers.Dropout(0.25)(hl1)
  hl2 = tf.keras.layers.Dense(layersize[1], activation='relu', name = "hidden_layer2")(hl1)
  hl2 = tf.keras.layers.Dropout(0.25)(hl2)
  hl3 = tf.keras.layers.Dense(layersize[2], activation='relu', name = "hidden_layer3")(hl2)
  hl3 = tf.keras.layers.Dropout(0.25)(hl3)
  o = tf.keras.layers.Dense(784, activation='sigmoid', name = "output")(hl3)
  o = tf.keras.layers.Reshape((28,28,1))(o)
  if encoder_only:
    model = tf.keras.models.Model(inputs=input_layer, outputs=hl2)
  else:
    model = tf.keras.models.Model(inputs=input_layer, outputs=o)
  return model