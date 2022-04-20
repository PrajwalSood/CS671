from ast import mod
import xdrlib
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import get_data, get_data_all, only_img_n, only_img
from utils import plot_historyae, visualise_pred_batch

from utils import visualise_pred_batch

model = tf.keras.models.load_model('ae/ae1_[64].h5')

model.summary()
enc_layer = model.layers[2].weights[0]

enc_layer = np.array(enc_layer).T

for i in range(enc_layer.shape[0]):
  s = sum(enc_layer[i])
  enc_layer[i] = enc_layer[i] / s

#make a figure with 64 subplots in 8x8
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i in range(64):
  axes[i//8, i%8].imshow(enc_layer[i].reshape(28,28))
  axes[i//8, i%8].axis('off')

plt.show()


train, val, test = only_img()
visualise_pred_batch(model, next(iter(val)))

model = tf.keras.models.load_model('ae/ae1_[64]_dn.h5')

model.summary()
enc_layer = model.layers[2].weights[0]

enc_layer = np.array(enc_layer).T

for i in range(enc_layer.shape[0]):
  s = sum(enc_layer[i])
  enc_layer[i] = enc_layer[i] / s

#make a figure with 64 subplots in 8x8
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i in range(64):
  axes[i//8, i%8].imshow(enc_layer[i].reshape(28,28))
  axes[i//8, i%8].axis('off')

plt.show()

train, val, test = only_img()
visualise_pred_batch(model, val)