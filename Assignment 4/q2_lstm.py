import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
from data_loader import data_loader_cv
from tensorflow import keras
import time

X_train, y_train, X_val, y_val, X_test, y_test  = data_loader_cv()
# X_train = X_train.astype(np.float32)
# y_train = y_train.astype(np.float32)

# train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#build rnn model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(256, return_sequences=False, input_shape=(None, 38), activation = 'tanh'))
model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
model.add(tf.keras.layers.Dense(5, activation='softmax'))


model.summary()


optimizer = keras.optimizers.Adam(learning_rate=1e-3)

loss_fn = keras.losses.CategoricalCrossentropy(from_logits = True)
train_acc_metric = keras.metrics.CategoricalAccuracy()

loss = []
acc = []

vloss = []
vacc = []

@tf.function(experimental_relax_shapes=True)
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

@tf.function(experimental_relax_shapes=True)
def val_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    return loss_value

epochs = 100
prev_loss = 1e10
for epoch in range(epochs):
  print("\nStart of epoch %d" % (epoch,))
  start_time = time.time()
  prev_loss = 1e10
  for step,i in enumerate(range(len(X_train))):
    loss_value = train_step(X_train[i].astype(np.float32), y_train[i].astype(np.float32))
    val_loss = val_step(X_train[i//5].astype(np.float32), y_train[i//5].astype(np.float32))
    if step % 170 == 0:
      print(
        "Training loss (for one batch) at step %d: %.4f \r"
        % (step, float(loss_value))
      )

      print(
        "Val loss (for one batch) at step %d: %.4f \r"
        % (step, float(val_loss))
      )
      loss.append(loss_value)
      vloss.append(val_loss)
      print("Seen so far: %s samples \r" % ((step + 1) * 1))
  diff = prev_loss - val_loss
  prev_loss = val_loss
  if abs(diff) < 1e-4:
    print('\n **************************** Breaking training *************************\n')
    print(
      "Training loss (for one batch) at step %d: %.4f \r"
      % (step, float(loss_value))
    )
    loss.append(loss_value)
    print("Seen so far: %s samples \r" % ((step + 1) * 1))
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    acc.append(float(train_acc))
    break

  train_acc = train_acc_metric.result()
  print("Training acc over epoch: %.4f" % (float(train_acc),))
  acc.append(float(train_acc))
  print("Time taken: %.2fs" % (time.time() - start_time))
  train_acc_metric.reset_states()


with open('metrics/LSTM_CV/acc.txt', 'w') as f:
  for i in acc:
    f.write(str(i) + '\n')

with open('metrics/LSTM_CV/loss.txt', 'w') as f:
  for i in loss:
    f.write(str(i.numpy()) + '\n')

with open('metrics/LSTM_CV/val_loss.txt', 'w') as f:
  for i in vloss:
    f.write(str(i.numpy()) + '\n')

model.save('models/LSTM_CV/model.h5')

acc = pd.read_csv('metrics/LSTM_CV/acc.txt', header=None).values
loss = pd.read_csv('metrics/LSTM_CV/loss.txt', header=None).values

# Plots
import matplotlib.pyplot as plt

#plot accuracy
plt.plot(acc)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('metrics/LSTM_CV/acc.png')
plt.show()

#plot loss
plt.plot(loss[::2], alpha = 0.5)
plt.plot(vloss[::2], c = 'orange', alpha = 0.5)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train loss', 'val loss'])
plt.savefig('metrics/LSTM_CV/loss.png')
plt.show()

############################################################
