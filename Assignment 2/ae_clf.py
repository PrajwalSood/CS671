from gc import callbacks
from unittest import result
import tensorflow as tf
import numpy as np
import pandas as pd
from data_loader import get_data, get_data_all
from models import fcnn, autoencoder1, autoencoder3
from utils import plot_historyae, ae2clf
from glob import glob

ae1_l = glob('ae/ae1_*.h5') 
ae3_l = glob('ae/ae3_*.h5')

train, val, test = get_data(64)

class myCallback(tf.keras.callbacks.Callback):
  def __init__(self) -> None:
      super().__init__()
      self.prev_epoch_loss = 0
      self.curr_epoch_loss = np.inf
  # break training if difference between average error of successive epochs fall below a threshold 10-4
  def on_epoch_end(self, epoch, logs={}):
    
    #prev epoch loss
    if epoch > 0:
      #current epoch loss
      if epoch > 2:
        self.curr_epoch_loss = self.model.history.history['val_loss'][epoch-1]
      #if difference between current and previous epoch loss is less than threshold, break training
      print('\n')
      # print(self.model.history.history)
      print('prev epoch loss: ', self.prev_epoch_loss)
      print('curr epoch loss: ', self.curr_epoch_loss)
      if abs(self.curr_epoch_loss - self.prev_epoch_loss) < 0.0001:
        print("breaking training")
        self.model.stop_training = True
        # write the results to results df
        results.loc[len(results)] = [self.model.name, epoch, self.model.history.history['val_accuracy'][epoch-1], self.model.history.history['val_loss'][epoch-1], self.model.history.history['accuracy'][epoch-1], self.model.history.history['val_accuracy'][epoch-1]]
    
    if epoch>2:
      self.prev_epoch_loss = self.model.history.history['val_loss'][epoch-1]

results = pd.DataFrame(columns=['model','epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

for i in ae1_l:
  print(i)
  model = ae2clf(i, layers=1)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(train, validation_data = val, epochs=200, callbacks=[myCallback()])
  plot_historyae(model.history, model.name)

results.to_csv('results/ae1_clf_results.csv')

results = pd.DataFrame(columns=['model','epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

for i in ae3_l:
  print(i)
  model = ae2clf(i, layers=3)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(train, validation_data = val, epochs=200, callbacks=[myCallback()])
  plot_historyae(model.history, model.name)

results.to_csv('results/ae3_clf_results.csv')