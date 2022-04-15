from gc import callbacks
from unittest import result
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from data_loader import get_data, get_data_all, only_img_n
from models import fcnn, autoencoder1, autoencoder3
from utils import plot_historyae, visualise_pred_batch
from os.path import exists

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

train, val, test = only_img_n()

ctrain, cval, ctest = get_data(64)

models = [[16], [32], [64]]

results = pd.DataFrame(columns=['model','epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
results_clf = pd.DataFrame(columns=['model','epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

for i in models:
  ae = autoencoder1(i, encoder_only = False)
  ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
  ae.summary()
  hist = ae.fit(train, epochs=50, batch_size=32, validation_data=val, callbacks=[myCallback()])

  plot_historyae(hist, 'ae1_dn_'+str(i))

  ae.save(f'ae/ae1_dn_{i}.h5')




results.to_csv('ae1_dn_resutls.csv')
# results_clf.to_csv('ae1_dn_clf_results.csv')