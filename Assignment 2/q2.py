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

train, val, test = only_img()

ctrain, cval, ctest = get_data(64)

models = [[16], [32], [64]]

results = pd.DataFrame(columns=['model','epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
results_clf = pd.DataFrame(columns=['model','epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

for i in models:
  ae = autoencoder1(i, encoder_only = False)
  ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
  ae.summary()
  hist = ae.fit(train, epochs=50, batch_size=32, validation_data=val, callbacks=[myCallback()])

  plot_historyae(hist, 'ae1_'+str(i))
  # inp = ae.layers[0]
  # ae.layers[1].trainable = (False)
  # hl1 = ae.layers[1](inp)
  # classification_layer = tf.keras.layers.Dense(5, activation='softmax', name = "output")(hl1)

  # clf_model = tf.keras.models.Model(inputs=inp, outputs=classification_layer)
  # clf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  # hist = clf_model.fit(ctrain, epochs=50, batch_size=32, validation_data=val, callbacks=[myCallback()])

  # plot_history(hist, 'clf_'+str(i))

  
  # results_clf = results_clf.append({'model': 'ae1', 'epochs': i, 'train_loss': clf_model.history.history['loss'][-1], 'val_loss': clf_model.history.history['val_loss'][-1], 'train_acc': clf_model.history.history['accuracy'][-1], 'val_acc': clf_model.history.history['val_accuracy'][-1]}, ignore_index=True)

  ae.save(f'ae/ae1_{i}.h5')
  # clf_model.save(f'ae/ae1_clf_{i[0]}.h5')
  # test_batch = iter(next(val))
  # visualise_pred_batch(ae, test_batch, 'ae1_'+str(i))



results.to_csv('ae1_resutls.csv')
# results_clf.to_csv('ae1_clf_results.csv')