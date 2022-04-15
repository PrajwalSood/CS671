from gc import callbacks
from unittest import result
import tensorflow as tf
import numpy as np
import pandas as pd
from data_loader import get_data, get_data_all
from models import fcnn, autoencoder1, autoencoder3
from utils import plot_history
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
      self.prev_epoch_loss = self.model.history.history['val_accuracy'][epoch-1]

# SGD Batch size 1
BS = 1
models = [[64,32,16], [128,64,32], [256,128,64]]

results = pd.DataFrame(columns=['model','epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
train, val, test = get_data(BS)

for i in models:
  model = fcnn(i)
  # if model exists at 'models/' + f'fcnn_{i[0]}_{i[1]}_{i[2]}' + '.h5' load model evaluate on train and val set and write in results
  if exists(f'models/fcnn_{i[0]}_{i[1]}_{i[2]}.h5'):
    model.load_weights(f'models/fcnn_{i[0]}_{i[1]}_{i[2]}.h5')
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.evaluate(train, verbose=0)
    model.evaluate(val, verbose=0)
    results.loc[len(results)] = [model.name, 0, model.history.history['val_accuracy'][0], model.history.history['val_loss'][0], model.history.history['accuracy'][0], model.history.history['val_accuracy'][0]]
  else:
    # model.summary()
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train, validation_data = val, epochs=100, callbacks = [myCallback()])
    plot_history(history, f'fcnn_{i[0]}_{i[1]}_{i[2]}')
    model.save('models/' + f'fcnn_{i[0]}_{i[1]}_{i[2]}' + '.h5')
    results.to_csv('results_sgd.csv')


# SGD with batch size as all training samples
models = [[64,32,16], [128,64,32], [256,128,64]]

results = pd.DataFrame(columns=['model','epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
train, val, test = get_data_all()

for i in models:
  model = fcnn(i)
  # if model exists at 'models/' + f'fcnn_{i[0]}_{i[1]}_{i[2]}' + '.h5' load model evaluate on train and val set and write in results
  if exists(f'models/fcnn_{i[0]}_{i[1]}_{i[2]}all.h5'):
    model.load_weights(f'models/fcnn_{i[0]}_{i[1]}_{i[2]}all.h5')
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    model.evaluate(train, verbose=0)
    model.evaluate(val, verbose=0)
    results.loc[len(results)] = [model.name, 0, model.history.history['val_accuracy'][0], model.history.history['val_loss'][0], model.history.history['accuracy'][0], model.history.history['val_accuracy'][0]]
  else:
    # model.summary()
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train, validation_data = val, epochs=100, callbacks = [myCallback()])
    plot_history(history, f'fcnn_{i[0]}_{i[1]}_{i[2]}_all')
    model.save('models/' + f'fcnn_{i[0]}_{i[1]}_{i[2]}' + 'all.h5')
    results.to_csv('results_sgd_all.csv')

results = pd.DataFrame(columns=['model','epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

# SGD with momentum (NAG) batch size 32
models = [[64,32,16], [128,64,32], [256,128,64]]
BS = 32
train, val, test = get_data(BS)

for i in models:
  model = fcnn(i)
  # if model exists at 'models/' + f'fcnn_{i[0]}_{i[1]}_{i[2]}' + '.h5' load model evaluate on train and val set and write in results
  if exists(f'models/fcnn_{i[0]}_{i[1]}_{i[2]}_NAG.h5'):
    model.load_weights(f'models/fcnn_{i[0]}_{i[1]}_{i[2]}_NAG.h5')
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    model.evaluate(train, verbose=0)
    model.evaluate(val, verbose=0)
    results.loc[len(results)] = [model.name, 0, model.history.history['val_accuracy'][0], model.history.history['val_loss'][0], model.history.history['accuracy'][0], model.history.history['val_accuracy'][0]]
  else:
    # model.summary()
    opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov = True)
    model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train, validation_data = val, epochs=100, callbacks = [myCallback()])
    plot_history(history, f'fcnn_{i[0]}_{i[1]}_{i[2]}_NAG')
    model.save('models/' + f'fcnn_{i[0]}_{i[1]}_{i[2]}' + '_NAG.h5')
    results.to_csv('results_sgd_NAG.csv')
  
#  RMSProp algorithm – (batch_size=32) Consider momentum parameter as 0.9, learning rate as 0.001 and β = 0.99 for RMSProp. 
results = pd.DataFrame(columns=['model','epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

for i in models:
  model = fcnn(i)
  # if model exists at 'models/' + f'fcnn_{i[0]}_{i[1]}_{i[2]}' + '.h5' load model evaluate on train and val set and write in results
  if exists(f'models/fcnn_{i[0]}_{i[1]}_{i[2]}_RMSprop.h5'):
    model.load_weights(f'models/fcnn_{i[0]}_{i[1]}_{i[2]}_RMSprop.h5')
    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.evaluate(train, verbose=0)
    model.evaluate(val, verbose=0)
    results.loc[len(results)] = [model.name, 0, model.history.history['val_accuracy'][0], model.history.history['val_loss'][0], model.history.history['accuracy'][0], model.history.history['val_accuracy'][0]]
  else:
    # model.summary()
    opt = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train, validation_data = val, epochs=100, callbacks = [myCallback()])
    plot_history(history, f'fcnn_{i[0]}_{i[1]}_{i[2]}_RMSprop')
    model.save('models/' + f'fcnn_{i[0]}_{i[1]}_{i[2]}' + '_RMSprop.h5')
    results.to_csv('results_sgd_RMSprop.csv')

# Adam optimizer – (batch_size=32) a. Consider β1 = 0.9, β2 = 0.999 and ε = 10-8 for Adam optimizer. 
results = pd.DataFrame(columns=['model','epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

for i in models:
  model = fcnn(i)
  # if model exists at 'models/' + f'fcnn_{i[0]}_{i[1]}_{i[2]}' + '.h5' load model evaluate on train and val set and write in results
  if exists(f'models/fcnn_{i[0]}_{i[1]}_{i[2]}_Adam.h5'):
    model.load_weights(f'models/fcnn_{i[0]}_{i[1]}_{i[2]}_Adam.h5')
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.evaluate(train, verbose=0)
    model.evaluate(val, verbose=0)
    results.loc[len(results)] = [model.name, 0, model.history.history['val_accuracy'][0], model.history.history['val_loss'][0], model.history.history['accuracy'][0], model.history.history['val_accuracy'][0]]
  else:
    # model.summary()
    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train, validation_data = val, epochs=100, callbacks = [myCallback()])
    plot_history(history, f'fcnn_{i[0]}_{i[1]}_{i[2]}_Adam')
    model.save('models/' + f'fcnn_{i[0]}_{i[1]}_{i[2]}' + '_Adam.h5')
    results.to_csv('results_sgd_Adam.csv')