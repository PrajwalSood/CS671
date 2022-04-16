import matplotlib.pyplot as plt
import tensorflow as tf

def plot_history(history, name):
  plt.figure()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.title(name)
  plt.savefig(f'plots/{name}.png')
  
  plt.figure()
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.title(name)
  plt.savefig(f'plots/{name}_acc.png')

def plot_historyae(history, name):
  plt.figure()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.title(name)
  plt.savefig(f'aeplots/{name}.png')
  
  plt.figure()
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.title(name)
  plt.savefig(f'aeplots/{name}_acc.png')


def visualise_pred_batch(model, batch, save = True, name = None):
  pred = model.predict(batch)
  plt.figure()
  plt.imshow(batch[0,:,:,0])
  plt.figure()
  plt.imshow(pred[0,:,:,0])
  plt.show()
  if save:
    if name is not None:
      plt.savefig(f'aeplots/{name}_pred.png')
    else:
      plt.savefig(f'aeplots/pred.png')

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