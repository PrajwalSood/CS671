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