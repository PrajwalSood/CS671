import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
#import classifiaction report
from sklearn.metrics import classification_report
import cv2

AUTO = tf.data.experimental.AUTOTUNE

def get_data(BS):
  train_datagen = ImageDataGenerator(rescale = 1./255)
  validation_datagen = ImageDataGenerator(rescale = 1./255)
  test_datagen = ImageDataGenerator(rescale = 1./255)

  train = train_datagen.flow_from_directory(directory = './data/train',
                                                    target_size = (224,224),
                                                    batch_size = BS,
                                                    class_mode = "categorical",
                                                   )
  
  val = validation_datagen.flow_from_directory(directory = './data/val',
                                                    target_size = (224,224),
                                                    batch_size = BS,
                                                    class_mode = "categorical",
                                                   )
  
  test = test_datagen.flow_from_directory(directory = './data/test',
                                                    target_size = (224,224),
                                                    batch_size = BS,
                                                    class_mode = "categorical",
                                                   )
  return train, val, test

train, val, test = get_data(1)

'''
Leverage Tensorflow Keras API, use VGG19 pretrained on ImageNet. Modify the classification
layer of VGG19 to 3 output nodes. Retrain only the classification layer. 
'''
vgg = VGG19(include_top = False, input_shape = (224,224,3))
for layer in vgg.layers:
  layer.trainable = False

# Add a classification layer
x = vgg.output
x = Flatten()(x)
x = Dense(3, activation = "softmax")(x)
model = Model(inputs = vgg.input, outputs = x)
model.summary()

model.compile(optimizer = Adam(lr = 0.0001),
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

history = model.fit(train, epochs = 3, validation_data = val, callbacks = [ModelCheckpoint("vgg19_weights.h5")])

model.save("vgg19_model.h5")

#get classification matrix for test set
pred = model.predict(test)
pred = np.argmax(pred, axis = 1)

# iterate through test set and create a dataframe with the actual labels
# and the predicted labels

test_l = []
test_g = iter(test)
for i in range(len(test)):
  test_l.append(test_g.__next__()[1])

test_l = np.array(test_l)
test_l = np.argmax(test_l, axis = 1)

#get confusion matrix
cm = tf.math.confusion_matrix(test_l, pred).numpy()

print(cm)
print(classification_report(test_l, pred))

#evaluate on test set
test_loss, test_acc = model.evaluate(test)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

img1 = cv2.imread('data/train/buddha/image_0001.jpg')
#grayscale the imaege
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)/255.
#resize
img1 = cv2.resize(img1, (224, 224))

img2 = cv2.imread('data/train/chandelier/image_0002.jpg')
#grayscale the imaege
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)/255.
#resize
img2 = cv2.resize(img2, (224, 224))

img3 = cv2.imread('data/train/ketch/image_0003.jpg')
#grayscale the imaege
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)/255.
#resize
img3 = cv2.resize(img3, (224, 224))

'''
For each of the same 3 images find out 5 neurons in the last
convolutional layer that are maximally activated.
'''

