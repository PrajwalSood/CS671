from cv2 import split
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data(BS):
  train_datagen = ImageDataGenerator(rescale = 1./255)
  validation_datagen = ImageDataGenerator(rescale = 1./255)
  test_datagen = ImageDataGenerator(rescale = 1./255)

  train = train_datagen.flow_from_directory(directory = './data/train',
                                                    target_size = (28,28),
                                                    batch_size = 64,
                                                    color_mode = "grayscale",
                                                    class_mode = "categorical",
                                                   )
  
  val = validation_datagen.flow_from_directory(directory = './data/val',
                                                    target_size = (28,28),
                                                    batch_size = 64,
                                                    color_mode = "grayscale",
                                                    class_mode = "categorical",
                                                   )
  
  test = test_datagen.flow_from_directory(directory = './data/test',
                                                    target_size = (28,28),
                                                    batch_size = 64,
                                                    color_mode = "grayscale",
                                                    class_mode = "categorical",
                                                   )
  return train, val, test

def get_data_all():
  train_datagen = ImageDataGenerator(rescale = 1./255)
  validation_datagen = ImageDataGenerator(rescale = 1./255)
  test_datagen = ImageDataGenerator(rescale = 1./255)

  train = train_datagen.flow_from_directory(directory = './data/train',
                                                    target_size = (28,28),
                                                    batch_size = 11385,
                                                    color_mode = "grayscale",
                                                    class_mode = "categorical",
                                                   )
  
  val = validation_datagen.flow_from_directory(directory = './data/val',
                                                    target_size = (28,28),
                                                    batch_size = 3795,
                                                    color_mode = "grayscale",
                                                    class_mode = "categorical",
                                                   )
  
  test = test_datagen.flow_from_directory(directory = './data/test',
                                                    target_size = (28,28),
                                                    batch_size = 3795,
                                                    color_mode = "grayscale",
                                                    class_mode = "categorical",
                                                   )
  return train, val, test
