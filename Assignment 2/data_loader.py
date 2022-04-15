from cv2 import split
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob

AUTO = tf.data.experimental.AUTOTUNE

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

def preprocess_img(img_path):
  print(img_path)
  img = tf.io.read_file(img_path)
  img = tf.image.decode_png(img, channels = 1)
  img = tf.image.resize(img, (28,28))
  img = img/255
  return img,img

def only_img():
  train = glob.glob('./data/train/*/*.jpg')
  val = glob.glob('./data/val/*/*.jpg')
  test = glob.glob('./data/test/*/*.jpg')

  train_ds = tf.data.Dataset.from_tensor_slices(train)
  val_ds = tf.data.Dataset.from_tensor_slices(val)
  test_ds = tf.data.Dataset.from_tensor_slices(test)

  train_ds = (
    train_ds
    .map(preprocess_img, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(64)
    .prefetch(AUTO)
    )
  val_ds = (
    val_ds
    .map(preprocess_img, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(64)
    .prefetch(AUTO)
    )
  
  test_ds = (
    test_ds
    .map(preprocess_img, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(64)
    .prefetch(AUTO)
    )


  return train_ds, val_ds, test_ds

def preprocess_img_n(img_path):
  img = tf.io.read_file(img_path)
  img = tf.image.decode_png(img, channels = 1)
  img = tf.image.resize(img, (28,28))
  img = img/255
  # add per pixel noise with a probability of 0.2
  imgn = tf.cond(tf.random.uniform(shape=[1])[0] < 0.2, lambda: tf.image.random_brightness(img, max_delta=0.2), lambda: img)
  return imgn, img

def only_img_n():
  train = glob.glob('./data/train/*/*.jpg')
  val = glob.glob('./data/val/*/*.jpg')
  test = glob.glob('./data/test/*/*.jpg')

  train_ds = tf.data.Dataset.from_tensor_slices(train)
  val_ds = tf.data.Dataset.from_tensor_slices(val)
  test_ds = tf.data.Dataset.from_tensor_slices(test)

  train_ds = (
    train_ds
    .map(preprocess_img_n, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(64)
    .prefetch(AUTO)
    )
  val_ds = (
    val_ds
    .map(preprocess_img_n, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(64)
    .prefetch(AUTO)
    )
  
  test_ds = (
    test_ds
    .map(preprocess_img_n, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(64)
    .prefetch(AUTO)
    )


  return train_ds, val_ds, test_ds