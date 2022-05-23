import numpy as np
import matplotlib.pyplot as plt
import cv2
from models import Network, ConvLayer, PoolingLayer, FCLayer, ActivationLayer
from models import tanh, tanh_prime, mse, mse_prime, sigmoid, sigmoid_prime

def display_image(image):
  cv2.imshow('image', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

'''
Pick any one image from each of the three classes and convert it into a single channel grayscale
image. Initialize a 3x3 convolutional filter using Kaiming initialization. Traverse the
convolution filter over all the pixels of the image with stride 1, padding 0 and obtain the final
output as a feature map. Calculate the expected dimension of the feature map using the
mathematical formula and verify it is the same as obtained by you. Report should include these
observations, image considered, filter values and the feature maps obtained etc.
'''
img1 = cv2.imread('data/train/buddha/image_0001.jpg')
#grayscale the imaege
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)/255.
display_image(img1)


'''Initialize a 3x3 convolutional filter using Kaiming initialization.'''
filter1 = np.random.randn(3, 3, 1, 1)
filter1 = filter1 * np.sqrt(2.0 / (3 * 3))

'''Traverse the
convolution filter over all the pixels of the image with stride 1, padding 0 and obtain the final
output as a feature map.'''
feature_map1 = np.zeros((img1.shape[0] - 2, img1.shape[1] - 2, 1))
for i in range(img1.shape[0] - 2):
  for j in range(img1.shape[1] - 2):
    feature_map1[i, j] = np.sum(img1[i:i + 3, j:j + 3] * filter1)

display_image(feature_map1)

#repeat for the other images
img2 = cv2.imread('data/train/chandelier/image_0004.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)/255.
display_image(img2)
filter2 = np.random.randn(3, 3, 1, 1)/10
filter2 = filter2 * np.sqrt(2.0 / (3 * 3))
feature_map2 = np.zeros((img2.shape[0] - 2, img2.shape[1] - 2, 1))
for i in range(img2.shape[0] - 2):
  for j in range(img2.shape[1] - 2):
    feature_map2[i, j] = np.sum(img2[i:i + 3, j:j + 3] * filter2)
display_image(feature_map2)

img3 = cv2.imread('data/train/ketch/image_0002.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)/255.
display_image(img3)
filter3 = np.random.randn(3, 3, 1, 1)/10
filter3 = filter3 * np.sqrt(2.0 / (3 * 3))
feature_map3 = np.zeros((img3.shape[0] - 2, img3.shape[1] - 2, 1))
for i in range(img3.shape[0] - 2):
  for j in range(img3.shape[1] - 2):
    feature_map3[i, j] = np.sum(img3[i:i + 3, j:j + 3] * filter3)
display_image(feature_map3)


