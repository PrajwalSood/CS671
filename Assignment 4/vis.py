import os
import numpy as np
import pandas as pd
from data_loader import load_hw
import time
import matplotlib.pyplot as plt

X_train, y_train, X_val, y_val, X_test, y_test = load_hw()


for idx,i in enumerate(np.argmax(y_train[:30], axis = 2)):
  if i == 4:
    print(idx)

pt = X_train[28]
X = pt[0, :, 0] - np.min(pt[0, :, 0])/ (np.max(pt[0, :, 0]) - np.min(pt[0, :, 0]))

Y = pt[0, :, 1] - np.min(pt[0, :, 1])/ (np.max(pt[0, :, 1]) - np.min(pt[0, :, 1]))

plt.plot(X,Y, 'xb-')
plt.title('a')
plt.show()