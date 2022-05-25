import os
import numpy as np
import pandas as pd
from data_loader import load_hw
import time
import matplotlib.pyplot as plt

X_train, y_train, X_val, y_val, X_test, y_test = load_hw()


for i in(np.argmax(y_train[:25], axis = 2):
  if i == 0:
    print(i)

plt.plot(X_train[8][0, :, 0], X_train[8][0, :, 1], 'xb-')
plt.title('a')
plt.show()