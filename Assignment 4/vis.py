import os
import numpy as np
import pandas as pd
from data_loader import load_hw
import time
import matplotlib.pyplot as plt

X_train, y_train, X_val, y_val, X_test, y_test = load_hw()


np.argmax(y_train[:10], axis = 2)
plt.plot(X_train[8][0, :, 0], X_train[8][0, :, 1], 'xb-')
plt.title('a')
plt.show()