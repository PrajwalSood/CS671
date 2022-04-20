import numpy as np
import tensorflow as tf
from data_loader import get_data
from sklearn.metrics import classification_report

best_clf = 'models/fcnn_128_64_32_RMSprop.h5'

train, val, test = get_data(1)
# y = np.concatenate([y for x, y in test], axis=0)

model = tf.keras.models.load_model(best_clf)

tp = model.evaluate(test)


best_ae = 'ae/ae1_dn_[64].h5'

model = tf.keras.models.load_model(best_ae)

tp = model.evaluate(test)