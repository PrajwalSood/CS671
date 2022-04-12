from data_loader import get_data
import tensorflow as tf

data,_,_ = get_data(64)

x,y=next(iter(data))
print(x[0].shape)
print(y[0].shape)