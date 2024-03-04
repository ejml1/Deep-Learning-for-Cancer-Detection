import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


with open('best_hps.pkl', 'rb') as f:
    best = pickle.load(f)


print(best.values)