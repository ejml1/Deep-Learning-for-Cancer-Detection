import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import kerastuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import uniform


# Avoid Out of Memory Error by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

gpus

# Build data pipline
training_data = tf.keras.utils.image_dataset_from_directory('Data_Subset', image_size=(250, 250))
val_data = tf.keras.utils.image_dataset_from_directory('Data_Subset_Validation', image_size=(250, 250))


#Scale data from 0-255 to 0-1
training_data = training_data.map(lambda image,label: (image/255, label))
val_data = val_data.map(lambda image,label: (image/255, label))

class MyTuner(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        # 16 filters, 3x3 size, stride of 1, 
        model.add(Conv2D(16, (3,3), 1, activation = 'relu', input_shape = (250,250,3)))
        model.add(MaxPooling2D())

        model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
        model.add(MaxPooling2D())

        model.add(Conv2D(16, (3,3), 1, activation = 'relu'))
        model.add(MaxPooling2D())

        model.add(Flatten())

        hp_learning_rate = hp.Choice('learning_rate', values = [0.001, 0.0001])
        optimizer = hp.Choice('optimizer', values = ['adam', 'sgd'])
        if optimizer == 'adam':
            optimizer = Adam(learning_rate = hp_learning_rate)
        else:
            optimizer = SGD(learning_rate = hp_learning_rate)

        hp_dropout_rate = hp.Choice('dropout_rate', values = [0.2, 0.3, 0.4, 0.5])
        model.add(Dropout(hp_dropout_rate))

        model.add(Dense(250, activation = 'relu'))
        model.add(Dense(21, activation='softmax')) 

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size=hp.Choice('batch_size', values=[32, 64, 128]), **kwargs)

# Custom Oracle to allow pickle
class MyOracle(kt.Oracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_hyperparameters_space')
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

callbacks = EarlyStopping(monitor='val_loss', patience=3)

tuner = kt.RandomSearch(MyTuner(), objective='val_accuracy', max_trials=2, executions_per_trial=1, directory='random_search', project_name='second_random_search')
tuner.search(training_data, validation_data=val_data, epochs=2, callbacks=[callbacks])

tuner.results_summary(num_trials=2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps)

with open('best_hps.pkl', 'wb') as f:
    pickle.dump(best_hps, f)