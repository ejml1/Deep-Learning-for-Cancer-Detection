import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(4099)

# Avoid Out of Memory Error by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Build data pipline
training_data = tf.keras.utils.image_dataset_from_directory('BM_cytomorphology_data', image_size=(250, 250))
val_data = tf.keras.utils.image_dataset_from_directory('validation', image_size=(250, 250))

#Scale data from 0-255 to 0-1
training_data = training_data.map(lambda image,label: (image/255, label))
val_data = val_data.map(lambda image,label: (image/255, label))

class MyTuner(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        #0
        model.add(Conv2D(8, (3, 3), 1, input_shape=(250, 250, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        #1
        model.add(Conv2D(8, (3, 3), 1, padding='same'))  # Adjusted the kernel size
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

        #2
        model.add(Conv2D(16, (3, 3), 1, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        #3
        model.add(Conv2D(16, (3, 3), 1, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

        #4
        model.add(Conv2D(32, (3, 3), 1, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        #5
        model.add(Conv2D(32, (3, 3), 1, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

        #6
        model.add(Conv2D(64, (3, 3), 1, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        #7
        model.add(Conv2D(64, (3, 3), 1, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))

        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(256))
        model.add(Activation('relu'))
        hp_dropout_rate = hp.Choice('dropout_rate', values = [0.2, 0.3, 0.4, 0.5])
        model.add(Dropout(hp_dropout_rate))
        model.add(Dense(21))
        model.add(Activation('softmax'))

        hp_learning_rate = hp.Choice('learning_rate', values = [0.001, 0.0001, 0.00001])
        optimizer = hp.Choice('optimizer', values = ['adam', 'sgd'])
        if optimizer == 'adam':
            optimizer = Adam(learning_rate = hp_learning_rate)
        else:
            if (hp_learning_rate == 0.00001):
                hp_learning_rate = 0.0001
            optimizer = SGD(learning_rate = hp_learning_rate)

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size=hp.Choice('batch_size', values=[32, 64, 128]), **kwargs)

callbacks = EarlyStopping(monitor='val_loss', patience=5)

tuner = kt.RandomSearch(MyTuner(), 
                        objective='val_accuracy', 
                        max_trials=60, 
                        executions_per_trial=1, 
                        directory='random_search', 
                        project_name='BM_Random_Search_copy')

tuner.search(training_data, validation_data=val_data, epochs=200, callbacks=[callbacks], verbose=2)

tuner.results_summary(num_trials=60)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps.values)

with open('best_hps.pkl', 'wb') as f:
    pickle.dump(best_hps, f)