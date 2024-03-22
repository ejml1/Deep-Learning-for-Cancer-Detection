import tensorflow as tf
import os
import time
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(4099)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

training_data = tf.keras.utils.image_dataset_from_directory('BM_cytomorphology_data_augmented', image_size=(250, 250))
val_data = tf.keras.utils.image_dataset_from_directory('Validation', image_size=(250, 250))

training_data = training_data.map(lambda image,label: (image/255, label))
val_data = val_data.map(lambda image,label: (image/255, label))

def build():
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
    model.add(Dropout(0.4))
    model.add(Dense(21))
    model.add(Activation('softmax'))

    hp_learning_rate = 0.001
    optimizer = Adam(learning_rate = hp_learning_rate)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

callbacks = EarlyStopping(monitor='val_loss', patience=15)

model = build()

t0 = time.time()
hist = model.fit(training_data, epochs = 500, validation_data = val_data, callbacks = [callbacks], verbose = 2, batch_size = 128)
t1 = time.time()-t0
print('Time taken: ', t1)

folder = 'pickle'
inner_folder = 'augmented'

with open(os.path.join(folder, inner_folder, 'model_pickle'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(folder, inner_folder, 'history_pickle'), 'wb') as f:
    pickle.dump(hist.history, f)