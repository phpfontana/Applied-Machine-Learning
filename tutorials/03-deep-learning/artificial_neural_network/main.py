# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical

# verifying version
print(f'tensorflow version: {tf.__version__}\n')

# Device configuration
print(f"Verifying GPU Access:\n")
device = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU IS", "AVAILABLE\n" if device else "NOT AVAILABLE\n")

# Hyper-parameters
num_epochs = 20
batch_size = 512
learning_rate = 0.001

# Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# visualizing images
plt.figure(figsize=(10, 5))

for i in range(5):

    plt.subplot(1, 5, i+1)

    plt.imshow(X_train[i], cmap='gray')
    plt.title(f'{class_names[y_train[i]]}')
    plt.axis(False)

plt.show()

# Reshaping the dataset
X_train = X_train.reshape(X_train.shape[0], 784)

X_test = X_test.reshape(X_test.shape[0], 784)

# normalizing data
X_train = X_train/255.
X_test = X_test/255.

# One-hot encoding y_true
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Clearing backend
backend.clear_session()

# setting random seed
tf.random.set_seed(42)

# Defining model
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))

# Output layer
model.add(Dense(len(class_names), activation="softmax"))

# Compiling model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(learning_rate=learning_rate),
              metrics='accuracy')

# Model summary
model.build()
model.summary()

# Training
history_01 = model.fit(X_train, y_train,
                       validation_split=0.2,
                       batch_size=batch_size,
                       verbose=1,
                       epochs=num_epochs)

# Plotting accuracy
dict_hist = history_01.history

list_ep = [i for i in range(1, 21)]

plt.figure(figsize=(8, 8))

plt.plot(list_ep, dict_hist['accuracy'], ls='--', label='accuracy')
plt.plot(list_ep, dict_hist['val_accuracy'], ls='--', label='val_accuracy')

plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.show()