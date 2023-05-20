# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import ZeroPadding2D, Dense, Conv2D, MaxPool2D, Flatten, LeakyReLU
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical

print(f"TensorFlow version: {tf.__version__}\n")

# Device configuration
print(f"Verifying GPU Access:\n")
device = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU IS", "AVAILABLE\n" if device else "NOT AVAILABLE\n")

# Hyper-parameters
num_epochs = 20
num_classes = 10
batch_size = 512
learning_rate = 0.001

# MNIST dataset
MNIST = mnist.load_data()

# Train test split
(X_train, y_train), (X_test, y_test) = MNIST

# Reshaping
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Loading image
image = X_train[0]
plt.imshow(image)
plt.show()

# Normalizing inputs
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encoding y_true
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Clearing backend
backend.clear_session()

# Setting random seed
tf.random.set_seed(42)

# Defining model
model = Sequential()

# Input layer
model.add(tf.keras.Input(shape=X_train[0].shape))

# Zero padding
model.add(ZeroPadding2D(padding=(2, 2)))

# Conv 1
model.add(Conv2D(filters=6, kernel_size=5, padding="valid", strides=1))
model.add(LeakyReLU(0.1))
model.add(MaxPool2D(pool_size=2, strides=2))

# Conv2
model.add(Conv2D(filters=16, kernel_size=5, padding="valid", strides=1))
model.add(LeakyReLU(0.1))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Flatten())

# FC 3
model.add(Dense(120))
model.add(LeakyReLU(0.1))

# FC 4
model.add(Dense(84))
model.add(LeakyReLU(0.1))

# Output layer
model.add(Dense(num_classes, activation="softmax"))

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