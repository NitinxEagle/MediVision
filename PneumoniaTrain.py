import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Set the image directory
image_directory = 'Datasets/'

# List the images in NORMAL and PNEUMONIA directories
normal_images = os.listdir(image_directory+ 'Pneumonia_no/')
pneumonia_images = os.listdir(image_directory+ 'Pneumonia_yes/')
dataset = []
label = []

INPUT_SIZE=64

# Load NORMAL images and assign label 0
for i , image_name in enumerate(normal_images):
    if(image_name.split('.')[1]=='jpeg'):
        image=cv2.imread(image_directory+'pneumonia_no/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)


# Load PNEUMONIA images and assign label 1
for i , image_name in enumerate(pneumonia_images):
    if(image_name.split('.')[1]=='jpeg'):
        image=cv2.imread(image_directory+'pneumonia_yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)


# Convert lists to numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize the pixel values to be between 0 and 1
x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)


# Build a new CNN model
model = Sequential()
    
model.add(Conv2D(32, (3,3),  input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3,3),  kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3),  kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile the new model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the new model
model.fit(x_train, y_train, 
batch_size=16, 
verbose=1, epochs=10, # type: ignore
validation_data=(x_test, y_test),
shuffle=False)


# Save the new model
model.save('PneumoniaDetectionModel.h5')

