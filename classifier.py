#!/usr/bin/env python3

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image as kimage

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
np.random.seed(123)

from os import listdir
import pandas as pd
from skimage import io
from skimage.transform import resize

num_classes = 2

def read_and_preprocess_image(file_path):
    img = kimage.load_img(file_path, target_size=(400,400))
    img_array = kimage.img_to_array(img)
    return img_array

def load_data_set(root_dir):
    data_set = []

    for subdir, label in [('positive/', 1), ('negative/', 0)]:
        file_dir = root_dir + subdir

        for file_name in listdir(file_dir):
            if 'image' not in file_name:
                continue
            file_path = file_dir + file_name
            logging.debug('loading image from ' + file_path)
            data_set.append((read_and_preprocess_image(file_path), label))

    # shuffle the positive & negative examples together
    np.random.shuffle(data_set)

    x = np.array([img for img, _ in data_set], dtype='float32')
    y = np.eye(num_classes, dtype='uint8')[[label for _, label in data_set]]
    return x, y

logging.info('loading training set')

x_train, y_train = load_data_set('data/training_set/')
logging.info('x_train shape: ' + str(x_train.shape))

#x_test, y_test = load_data_set('data/test_set/')

# define the model
logging.info('defining the model')

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', 
    input_shape=x_train.shape[1:], 
    data_format='channels_last'))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

# compile model
logging.info('compiling the model')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=40, batch_size=128, verbose=1)

model.save('model.h5')
