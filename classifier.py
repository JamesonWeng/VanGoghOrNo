#!/usr/bin/env python3

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing import image as kimage
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

logging.basicConfig(level=logging.DEBUG)
np.random.seed(123)
Image.MAX_IMAGE_PIXELS=1000000000

BATCH_SIZE = 32
INPUT_WIDTH = 400
INPUT_HEIGHT = 400
NUM_CLASSES = 2
NUM_EPOCHS = 50

def train_cnn_model():
    # prepare the data generators
    logging.info('creating the data generators')

    # we split the data into training and validation, 70:30
    train_datagen = kimage.ImageDataGenerator(rescale=1./255, data_format='channels_last')
    train_generator = train_datagen.flow_from_directory(
            'data/training/',
            target_size=(INPUT_WIDTH, INPUT_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

    train_size = len(os.listdir('data/training/positive')) + len(os.listdir('data/training/negative/'))
    logging.info('training_size: ' + str(train_size))


    validation_datagen = kimage.ImageDataGenerator(rescale=1./255, data_format='channels_last')
    validation_generator = validation_datagen.flow_from_directory(
            'data/validation/',
            target_size=(INPUT_WIDTH, INPUT_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

    validation_size = len(os.listdir('data/validation/positive')) + len(os.listdir('data/validation/negative'))
    logging.info('validation_size: ' + str(validation_size))

    # define the model
    logging.info('defining the model')

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', 
        input_shape=(INPUT_WIDTH, INPUT_HEIGHT, 3),
        data_format='channels_last'))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # compile model
    logging.info('compiling the model')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the model
    logging.info('fitting the model')

    model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_size/BATCH_SIZE,
            epochs=NUM_EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_size/BATCH_SIZE)

    # save the final weigths
    logging.info('saving the model')
    model.save('cnn_model.h5')


train_cnn_model()
