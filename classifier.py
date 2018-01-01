#!/usr/bin/env python3

from keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing import image as kimage
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

logging.basicConfig(level=logging.DEBUG)
np.random.seed(123)
Image.MAX_IMAGE_PIXELS=1000000000

BATCH_SIZE = 64
DATA_FORMAT = 'channels_last'
INPUT_WIDTH = 400
INPUT_HEIGHT = 400
NUM_CLASSES = 2
NUM_EPOCHS = 35

def count_images(directory):
    num_img = 0
    for root, dirs, files in os.walk(directory):
        files = [f for f in files if f[0] != '.']
        num_img += len(files)
    return num_img

def calculate_mean_image(directory):
    num_img = count_images(directory)
    avg_img = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), np.float)
    for root, dirs, files in os.walk(directory): 
        for f in files: # we assume all unhidden files are images
            if f[0] == '.': 
                continue
            img = Image.open(os.path.join(root, f)).resize((INPUT_WIDTH, INPUT_HEIGHT))
            img_arr = np.array(img, dtype=np.float)
            avg_img += img_arr / num_img
    avg_img = np.array(np.round(avg_img), dtype=np.uint8)
    return avg_img

MEAN_IMAGE = calculate_mean_image('data/training/')

def get_train_data(directory):
    train_datagen = kimage.ImageDataGenerator(
            featurewise_center = True,
            data_format=DATA_FORMAT,
            horizontal_flip=True,
            zoom_range=0.1)

    train_datagen.mean = MEAN_IMAGE

    train_generator = train_datagen.flow_from_directory(
            directory,
            target_size=(INPUT_WIDTH, INPUT_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

    train_size = count_images(directory)
    logging.info('training_size: ' + str(train_size))
    return (train_generator, train_size)

def get_test_data(directory):
    test_datagen = kimage.ImageDataGenerator(
            featurewise_center = True,
            data_format=DATA_FORMAT)

    test_datagen.mean = MEAN_IMAGE

    test_generator = test_datagen.flow_from_directory(
            directory,
            target_size=(INPUT_WIDTH, INPUT_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

    test_size = count_images(directory)
    logging.info('test_size: ' + str(test_size))
    return (test_generator, test_size)

def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', 
        input_shape=(INPUT_WIDTH, INPUT_HEIGHT, 3),
        data_format=DATA_FORMAT))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D(data_format=DATA_FORMAT))

    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # compile model
    logging.info('compiling the model')
    adam = Adam(lr=3e-6, decay=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def train_cnn_model():
    # prepare the data generators
    logging.info('creating the data generators')

    train_generator, train_size = get_train_data('data/training/')
    logging.info('train_size: ' + str(train_size))
    # validation_generator, validation_size = get_test_data('data/validation/')

    # create the model
    logging.info('creating the model')
    model = create_model()

    # fit the model
    logging.info('fitting the model')

    model.fit_generator(
            generator=train_generator,
            steps_per_epoch=1, #train_size/BATCH_SIZE,
            epochs=NUM_EPOCHS) #,
            #validation_data=validation_generator,
            #validation_steps=validation_size/BATCH_SIZE)

    # save the final weights
    logging.info('saving the model')
    model.save('cnn_model.h5')

train_cnn_model()
