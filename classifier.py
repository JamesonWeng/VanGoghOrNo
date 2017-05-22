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

def preprocess_and_generate_samples(file_path, label):
    img = Image.open(file_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    hw_tuple = (INPUT_HEIGHT, INPUT_WIDTH)
    if img.size != hw_tuple:
        img = img.resize(hw_tuple)

    img_array = kimage.img_to_array(img, 'channels_last')

    #img.show()
    #plt.imshow(img_array)
    #plt.show()

    return [(img_array, label)]

def load_data_set(root_dir):
    data_set = []

    for subdir, label in [('positive/', 1), ('negative/', 0)]:
        file_dir = root_dir + subdir

        for file_name in os.listdir(file_dir):
            if 'image' not in file_name:
                continue
            file_path = file_dir + file_name
            logging.debug('loading image from ' + file_path)
            data_set.extend(preprocess_and_generate_samples(file_path, label))
            # data_set.append((read_and_preprocess_image(file_path), label))
            # break

    # shuffle the positive & negative examples together
    np.random.shuffle(data_set)

    x = np.array([img for img, _ in data_set], dtype='float32')
    y = np.eye(NUM_CLASSES, dtype='uint8')[[label for _, label in data_set]]

    return x, y

def train_cnn_model():
    # prepare the data generators
    logging.info('creating the data generators')

    x_train, y_train = load_data_set('data/training/')
    # x_validation, y_validation = load_data_set('data/validation/')

    train_datagen = kimage.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True)
    train_datagen.fit(x_train)

    '''
    validation_datagen = kimage.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True)
    validation_datagen.fit(x_train)
    '''

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
    sgd = SGD(lr=0.01) # 0.0001, 0.00001
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the model
    logging.info('fitting the model')

    model.fit_generator(
            generator=train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=len(x_train) / BATCH_SIZE,
            epochs=NUM_EPOCHS)

    # save the final weigths
    logging.info('saving the model')
    model.save('cnn_model.h5')


logging.info('loading training set')

train_cnn_model()
