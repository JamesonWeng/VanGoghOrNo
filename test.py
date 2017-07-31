#!/usr/bin/env python3

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, Sequential
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
INPUT_WIDTH = 400
INPUT_HEIGHT = 400
NUM_CLASSES = 2
NUM_EPOCHS = 35

def test_cnn_model():
    # prepare the data generators
    logging.info('creating the data generators')

    test_datagen = kimage.ImageDataGenerator(
            # rescale=1./255, 
            data_format='channels_last',
            samplewise_center=True,
            samplewise_std_normalization=True,
            horizontal_flip=True,
            zoom_range=0.1)

    test_generator = test_datagen.flow_from_directory(
            'data/test/',
            target_size=(INPUT_WIDTH, INPUT_HEIGHT),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

    test_size = len(os.listdir('data/test/positive')) + len(os.listdir('data/test/negative/'))
    logging.info('test_size: ' + str(test_size))

    # load the model
    logging.info('loading the model')
    model = load_model('cnn_model.h5')

    scores = model.evaluate_generator(test_generator, test_size)

    for metric, score in zip(model.metrics_names, scores):
        print(metric, ':', score);

logging.info('testing the model')
test_cnn_model()
