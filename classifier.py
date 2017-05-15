#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.DEBUG)

import matplotlib.pyplot as plt

import numpy as np
np.random.seed(123)

from os import listdir
import pandas as pd
from skimage import io
from skimage.transform import resize

def load_data_set(root_dir):
    data_set = []

    # read the positive examples
    positive_example_dir = root_dir + 'positive/'
    for file_name in listdir(positive_example_dir):
        file_path = positive_example_dir + file_name;
        logging.debug('loading image from ' + file_path)
        data_set.append((io.imread(file_path), True)) 
        break

    # read the negative examples
    negative_example_dir = root_dir + 'negative/'
    for file_name in listdir(negative_example_dir):
        file_path = negative_example_dir + file_name;
        logging.debug('loading image from ' + file_path)
        data_set.append((io.imread(file_path), True)) 
        break

    return data_set

logging.info('loading training set')
training_set = load_data_set('data/training_set/')

# resize all the images
logging.info('resizing the images')
for image, _ in training_set:
    image = resize(image, (400, 400), mode='reflect')

# define the model
