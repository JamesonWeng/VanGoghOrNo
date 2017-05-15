#!/usr/bin/env python3

import logging
import matplotlib.pyplot as plt
from os import listdir
import os.path
import pandas as pd
from skimage import io

def load_data_set(root_dir):
    data_set = []

    # read the positive examples
    positive_example_dir = root_dir + 'positive/'
    for file_name in os.listdir(positive_example_dir):
        file_path = positive_example_dir + file_name;
        logging.debug('loading image from ' + file_path)
        data_set.append((io.imread(file_path), True)) 

    # read the negative examples
    negative_example_dir = root_dir + 'negative/'
    for file_name in os.listdir(negative_example_dir):
        file_path = negative_example_dir + file_name;
        logging.debug('loading image from ' + file_path)
        data_set.append((io.imread(file_path), True)) 

    return data_set


logging.basicConfig(level=logging.DEBUG)

logging.info('loading training set')
training_set = load_data_set('data/training_set/')
