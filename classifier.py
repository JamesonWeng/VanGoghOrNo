#!/usr/bin/env python3

import logging
import matplotlib.pyplot as plt
from os import listpath
import os.path
import pandas as pd
import pickle
from skimage import io

def load_data_set(root_dir):
    data_set = []


logging.basicConfig(level=logging.DEBUG)

logging.info('loading training set')
training_set = load_data_set('data/training_set')

plt.imshow(training_set[3][0])
plt.show()
