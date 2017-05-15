#!/usr/bin/env python3

import logging
import matplotlib.pyplot as plt
import os.path
import pandas as pd
import pickle
from skimage import io

def load_data_set(file_path):
    max_bytes = 2**31 - 1

    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    print(input_size)
    with open(file_path, 'rb') as data_file:
        for _ in range(0, input_size, max_bytes):
            bytes_in += data_file.read(max_bytes)

    return pickle.loads(bytes_in)


logging.basicConfig(level=logging.DEBUG)

logging.info('loading data set')
data_set = load_data_set('image_data.serialized')

plt.imshow(data_set[3][0])
plt.show()
