#!/usr/bin/env python3

import logging
import pandas as pd
import pickle
from skimage import io

logging.basicConfig(level=logging.DEBUG)

# construct the data set
# - import data from CSV file
# - load images from specified URLs
# - record whether or not each image was painted by van Gogh
images_info = pd.read_csv('data/vgdb_2016.csv')

data_set = []

for index, row in images_info.iterrows():
    logging.info('Retrieving image from ' + row['ImageURL'])

    image = io.imread(row['ImageURL'])
    by_van_gogh = row['Artist'] == 'Vincent van Gogh'

    data_set.append((image, by_van_gogh))

    if(index == 10):
        break

max_bytes = 2**31 - 1
bytes_out = pickle.dumps(data_set)
with open('image_data.serialized', 'wb') as data_file:
    for idx in range(0, len(bytes_out), max_bytes):
        data_file.write(bytes_out[idx : idx + max_bytes])
