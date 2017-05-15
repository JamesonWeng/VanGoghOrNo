#!/usr/bin/env python3

import logging
import os
import pandas as pd
import shutil
import urllib.request
from urllib.parse import urlparse

def save_images(data_set, root_dir):
    positive_example_idx = 0
    negative_example_idx = 0

    for image_url, by_van_gogh in data_set:
        _, file_extension = os.path.splitext(urlparse(image_url).path)
        path_to_save = ''

        if by_van_gogh:
            path_to_save = ''.join([root_dir, 'positive/image_', str(positive_example_idx), file_extension])
            positive_example_idx += 1
        else:
            path_to_save = ''.join([root_dir, 'negative/image_', str(negative_example_idx), file_extension])
            negative_example_idx += 1

        logging.info('Saving file from URL: ' + image_url)
        logging.info('to file path: ' + path_to_save)

        with urllib.request.urlopen(image_url) as response, open(path_to_save, 'wb') as output_file:
            shutil.copyfileobj(response, output_file)

logging.basicConfig(level=logging.DEBUG)

# construct the data set
# - import data from CSV file
# - load images from specified URLs
# - record whether or not each image was painted by van Gogh
images_info = pd.read_csv('data/vgdb_2016.csv')
data_set = [(row['ImageURL'], row['Artist'] == 'Vincent van Gogh') for _, row in images_info.iterrows()]

# we split the data set into testing and training
split_idx = int(0.7 * len(data_set))
training_set = data_set[:split_idx]
test_set = data_set[split_idx:]

save_images(training_set, 'data/training_set/')
save_images(test_set, 'data/test_set/')
