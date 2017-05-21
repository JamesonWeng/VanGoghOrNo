#!/usr/bin/env python3

import io
import logging
import os
import pandas as pd
from PIL import Image
import requests
import shutil
import sys

INPUT_HEIGHT = 400
INPUT_WIDTH = 400

# crop out INPUT_WIDTH x INPUT_HEIGHT sized images from the dataset
# and save them to the specified directory
def crop_and_save_dataset(data_set, target_dir):
    image_idx = 0

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for image_url in data_set:
        logging.info('Saving image(s) from URL: ' + image_url)

        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        # crop out regions of size (INPUT_WIDTH, INPUT_HEIGHT)
        # from the image
        width, height = img.size

        for topleft_x in range(0, width // 2 + 1, width // 2):
            for topleft_y in range(0, height // 2 + 1, height // 2):
                img_portion = img.crop((topleft_x, topleft_y, topleft_x + width // 2, topleft_y + height // 2))
                
                path = ''.join([target_dir, 'image_', str(image_idx), '.bmp'])
                image_idx += 1

                logging.info('Saving image to path: ' + path)
                img_portion.save(path)

        '''
        for topleft_x in range(0, width - INPUT_WIDTH, INPUT_WIDTH):
            for topleft_y in range(0, height - INPUT_HEIGHT, INPUT_HEIGHT):
                img_portion = img.crop((topleft_x, topleft_y, topleft_x + INPUT_WIDTH, topleft_y + INPUT_HEIGHT))
                
                path = ''.join([target_dir, 'image_', str(image_idx), '.bmp'])
                image_idx += 1

                logging.info('Saving image to path: ' + path)
                img_portion.save(path)
        '''

# returns training_set, validation_set, test_set 
# grouped according to the specified ratios
def split_data_set(data_set, training_percentage, validation_percentage, test_percentage):
    total_percentage = training_percentage + validation_percentage + test_percentage

    length = len(data_set)
    split_idx1 = int(length * (float(training_percentage) / total_percentage))
    split_idx2 = int(length * (float(training_percentage + validation_percentage) / total_percentage))

    return data_set[:split_idx1], data_set[split_idx1:split_idx2], data_set[split_idx2:]


Image.MAX_IMAGE_PIXELS=1000000000
logging.basicConfig(level=logging.INFO)

# construct and save the data set
# - import data from CSV file
# - load images from specified URLs
# - save them to the appropriate subdirectory

images_info = pd.read_csv('data/vgdb_2016.csv')

positive_samples = []
negative_samples = []

for _, row in images_info.iterrows():
    if row['Artist'] == 'Vincent van Gogh':
        positive_samples.append(row['ImageURL'])
    else:
        negative_samples.append(row['ImageURL'])

# we split the data set into training, validation, and testing
positive_training_set, positive_validation_set, positive_test_set = split_data_set(positive_samples, 60, 20, 20)
negative_training_set, negative_validation_set, negative_test_set = split_data_set(negative_samples, 60, 20, 20)

crop_and_save_dataset(positive_training_set, 'data/training/positive/')
crop_and_save_dataset(negative_training_set, 'data/training/negative/')

crop_and_save_dataset(positive_validation_set, 'data/validation/positive/')
crop_and_save_dataset(negative_validation_set, 'data/validation/negative/')

crop_and_save_dataset(positive_test_set, 'data/test/positive/')
crop_and_save_dataset(negative_test_set, 'data/test/negative/')
