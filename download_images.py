#!/usr/bin/env python3

import csv
import io
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
from PIL import Image
import requests

logging.basicConfig(level=logging.DEBUG)
Image.MAX_IMAGE_PIXELS = int(3e9)

download_dir = 'data/images'

positive_dir = os.path.join(download_dir, 'positive')
if not os.path.exists(positive_dir):
    os.makedirs(positive_dir)
    
negative_dir = os.path.join(download_dir, 'negative')
if not os.path.exists(negative_dir):
    os.makedirs(negative_dir)

images_info = pd.read_csv('data/vgdb_2016.csv')

artists = np.unique([row['Artist'] for _, row in images_info.iterrows()])
logging.debug('artists: {}'.format(artists))


def download_row(row_num, row):
    try:
        logging.debug('processing row #: {}'.format(row_num))
        
        # for now, ignore controversial & disputed paintings
        artist_name = row['Artist'].lower()
        
        if 'vincent van gogh' in artist_name:
            if 'controversial' in artist_name or 'disputed' in artist_name:
                logging.info('skipping: controversial van gogh painting')
                return
            
            target_dir = positive_dir
        else:
            target_dir = negative_dir
        
        # download image
        image_url = row['ImageURL']
        logging.debug('saving image from URL: {}'.format(image_url))

        response = requests.get(image_url)
        img = Image.open(io.BytesIO(response.content))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.save(os.path.join(target_dir, 'image_{}.jpg'.format(row_num)))
    except Exception:
        logging.exception('Encountered exception on row #: {}'.format(row_num))


pool = multiprocessing.Pool(8)
pool.starmap(download_row, images_info.iterrows())
pool.close()


# rows = list(images_info.iterrows())
# for i, row in rows:
#     if i % 10 == 0:
#         logging.debug('done {} / {}'.format(i, len(rows)))
    
#     download_row(i, row)

