# Extract image masks from the CSV files

import pandas as pd
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
import os
import glob
import pydicom
from PIL import Image
import cv2
from skimage.transform import resize

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

# The main script 
if __name__ == "__main__":
    # Load the CSV file
    data_test = pd.read_csv('path_to_csv_files', delimiter=',')
    # Extract the UIDs and the ground truth labels
    uid = data_test['ImageId'].to_numpy()
    pixels = data_test[' EncodedPixels'].to_numpy()
    uid_unique, uid_counts = np.unique(uid, return_counts=True)

    # Set the new resolution
    newResolution = 256
    # Path to the image masks
    file_path = 'path_to_masks'

    # Iterate through the UIDs
    for i in range(len(uid)):
        if pixels[i] != '-1':
            mask = rle2mask(pixels[i], 1024, 1024).T
            resMask = resMask = resize(mask, (newResolution, newResolution), anti_aliasing = True)
            if(os.path.exists(file_path + uid[i] + '.npy')):
                old_mask = np.load(file_path + uid[i] + '.npy')
                new_mask = old_mask + resMask
                np.save('%s' %uid[i], new_mask)
            else:
                np.save('%s' %uid[i], resMask)
        else:
            continue