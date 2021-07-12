import glob

from skimage.transform import resize
img_px_size = 256

import numpy as np
import cv2
import pydicom as dicom
from skimage import exposure
import random

#path = '../content/siim/dicom-images-train/*/*/*.dcm'

def downscale_images(dataset):
    dicom_data = dataset.pixel_array
    dicom_data = resize(dicom_data, (img_px_size, img_px_size), anti_aliasing=True)
    # print(dicom_data.shape)


import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage, misc
import pandas as pd
import numpy as np

data = pd.read_csv('train-rle.csv', delimiter=',', index_col='ImageId')



w, h = 256, 256
id = 0
images, masks = [],[]

def augment_image(imageID, blur: int, angle:int, brightness:float, id, source, destination):
    data = np.load(source+'/'+imageID+'.npy')+brightness # between 0.0-0.1
    blurred = gaussian_filter(data, sigma=blur)
    blurred_30 = ndimage.rotate(blurred, angle, reshape=False)
    plt.imshow(blurred_30, cmap=plt.cm.bone)
    plt.savefig(destination+'/'+str(id)+'.png')


def generate_image(imageID):
    global id
    random_blur = random.uniform(0, 2)
    random_angle = random.uniform(-15, 15)
    random_brightness = random.uniform(-0.9, 0.9)
    print(random_blur, random_angle, random_brightness)
    augment_image(imageID, random_blur, random_angle, random_brightness, id,"Extracted-images-resized", "aug_images")
    augment_image(imageID, random_blur, random_angle, random_brightness, id,"extracted-images-resized-masks", "aug_masks")
    id += 1

def generate_image_and_masks(img_mask):
    for key in list(img_mask.keys())[:5]:
        generate_image(key)



#generate_image_and_masks(glob.glob('Extracted-images-resized/*.npy'),glob.glob('extracted-images-resized-masks/*.npy'))

image_masks = dict()

for q, image in enumerate(glob.glob('Extracted-images-resized/*.npy')):
    images.append(image.split('\\')[-1][:-4])

for q, image in enumerate(glob.glob('extracted-images-resized-masks/*.npy')):
    masks.append(image.split('\\')[-1][:-4])

for i in images:
    if i in masks:
        image_masks[i] = masks[masks.index(i)]

print(len(image_masks.keys()))
#generate_image_and_masks(image_masks)





