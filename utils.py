from skimage.transform import resize
img_px_size = 256

import numpy as np
import cv2
import pydicom as dicom
from skimage import exposure

#path = '../content/siim/dicom-images-train/*/*/*.dcm'

def downscale_images(dataset):
    dicom_data = dataset.pixel_array
    dicom_data = resize(dicom_data, (img_px_size, img_px_size), anti_aliasing=True)
    # print(dicom_data.shape)


import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage, misc
from PIL import Image



w, h = 256, 256

def augment_image(image_path, blur: int, angle:int, brightness:float):
    data = np.load(image_path)+brightness # between 0.0-0.1

    blurred = gaussian_filter(data, sigma=blur)
    blurred_30 = ndimage.rotate(blurred, angle, reshape=False)
    plt.imshow(blurred_30, cmap=plt.cm.bone)
    plt.show()


augment_image('SSIP-test-images-resized/1.2.276.0.7230010.3.1.2.8323329.14592.1517875252.883262.npy', 2, 15, 0.01)





