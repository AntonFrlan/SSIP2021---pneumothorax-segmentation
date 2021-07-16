# Extract 2D numpy arrays containing the image data from DICOM images and save them to a local file
# Name the new files using the UID from the dataset
import numpy as np
import pydicom
import os
import glob
from skimage.transform import resize

# Define the function to extract the data
# Input: source - path to the source directory
#        newResolution - new resolution (x and y) of the resized images
def extractDataMatricesFromDICOM(source, destination, newResolution):
  # Get the paths to the data
  source = os.path.join(os.getcwd(), "siim", "*", "*", "*", "*.dcm")
  images_path = glob.glob(source)
  # Iterate through all images
  for img_path in images_path:
    # Read the DICOM file
    img = pydicom.dcmread(img_path)
    # Extract the 2D numpy array
    tempImg = img.pixel_array
    # Resize the image
    resTempImg = resize(tempImg, (newResolution, newResolution), anti_aliasing = True)
    # Extract the UID from the metadata
    uid = img[0x0008, 0x0018].value
    # Save the numpy file to the destination directory
    np.save(destination + '%s' %uid, resTempImg)
 
extractDataMatricesFromDICOM('path_to_source_directory', 'path_to_destination_directory', newResolution)