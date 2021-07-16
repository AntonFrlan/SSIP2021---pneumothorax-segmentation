# Load the image in DICOM format
import matplotlib.pyplot as plt
import pydicom

# Path to the image in DICOM format
filename = 'path_to_dicom_image'
# Load the image
img = pydicom.dcmread(filename)
# Show the image
plt.imshow(img.pixel_array, cmap=plt.cm.bone)
plt.colorbar()
plt.show()
# Print the information about the image
print('Metadata: \n', img)
print('Image resolution:', img.pixel_array.shape)
print('Image UID:', img[0x0008, 0x0018].value)