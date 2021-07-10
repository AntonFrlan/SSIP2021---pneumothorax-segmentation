from skimage.transform import resize
img_px_size = 256

#path = '../content/siim/dicom-images-train/*/*/*.dcm'

def downscale_images(dataset):
    dicom_data = dataset.pixel_array
    dicom_data = resize(dicom_data, (img_px_size, img_px_size), anti_aliasing=True)
    # print(dicom_data.shape)
