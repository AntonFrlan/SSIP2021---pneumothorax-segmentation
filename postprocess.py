import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
images = []
for name in glob.glob('png/*.png'):
    images.append(name)

w = 256
h = 256
fig = plt.figure(figsize=(8, 8))
columns = 3
rows = 3

for i in range(len(images)):
    im = cv2.imread(images[i])
    mask = np.zeros(im.shape[:2], dtype="uint8")
    im_orig = im.copy()
    im_gray = 255 - cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im_gray, 160, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    #print('ContourNum:', len(contours))
    for cntrIdx in range(0, len(contours)):
        # dismiss contours with parents
        if hierarchy[0][cntrIdx][3] != -1:
            continue
        if cv2.contourArea(contours[cntrIdx]) > 50:
            cv2.drawContours(mask, contours, cntrIdx, (255, 255, 255), -1)

            # cv2.imshow('Contours', mask)

    #plt.imshow(mask, cmap=plt.cm.bone)
    #print(type(mask))
    np.save(images[i].split('\\')[-1][:-4] + '.npy', mask)
    



#plt.show()



key = cv2.waitKey(0)
cv2.destroyAllWindows()
