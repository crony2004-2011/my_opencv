import cv2
import numpy as np
import skimage
import PIL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import cv2

# path to input image is specified and
# image is loaded with imread command
image1 = cv2.imread('sat.jpg')
# cv2.cvtColor is applied over the
# image input with applied parameters
# to convert the image in grayscale
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  #MAKE IMAGE GRAY
# If pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black)  KEYY LINEEEEE
img_thresh = cv2.threshold(img,125,255,cv2.THRESH_BINARY)
image2 = img[0:340,0:340]  #CROPPING IMAGE
# applying different thresholding
# techniques on the input image

thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 199, 5)

thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 199, 5)

# the window showing output images
# with the corresponding thresholding
# techniques applied to the input image
cv2.imshow('cropped-image',image2)  #CROPPED
#cv2.imshow('Simple-Threshold',img_thresh)   #SIMPLE THRESHOLD
cv2.imshow('Original Image',img)  #ORIGINAL IMAGE
cv2.imshow('Adaptive Mean Thresh', thresh1) # ADAPTIVE THRESH
cv2.imshow('Adaptive Gaussian Thresh', thresh2) #MEAN THRESH


# De-allocate any associated memory usage
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

#OTSU FILTER THIS FILTER VIA THE HISTOGRAM DIAGRAM OF THE IMAGE SETS ITS THRESHOLD VALUE, AND IN ADVANCED OTSU WE GO THROUGH A 5*5 GAUSSIAN MATRIX. iTS VALUE IS
# cv.THRESH_OTSU. oTSU THRESHOLD IS
new_img,otsu_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#0 and 255 are min and max values, after this we create the otsu filter)
gausian_only = cv2.GaussianBlur(img,(5,5),0)
gauusian_otsu_thresh = cv2.threshold(gausian_only,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#plt.imshow(gausian_only,"Only Gausian Filter")
#plt.imshow(gauusian_otsu_thresh,'Gausian and Threshold')

# erosion, eroding away extra edges
#kernels = np.array((5,5),np.unit8)
#erode = cv2.erode(img, kernels, iteration=1)
#dilation = cv2.dilate(img,kernels,iterations = 1)


#CANNY, DISPLAYS ONLY EDGES, KIND OF PREPARING A SKETCH OF THE AREA.
edges = cv2.Canny(image1,100,200)
plt.imshow(edges, cmap = 'gray')
cv2.waitKey(0)
cv2.destroyAllWindows()

#CONTOURS
#imgray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,127,255,0)
#image, contours, hierarchy = cv2.findContours(thresh ,0 , 0)
#cv2.drawContours(edges, contours, -1, (0, 255, 0), 3)
#cv2.imshow(image)

#histograms

image = cv2.imread("sat_image.jpg",0)
color = ('b','g','r')

for i,col in enumerate(color):
    histr = cv2.calcHist([image], [i], None, [256], [0,256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])
    plt.show()
#CREATE A MASK
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(image,image,mask = mask)

#MASKED CALCHIST
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

#POST MASKED PLOTTING
plt.subplot(221), plt.imshow(image, 'gray')
plt.subplot(222), plt.imshow(masked_img, 'gray')
plt.subplot(223), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()