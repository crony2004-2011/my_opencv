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

image1 = cv2.imread('at.jpg')
red = image1[:,:,0]
rows, columns, channels = image1.shape
img_roi = image1[110:200,100:180]
lower_blue = np.array([111,110,110])
upper_blue = np.array([188,240,240])
hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
lower_blue = np.array([0,50,50])
upper_blue = np.array([130,255,255])
#USING MASKING ONLY HIGH INTENSITY AREAS CAN BE SPOTTED
mask = cv2.inRange(hsv,lower_blue,upper_blue)
res = cv2.bitwise_and(hsv,hsv, mask= mask)
#BITWISE I GUES REDUCES THE DATA SIZE IN TERMS OF BITS KEEPING THE MASK MEANWHILE
scaler = cv2.resize(image1,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

cv2.imshow('image', image1)
cv2.imshow('ROI',img_roi)
cv2.imshow("Mask + bitwise OP", res)
cv2.imshow("HSV", hsv)
cv2.waitKey(0)
print(rows)