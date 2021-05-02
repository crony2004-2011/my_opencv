import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract as tess
import PIL
from PIL import Image
import wand
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img = cv2.imread('sat_image.jpg', 0)
#SHAPE
#430,730  h, w
#threshold
_, th1 = cv2.threshold(img,90,255,cv2.THRESH_TOZERO)
#ADAPTIVE. CALCULATING AND APPLYING THRESOLD FROM A SMALL PART OF IMAGE TO THE WHOLE IMAGE
adapt = cv2.adaptiveThreshold(th1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#OTSU THRESHOLDING MAKES HISTOGRAM OF IMAGE AND CALCULATES THRESHOLD VALUE BASED ON THAT
blur = cv2.GaussianBlur(adapt, (5, 5), 0)
ret3,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#PERSPECTIVE

pts1 = np.float32([[56,65],[368,52],[56,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
#ZOOOM IN
dst = cv2.warpPerspective(img,M,(300,300))
#median blur, GAUSSIAN BLUR, this was the only interesting one
median = cv2.medianBlur(dst,5)
gauss = cv2.GaussianBlur(dst,(5,5),0)
#cv2.imshow('Adaptive Threshold', th1)
#cv2.imshow('Adaptive Threshold', adapt)
#cv2.imshow('OTSU Threshold', otsu)
#cv2.imshow('Perspective', dst)
#cv2.imshow('Median-Blur', median)
#cv2.imshow('Gaussian Blur', gauss)
#cv2.waitKey(0)

#contours
contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
cont_img = cv2.drawContours(dst, contours, -1, (0,255,0), 3)
#cv2.imshow('contours', cont_img)
#cv2.waitKey(0)

#CORNERHARRIS
gray = np.float32(img)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
img[dst > 0.01*dst.max()] = [0, 0, 255]
#cv2.imshow('CornerHarris', img)
#cv2.waitKey(0)

#SHI-THOMAS
shi_corners = cv2.goodFeaturesToTrack(img,10,.01,3)
shi_corners = np.int0(shi_corners)

for i in shi_corners:
    x, y = i.ravel()
    cv2.circle(img,(x,y), 3, 255, -1)

cv2.imshow('shi-thomas', img)
cv2.waitKey(0)
