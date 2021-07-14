import cv2
from wand.image import Image
import os
import easyocr
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
os.chdir(r'C:\Users\Acer\Desktop')

img = cv2.imread('Super Resolution3.png')
width, height = 380, 390
pts1 = np.float32([[665, 186], [795, 186],[788, 302] ,[666, 295]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (width,height))
cv2.circle(img, (665, 180), 5, (255, 0, 2), -1)
cv2.circle(img, (800, 190), 5, (255, 0, 2), -1)
cv2.circle(img, (788, 302), 5, (255, 0, 2), -1)
cv2.circle(img, (665, 295), 5, (255, 0, 2), -1)

plt.imshow(img)
#plt.imshow(result)
plt.show()
