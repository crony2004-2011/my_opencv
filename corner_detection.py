import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
os.chdir(r'C:\Users\hp\Desktop')
i = cv2.imread('mask_person2.jpg',0)
#grayscale the image
g = np.float32(i)
dst = cv2.cornerHarris(g,2,3,0,0.04)
#SIFT
sift = cv2.SIFT()
kp = sift.detect(i,None)
i = cv2.drawKeypoints(kp,i)
print(i)


