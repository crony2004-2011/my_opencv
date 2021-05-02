import numpy as np
import cv2
import datetime
from datetime import datetime
import os
def nothing(x):
    pass
os.chdir(r'C:\Users\hp\Desktop\jupyter')
#cap = cv2.VideoCapture(r'video1.mp4',0)
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):
    ret, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for rect in faces:
        (x,y,w,h) = rect
        f = cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),2)
        roi = frame[y:y+h,x:x+w]

        cv2.imwrite('face.jpg', roi)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break
        cap.release()
        cv2.destroyAllWindows()
